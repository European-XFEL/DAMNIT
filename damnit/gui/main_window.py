import logging
import os
import re
import shelve
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from socket import gethostname

import h5py
import numpy as np
import xarray as xr
from kafka.errors import NoBrokersAvailable
from PyQt5 import QtCore, QtGui, QtSvg, QtWidgets
from PyQt5.Qsci import QsciLexerPython, QsciScintilla
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtQuick import QQuickWindow, QSGRendererInterface
from PyQt5.QtWebEngineWidgets import QWebEngineProfile
from PyQt5.QtWidgets import QAction, QFileDialog, QMessageBox, QTabWidget

from ..api import DataType, RunVariables
from ..backend import initialize_proposal
from ..backend.db import DamnitDB, MsgKind, ReducedData, db_path
from ..backend.extraction_control import ExtractionSubmitter, process_log_path
from ..backend.user_variables import UserEditableVariable
from ..definitions import UPDATE_BROKERS
from ..util import isinstance_no_import
from .editor import ContextTestResult, Editor, SaveConflictDialog
from .kafka import UpdateAgent
from .new_context_dialog import NewContextFileDialog
from .open_dialog import OpenDBDialog
from .plot import (ImagePlotWindow, PlottingControls, ScatterPlotWindow,
                   Xarray1DPlotWindow)
from .process import ProcessingDialog
from .standalone_comments import TimeComment
from .table import DamnitTableModel, TableView, prettify_notation
from .theme import Theme, ThemeManager, set_lexer_theme
from .user_variables import AddUserVariableDialog
from .util import icon_path, StatusbarStylesheet
from .web_viewer import PlotlyPlot, UrlSchemeHandler
from .zulip_messenger import ZulipMessenger

log = logging.getLogger(__name__)

class Settings(Enum):
    COLUMNS = "columns"
    THEME = "theme"

class MainWindow(QtWidgets.QMainWindow):

    context_dir_changed = QtCore.pyqtSignal(str)
    save_context_finished = QtCore.pyqtSignal(bool)  # True if saved
    check_context_file_timer = None
    vars_ctx_size_mtime = None
    editor_ctx_size_mtime = None
    current_theme = Theme.LIGHT

    db = None
    db_id = None
    _columns_dialog = None

    def __init__(self, context_dir: Path = None, connect_to_kafka: bool = True):
        super().__init__()

        self._connect_to_kafka = connect_to_kafka
        self._updates_thread = None
        self._received_update = False
        self._context_path = None
        self._context_is_saved = True
        self._context_code_to_save = None

        self._settings_db_path = Path.home() / ".local" / "state" / "damnit" / "settings.db"

        self.setWindowTitle("Data And Metadata iNspection Interactive Thing")
        self.setWindowIcon(QtGui.QIcon(icon_path("AMORE.png")))
        self._create_status_bar()

        # Load theme from settings
        self.current_theme = self._load_theme()
        self.apply_theme(self.current_theme)

        self._create_menu_bar()

        self._view_widget = QtWidgets.QWidget(self)
        self._editor = Editor()
        self._error_widget = QsciScintilla()
        self._editor_parent_widget = QtWidgets.QSplitter(Qt.Vertical)

        self._tab_widget = QTabWidget()
        self._tabbar_style = TabBarStyle()
        self._tab_widget.tabBar().setStyle(self._tabbar_style)
        self._tab_widget.addTab(self._view_widget, "Run table")
        self._tab_widget.addTab(self._editor_parent_widget, "Context file")
        self._tab_widget.currentChanged.connect(self.on_tab_changed)

        # Disable the main window at first since we haven't loaded any database yet
        self._tab_widget.setEnabled(False)
        self.setCentralWidget(self._tab_widget)

        self.save_context_finished.connect(self._save_context_finished)

        self.table = None

        self.zulip_messenger = None

        self._create_view()
        self.configure_editor()
        self.center_window()

        if context_dir is not None:
            self.autoconfigure(context_dir)

        self._canvas_inspect = []

        self.apply_theme(self.current_theme)

    def on_tab_changed(self, index):
        if index == 0:
            self._status_bar.showMessage("Double-click on a cell to inspect results.")
        elif index == 1:
            self._status_bar.showMessage(self._editor_status_message)

    def closeEvent(self, event):
        if not self._context_is_saved:
            dialog = QMessageBox(QMessageBox.Warning,
                                 "Warning - unsaved changes",
                                 "There are unsaved changes to the context, do you want to go back and save?",
                                 QMessageBox.Discard | QMessageBox.Cancel)
            result = dialog.exec()

            if result == QMessageBox.Cancel:
                event.ignore()
                return

        self.stop_update_listener_thread()
        self.stop_watching_context_file()
        super().closeEvent(event)

    def stop_update_listener_thread(self):
        if self._updates_thread is not None:
            self.update_agent.stop()
            self._updates_thread.exit()
            self._updates_thread.wait()
            self._updates_thread = None

    def center_window(self):
        """
        Center and resize the window to the screen the cursor is placed on.
        """
        screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())

        screen_size = screen.size()
        max_width = screen_size.width()
        max_height = screen_size.height()

        # Resize to a reasonable default
        self.resize(int(max_width * 0.8), int(max_height * 0.8))

        # Center window
        self.move(screen.geometry().center() - self.frameGeometry().center())

    def _create_status_bar(self) -> None:
        self._status_bar = QtWidgets.QStatusBar()

        self._status_bar.messageChanged.connect(self.on_status_message_changed)

        self._status_bar.setStyleSheet("QStatusBar::item {border: None;}")
        self._status_bar.showMessage("Autoconfigure AMORE.")
        self.setStatusBar(self._status_bar)

        self._status_bar_connection_status = QtWidgets.QLabel()
        self._status_bar.addPermanentWidget(self._status_bar_connection_status)

    def on_status_message_changed(self, msg):
        if msg == "":
            self.show_default_status_message()

    def show_status_message(self, message, timeout = 0, stylesheet = ''):
        if isinstance(stylesheet, StatusbarStylesheet):
            stylesheet = stylesheet.value

        self._status_bar.showMessage(message, timeout)
        self._status_bar.setStyleSheet(stylesheet)

    def show_default_status_message(self):
        self._status_bar.showMessage("Double-click on a cell to inspect results.")
        self._status_bar.setStyleSheet('QStatusBar {}')

    def _menu_create_user_var(self) -> None:
        dialog = AddUserVariableDialog(self)
        dialog.exec()

    def _menu_bar_help(self) -> None:
        dialog = QtWidgets.QMessageBox(self)

        font = QtGui.QFont()
        font.setBold(False)
        dialog.setFont(font)

        dialog.setWindowTitle("Getting help!")
        dialog.setText(
            """To start inspecting experimental results,
autoconfigure AMORE by selecting the proposal directory.

If you experience any issue, please contact us at:
da-dev@xfel.eu"""
        )
        dialog.exec()

    def _menu_bar_autoconfigure(self) -> None:
        open_dialog = OpenDBDialog(self)
        context_dir, prop_no = open_dialog.run_get_result()
        if context_dir is None:
            return
        if not prompt_setup_db(context_dir, prop_no, parent=self):
            # User said no to setting up a new database
            return

        self.autoconfigure(context_dir)

    def save_settings(self):
        self._settings_db_path.parent.mkdir(parents=True, exist_ok=True)

        with shelve.open(str(self._settings_db_path)) as db:
            settings = {
                Settings.COLUMNS.value: self.table_view.get_column_states(),
                Settings.THEME.value: self.current_theme.value
            }
            db[str(self._context_path)] = settings

    def autoconfigure(self, path: Path):
        sqlite_path = db_path(path)
        # If the user selected an empty folder in the GUI, the database has been
        # created before we reach this point, so this is just a sanity check.
        if not sqlite_path.is_file():
            QMessageBox.critical(self, "No DAMNIT database",
                                 "The selected folder doesn't contain a DAMNIT database (runs.sqlite)")
            return

        self.context_dir = path
        self._context_path = path / "context.py"
        self.extracted_data_template = str(path / "extracted_data/p{}_r{}.h5")

        log.info("Reading data from database")
        self.db = DamnitDB(sqlite_path)
        self.db_id = self.db.metameta['db_id']
        self.stop_update_listener_thread()
        self._updates_thread_launcher()

        self.reload_context()

        # Load the users settings
        col_settings = { }
        if self._settings_db_path.parent.is_dir():
            with shelve.open(str(self._settings_db_path)) as db:
                key = str(self._context_path)
                if key in db:
                    col_settings = db[key][Settings.COLUMNS.value]

        if self.table is not None:
            self.table.deleteLater()
        self.table = self._create_table_model(self.db, col_settings)
        self.table_view.setModel(self.table)
        self.table_view.sortByColumn(self.table.find_column("Timestamp", by_title=True),
                                     Qt.SortOrder.AscendingOrder)

        # Always keep these columns as small as possible to save space
        header = self.table_view.horizontalHeader()
        for column in ["Status", "Proposal", "Run", "Timestamp"]:
            column_index = self.table.find_column(column, by_title=True)
            header.setSectionResizeMode(column_index, QtWidgets.QHeaderView.ResizeToContents)
        header.setVisible(True)

        # Update the column widget and plotting controls with the new columns
        titles = self.table.column_titles
        self.table_view.set_columns(titles,
                                    [col_settings.get(col, True) for col in titles])
        self.plot.update_columns()

        self._tab_widget.setEnabled(True)
        self.show_default_status_message()
        self.context_dir_changed.emit(str(path))
        self.launch_update_computed_vars()
        self.start_watching_context_file()

    def _save_context_finished(self, saved):
        if saved:
            self.launch_update_computed_vars()

    def launch_update_computed_vars(self, ctx_size_mtime=None):
        # Triggered when we open a proposal & when saving the context file
        log.debug("Launching subprocess to read variables from context file")
        # Store the size & mtime before processing the file: better to capture
        # this just before a change and process the same version twice than
        # just after & potentially miss a change.
        self.vars_ctx_size_mtime = ctx_size_mtime or self.get_context_size_mtime()
        proc = QtCore.QProcess(parent=self)
        # Show stdout & stderr with the parent process
        proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.ForwardedChannels)
        proc.finished.connect(proc.deleteLater)
        proc.setWorkingDirectory(str(self.context_dir))
        proc.start(sys.executable, ['-m', 'damnit.cli', 'read-context'])
        proc.closeWriteChannel()
        # The subprocess will send updates for any changes: see .handle_update()

    def get_context_size_mtime(self):
        st = self._context_path.stat()
        return st.st_size, st.st_mtime

    def poll_context_file(self):
        size_mtime = self.get_context_size_mtime()
        if self.vars_ctx_size_mtime != size_mtime:
            log.info("Context file changed, updating computed variables")
            self.launch_update_computed_vars(size_mtime)

        if (self.editor_ctx_size_mtime != size_mtime) and self._context_is_saved:
            log.info("Context file changed, reloading editor")
            self.reload_context()

    def start_watching_context_file(self):
        self.stop_watching_context_file()  # Only 1 timer at a time

        self.check_context_file_timer = tmr = QtCore.QTimer(self)
        tmr.setInterval(30_000)
        tmr.timeout.connect(self.poll_context_file)
        tmr.start()

    def stop_watching_context_file(self):
        if self.check_context_file_timer is not None:
            self.check_context_file_timer.stop()
            self.check_context_file_timer.deleteLater()
            self.check_context_file_timer = None

    def add_variable(self, name, title, variable_type, description="", before=None):
        n_static_cols = self.table_view.get_static_columns_count()
        before_pos = n_static_cols
        if before == None:
            before_pos += self.table_view.get_movable_columns_count()
        else:
            before_pos += before
        variable = UserEditableVariable(name, title=title, variable_type=variable_type, description=description)
        self.table.user_variables[name] = variable
        self.db.add_user_variable(variable)
        self.table.insert_columns(
            before_pos, [title], [name], variable.get_type_class(), editable=True
        )
        self.table_view.add_new_columns([title], [True], [before_pos - n_static_cols - 1])
        self.table.add_editable_column(name)

        if self._connect_to_kafka:
            self.update_agent.variable_set(name, title, description, variable_type)

    def open_column_dialog(self):
        if self._columns_dialog is None:
            self._columns_dialog = QtWidgets.QDialog(self)
            self._columns_dialog.setWindowTitle("Column settings")
            layout = QtWidgets.QVBoxLayout()

            layout.addWidget(QtWidgets.QLabel("These columns can be hidden but not reordered or deleted:"))
            layout.addWidget(self.table_view._static_columns_widget)
            layout.addWidget(QtWidgets.QLabel("Drag these columns to reorder them, right-click to delete:"))
            layout.addWidget(self.table_view._columns_widget)
            self._columns_dialog.setLayout(layout)

        self._columns_dialog.show()

    def precreate_runs_dialog(self):
        n_runs, ok = QtWidgets.QInputDialog.getInt(self, "Pre-create new runs",
                                                   "Select how many runs to create in the database immediately:",
                                                   value=1, min=1)
        if ok:
            self.table.precreate_runs(n_runs)

    def _create_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)

        action_autoconfigure = QtWidgets.QAction(
            QtGui.QIcon("autoconfigure.png"), "Connect with &autoconfiguration", self
        )
        action_autoconfigure.setShortcut("Shift+A")
        action_autoconfigure.setStatusTip(
            "Autoconfigure AMORE by selecting the proposal folder."
        )
        action_autoconfigure.triggered.connect(self._menu_bar_autoconfigure)

        self.action_create_var = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("accessories-text-editor"),
            "&Create user variable",
            self
        )
        self.action_create_var.setShortcut("Shift+U")
        self.action_create_var.setStatusTip("Create user editable variable")
        self.action_create_var.triggered.connect(self._menu_create_user_var)
        self.action_create_var.setEnabled(False)
        self.context_dir_changed.connect(lambda _: self.action_create_var.setEnabled(True))

        self.action_export = QtWidgets.QAction(QtGui.QIcon(icon_path("export.png")), "&Export", self)
        self.action_export.setStatusTip("Export to Excel/CSV")
        self.action_export.setEnabled(False)
        self.context_dir_changed.connect(lambda _: self.action_export.setEnabled(True))
        self.action_export.triggered.connect(self.export_table)
        self.action_process = QtWidgets.QAction("Reprocess runs", self)
        self.action_process.triggered.connect(self.process_runs)

        action_adeqt = QtWidgets.QAction("Python console", self)
        action_adeqt.setShortcut("F12")
        action_adeqt.triggered.connect(self.show_adeqt)

        action_help = QtWidgets.QAction(QtGui.QIcon("help.png"), "&Help", self)
        action_help.setShortcut("Shift+H")
        action_help.setStatusTip("Get help.")
        action_help.triggered.connect(self._menu_bar_help)

        action_exit = QtWidgets.QAction(QtGui.QIcon("exit.png"), "&Exit", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.setStatusTip("Exit AMORE GUI.")
        action_exit.triggered.connect(QtWidgets.QApplication.instance().quit)

        fileMenu = menu_bar.addMenu(
            QtGui.QIcon(icon_path("AMORE.png")), "&AMORE"
        )
        fileMenu.addAction(action_autoconfigure)
        fileMenu.addAction(self.action_create_var)
        fileMenu.addAction(self.action_process)
        fileMenu.addAction(self.action_export)
        fileMenu.addAction(action_adeqt)
        fileMenu.addAction(action_help)
        fileMenu.addAction(action_exit)

        # Table menu
        action_columns = QtWidgets.QAction("Select, delete, && reorder columns", self)
        action_columns.triggered.connect(self.open_column_dialog)
        self.action_autoscroll = QtWidgets.QAction('Scroll to newly added runs', self)
        self.action_autoscroll.setCheckable(True)
        action_precreate_runs = QtWidgets.QAction("Pre-create new runs", self)
        action_precreate_runs.triggered.connect(self.precreate_runs_dialog)
        tableMenu = menu_bar.addMenu("Table")

        tableMenu.addAction(action_columns)
        tableMenu.addAction(self.action_autoscroll)
        tableMenu.addAction(action_precreate_runs)

        #jump to run
        menu_bar_right = QtWidgets.QMenuBar(self)
        searchMenu = menu_bar_right.addMenu(
            QtGui.QIcon(icon_path("search_icon.png")), "&Search Run")
        searchMenu.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.jump_search_run = QtWidgets.QLineEdit(self)
        self.jump_search_run.setPlaceholderText("Jump to run:")
        self.jump_search_run.setStyleSheet("width: 120px")
        self.jump_search_run.returnPressed.connect(lambda: self.scroll_to_run(
            self.jump_search_run.text()))
        actionWidget = QtWidgets.QWidgetAction(menu_bar)
        actionWidget.setDefaultWidget(self.jump_search_run)
        searchMenu.addAction(actionWidget)
        menu_bar.setCornerWidget(menu_bar_right, Qt.TopRightCorner)

        # Add View menu
        view_menu = self.menuBar().addMenu("View")

        # Add theme toggle action
        self.dark_mode_action = QAction("Dark Mode", self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(self.current_theme == Theme.DARK)
        self.dark_mode_action.setShortcut("Ctrl+Shift+D")
        self.dark_mode_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self.dark_mode_action)

    def scroll_to_run(self, run):
        try:
            run = int(run)
        except:
            log.info("Invalid input when searching run.")
            return

        proposal = self.db.metameta['proposal']
        try:
            index_row = self.table.find_row(proposal, run)
        except KeyError:
            log.info('p%d r%d not found when searching run', proposal, run)
            return
        self.scroll_to_row(index_row)

    def scroll_to_row(self, index_row):
        visible_column = self.table_view.columnAt(0)
        if visible_column == -1:
            visible_column = 0
        index = self.table_view.model().index(index_row, visible_column)

        self.table_view.scrollTo(index)
        self.table_view.selectRow(index.row())

    def on_rows_inserted(self, _modelix, first, _last):
        if self.action_autoscroll.isChecked():
            self.scroll_to_row(first)

    def export_table(self):
        export_path, file_type = QFileDialog.getSaveFileName(self, "Export table to file",
                                                             str(Path.home()),
                                                             "Excel (*.xlsx);;CSV (*.csv)")

        # If the user cancelled the dialog, return
        if len(export_path) == 0:
            return

        # Make sure the path ends with the right extension
        export_path = Path(export_path)
        if export_path.suffix == "":
            extension = file_type.split()[1][2:-1]
            export_path = export_path.with_suffix(extension)
        else:
            extension = export_path.suffix

        # Select columns in order of their appearance in the table (note: this
        # drops the comment_id column).
        columns = ["Status"] + list(self.table_view.get_column_states().keys())
        cleaned_df = self.table.dataframe_for_export(columns)

        if extension == ".xlsx":
            proposal = self.db.metameta["proposal"]
            cleaned_df.to_excel(export_path, sheet_name=f"p{proposal} DAMNIT run table")
        elif extension == ".csv":
            cleaned_df.to_csv(export_path, index=False)
        else:
            self.show_status_message(f"Unrecognized file extension: {extension}",
                                     stylesheet=StatusbarStylesheet.ERROR)

    def handle_update(self, message):
        if not self._received_update:
            self._received_update = True
            self._status_bar_connection_status.setStyleSheet(
                "color:green;font-weight:bold;"
            )
            self._status_bar_connection_status.setText(
                f"Getting updates ({self.db_id})"
            )

        if 'msg_kind' not in message:
            # Old message format. Temporarily handled so GUIs with new code can
            # work with listeners with older code, but can be removed soon.
            message = message.copy()
            proposal = message.pop("Proposal")
            run = message.pop("Run")
            message = {
                'msg_kind': MsgKind.run_values_updated.value,
                'data': {
                    'proposal': proposal,
                    'run': run,
                    'values': message
                }
            }

        msg_kind = MsgKind(message['msg_kind'])
        data = message['data']
        if msg_kind == MsgKind.run_values_updated:
            self.handle_run_values_updated(
                data['proposal'], data['run'], data['values']
            )
        elif msg_kind == MsgKind.variable_set:
            self.table.handle_variable_set(data)
            # update tag filtering
            self.table_view.apply_tag_filter(
                self.table_view._current_tag_filter
            )
        elif msg_kind == MsgKind.processing_state_set:
            self.table.handle_processing_state_set(data)
        elif msg_kind == MsgKind.processing_finished:
            self.table.handle_processing_finished(data)

    def handle_run_values_updated(self, proposal, run, values: dict):
        self.table.handle_run_values_changed(proposal, run, values)

        # update plots and plotting controls
        self.plot.update_columns()
        self.plot.update()


    def _updates_thread_launcher(self) -> None:
        if not self._connect_to_kafka:
            return

        assert self.db_id is not None

        try:
            self.update_agent = UpdateAgent(self.db_id)
        except NoBrokersAvailable:
            QtWidgets.QMessageBox.warning(self, "Broker connection failed",
                                          f"Could not connect to any Kafka brokers at: {' '.join(UPDATE_BROKERS)}\n\n" +
                                          "DAMNIT can operate offline, but it will not receive any updates from new or reprocessed runs.")
            return

        self._updates_thread = QtCore.QThread()
        self.update_agent.moveToThread(self._updates_thread)

        self._updates_thread.started.connect(self.update_agent.listen_loop)
        self.update_agent.message.connect(self.handle_update)
        QtCore.QTimer.singleShot(0, self._updates_thread.start)

    def get_run_file(self, proposal, run, log=True):
        file_name = self.extracted_data_template.format(proposal, run)

        try:
            run_file = h5py.File(file_name)
            return file_name, run_file
        except FileNotFoundError as e:
            if log:
                log.warning("{} not found...".format(file_name))
            raise e

    def col_title_to_name(self, title):
        return self.table.column_title_to_id(title)

    def _inspect_data_proxy_idx(self, index):
        # There is a 'proxy model' for sorting - we need to translate the index
        # to the real underlying model to look it up
        real_index = self.table_view.model().mapToSource(index)
        self.inspect_data(real_index)

    def inspect_data(self, index):
        proposal, run = self.table.row_to_proposal_run(index.row())
        if run is None:
            return   # Standalone comment row
        quantity_title = self.table.column_title(index.column())
        quantity = self.table.column_id(index.column())

        # Don't try to plot strings
        if quantity in { "Status" } | self.table.editable_columns:
            return

        log.info(
            "Selected proposal {} run {}, property {}".format(
                proposal, run, quantity_title
            )
        )

        try:
            preview = RunVariables(self.context_dir, run)[quantity].preview_data()
        except FileNotFoundError:
            self.show_status_message(f"Couldn't get run variables for p{proposal}, r{run}",
                                     timeout=7000,
                                     stylesheet=StatusbarStylesheet.ERROR)
            return
        except Exception:
            self.show_status_message(f"Error getting preview for '{quantity}'",
                                     timeout=7000,
                                     stylesheet=StatusbarStylesheet.ERROR)
            return

        if preview is None:
            self.show_status_message(f"No preview data found for variable {quantity}",
                                     timeout=7000)
            return

        if isinstance_no_import(preview, 'plotly.graph_objs', 'Figure'):
            pp = PlotlyPlot(self.context_dir, proposal, run, quantity, self)
            self._canvas_inspect.append(pp)
            pp.show()
            return

        if not isinstance(preview, (np.ndarray, xr.DataArray)):
            log.error("Only array objects are expected here, not %r", type(preview))
            return

        title = f'{quantity_title} (run {run})'

        data = preview.squeeze()

        if data.ndim == 1:
            if isinstance(data, xr.DataArray):
                try:
                    canvas = Xarray1DPlotWindow(self, data, title=title)
                except Exception as exc:
                    QMessageBox.warning(
                        self, f"Can't inspect variable {quantity}", str(exc))
                    return
            else:
                if data.dtype == np.bool_:
                    data = data.astype(np.float64)
                canvas = ScatterPlotWindow(self,
                    x=[np.arange(len(data))],
                    y=[data],
                    xlabel=f"Event (run {run})",
                    ylabel=quantity_title,
                    title=title,
            )
        elif data.ndim == 2 or (data.ndim == 3 and data.shape[-1] in (3, 4)):
            try:
                canvas = ImagePlotWindow(
                    self,
                    image=data,
                    title=f"{quantity_title} (run {run})",
                )
            except Exception as exc:
                QMessageBox.warning(
                    self, f"Can't inspect variable {quantity}", str(exc))
                return
        elif data.ndim == 0:
            # If this is a scalar value, then we can't plot it
            QMessageBox.warning(self, "Can't inspect variable",
                                f"'{quantity}' is a scalar, there's nothing more to plot.")
            return
        else:
            QMessageBox.warning(self, "Can't inspect variable",
                                f"'{quantity}' with {data.ndim} dimensions (not supported).")
            return

        self._canvas_inspect.append(canvas)
        canvas.show()

    def show_run_logs(self, proposal, run):
        # Triggered from right-click menu entry in table
        file = process_log_path(run, proposal, self.context_dir, create=False)
        if file.is_file():
            log_window = LogViewWindow(file, self)
            log_window.show()
            vsb = log_window.text_edit.verticalScrollBar()
            vsb.setValue(vsb.maximum())
        else:
            self.show_status_message(f"No log found for run {run}")

    def _create_table_model(self, db, col_settings):
        table = DamnitTableModel(db, col_settings, self)
        table.value_changed.connect(self.save_value)
        table.run_visibility_changed.connect(lambda row, state: self.plot.update())
        table.rowsInserted.connect(self.on_rows_inserted)
        return table

    def _create_view(self) -> None:
        vertical_layout = QtWidgets.QVBoxLayout()

        # Create toolbar for table controls
        toolbar = QtWidgets.QToolBar()
        vertical_layout.addWidget(toolbar)

        # the table
        self.table_view = TableView(self)
        self.table_view.doubleClicked.connect(self._inspect_data_proxy_idx)
        self.table_view.settings_changed.connect(self.save_settings)
        self.table_view.zulip_action.triggered.connect(self.export_selection_to_zulip)
        self.table_view.process_action.triggered.connect(self.process_runs)
        self.table_view.log_view_requested.connect(self.show_run_logs)

        # Initialize plot controls
        self.plot = PlottingControls(self)

        self.plot_dialog_button = QtWidgets.QPushButton("Plot")
        self.plot_dialog_button.clicked.connect(self.plot.show_dialog)
        self.comment_button = QtWidgets.QPushButton("Time comment")
        self.comment_button.clicked.connect(lambda: TimeComment(self).show())

        toolbar.addWidget(self.plot_dialog_button)
        toolbar.addWidget(self.comment_button)
        for widget in self.table_view.get_toolbar_widgets():
            toolbar.addWidget(widget)

        vertical_layout.addWidget(self.table_view)
        vertical_layout.setContentsMargins(0, 7, 0, 0)

        self._view_widget.setLayout(vertical_layout)

    def configure_editor(self):
        test_widget = QtWidgets.QWidget()

        self._editor.textChanged.connect(self.on_context_changed)
        self._editor.check_result.connect(self.test_context_result)

        vbox = QtWidgets.QGridLayout()
        test_widget.setLayout(vbox)

        self._save_btn = QtWidgets.QPushButton("Save")
        self._save_btn.clicked.connect(self.save_context)
        self._save_btn.setToolTip("Ctrl + S")
        self._save_btn.setShortcut(QtGui.QKeySequence(Qt.ControlModifier | Qt.Key_S))

        self._check_btn = QtWidgets.QPushButton("Validate")
        self._check_btn.clicked.connect(self.test_context)

        self._reload_btn = QtWidgets.QPushButton("Reload from disk")
        self._reload_btn.setToolTip("Reload the context file from disk")
        self._reload_btn.clicked.connect(self.reload_context)

        self._context_status_icon = QtSvg.QSvgWidget()
        self._context_status_icon.setMinimumSize(20, 20)

        font = QtGui.QFont("Monospace", pointSize=9)
        self._error_widget_lexer = QsciLexerPython()
        self._error_widget_lexer.setDefaultFont(font)

        self._error_widget.setReadOnly(True)
        self._error_widget.setCaretWidth(0)
        self._error_widget.setLexer(self._error_widget_lexer)

        vbox.addWidget(self._save_btn, 0, 0)
        vbox.addWidget(self._check_btn, 1, 0)
        vbox.addWidget(self._reload_btn, 2, 0)
        vbox.addWidget(self._context_status_icon, 3, 0)
        vbox.addWidget(self._error_widget, 0, 1, 4, 1)
        vbox.setColumnStretch(0, 1)
        vbox.setColumnStretch(1, 100)

        self._editor_parent_widget.addWidget(self._editor)
        self._editor_parent_widget.addWidget(test_widget)

    def on_context_changed(self):
        self._tabbar_style.enable_bold = True
        self._tab_widget.tabBar().setTabTextColor(1, QtGui.QColor("red"))
        self._tab_widget.setTabText(1, " Context file* ")
        self._editor_status_message = "Context file changed! Press Ctrl + S to save."
        self.on_tab_changed(self._tab_widget.currentIndex())
        self._context_is_saved = False

    def _ctx_contents_size_mtime(self):
        """Get the contents of the context file, plus its size & mtime"""
        # There's no way to do this atomically, so we stat the file before &
        # after reading and check that the results match.
        size_mtime_before = self.get_context_size_mtime()
        for _ in range(20):
            contents = self._context_path.read_text()
            size_mtime_after = self.get_context_size_mtime()
            if size_mtime_after == size_mtime_before:
                return contents, size_mtime_after

            size_mtime_before = size_mtime_after

        raise RuntimeError(
            "Could not get consistent filesystem metadata for context file"
        )

    def reload_context(self):
        if not self._context_path.is_file():
            self.show_status_message("No context.py file found")
            return
        contents, size_mtime = self._ctx_contents_size_mtime()
        self._editor.setText(contents)
        self.test_context()
        self.mark_context_saved(size_mtime)

    def test_context(self):
        self.set_error_icon('wait')
        self._editor.launch_test_context(self.db)

    def _maybe_write_context_file(self, check_ok):
        code_to_save = self._context_code_to_save
        self._context_code_to_save = None
        if check_ok:
            if self.editor_ctx_size_mtime != self.get_context_size_mtime():
                log.info("Context file has changed on disk & in editor")
                dlg = SaveConflictDialog(self, code_to_save)
                action = dlg.exec_get_action()
                if action == 'overwrite':
                    code_to_save = dlg.editor_code
                elif action == 'reload':
                    self.reload_context()
                    code_to_save = None
                else:  # Cancelled
                    code_to_save = None

            if code_to_save is not None:
                self._context_path.write_text(code_to_save)
                self.mark_context_saved()

        self.save_context_finished.emit(check_ok)

    def test_context_result(self, test_result, output, checked_code):
        # want_save, self._context_save_wanted = self._context_save_wanted, False
        if self._context_code_to_save == checked_code:
            self._maybe_write_context_file(test_result is not ContextTestResult.ERROR)

        if test_result == ContextTestResult.ERROR:
            self.set_error_widget_text(output)
            self.set_error_icon("red")

            # Resize the error window
            height_unit = self._editor_parent_widget.height() // 3
            self._editor_parent_widget.setSizes([2 * height_unit, height_unit])
        else:
            # Move the error widget down
            height_unit = self.height() // 7
            self._editor_parent_widget.setSizes([6 * height_unit, height_unit])

            if test_result == ContextTestResult.WARNING:
                self.set_error_widget_text(output)

                # We don't treat pyflakes warnings as fatal errors because sometimes
                # pyflakes reports valid but harmless problems, like unused
                # variables or unused imports.
                self.set_error_icon("yellow")
            elif test_result == ContextTestResult.OK:
                self.set_error_widget_text("Valid context.")
                self.set_error_icon("green")

        self._editor.setFocus()

    def set_error_icon(self, icon):
        self._context_status_icon.load(icon_path(f"{icon}_circle.svg"))
        self._context_status_icon.renderer().setAspectRatioMode(Qt.KeepAspectRatio)

    def set_error_widget_text(self, text):
        # Clear the widget and wait for a bit to visually indicate to the
        # user that something happened.
        self._error_widget.setText("")

        # We use sleep() instead of a QTimer because otherwise during the tests
        # the error widget may be free'd before the timer fires, leading to a
        # segfault when the timer function attempts to use it.
        import time
        time.sleep(0.1)
        self._error_widget.setText(text)

    def save_context(self):
        self._context_code_to_save = self._editor.text()
        self.test_context()
        # If the check passes, .test_context_result() saves the file

    def mark_context_saved(self, ctx_size_mtime=None):
        self._context_is_saved = True
        self._tabbar_style.enable_bold = False
        self._tab_widget.setTabText(1, "Context file")
        self._tab_widget.tabBar().setTabTextColor(1, QtGui.QColor("black"))
        self._editor_status_message = str(self._context_path.resolve())
        self.on_tab_changed(self._tab_widget.currentIndex())
        self.editor_ctx_size_mtime = ctx_size_mtime or self.get_context_size_mtime()

    def save_value(self, prop, run, name, value):
        if self.db is None:
            log.warning("No SQLite database in use, value not saved")
            return

        log.debug("Saving data for variable %s for prop %d run %d", name, prop, run)
        self.db.set_variable(prop, run, name, ReducedData(value))
        if self._connect_to_kafka:
            self.update_agent.run_values_updated(prop, run, name, value)

    def check_zulip_messenger(self):
        if not isinstance(self.zulip_messenger, ZulipMessenger):
            self.zulip_messenger = ZulipMessenger(self)

        if not self.zulip_messenger.ok:
            self.zulip_messenger = None
            return False
        return True

    def export_selection_to_zulip(self):
        if not self.check_zulip_messenger():
            log.warning("Unable to connect to Zulip to export table")
            return

        selected_rows = [ix.row() for ix in self.table_view.selected_rows()]

        blacklist_columns = ['Proposal', 'Status']
        columns = [title for (title, vis) in self.table_view.get_column_states().items()
                   if vis and (title not in blacklist_columns)]

        df = self.table.dataframe_for_export(columns, selected_rows, drop_image_cols=True)
        df.sort_values('Run', axis=0, inplace=True)

        df = df.map(prettify_notation)
        df.replace(["None", '<NA>', 'nan'], '', inplace=True)
        self.zulip_messenger.send_table(df)

    def process_runs(self):
        sel_runs_by_prop = {}
        for ix in self.table_view.selected_rows():
            run_prop, run_num = self.table.row_to_proposal_run(ix.row())
            sel_runs_by_prop.setdefault(run_prop, []).append(run_num)

        if sel_runs_by_prop:
            prop, sel_runs = max(sel_runs_by_prop.items(), key=lambda p: len(p[1]))
            sel_runs.sort()
        else:
            prop = self.db.metameta.get("proposal", "")
            sel_runs = []

        var_ids_titles = zip(self.table.computed_columns(),
                             self.table.computed_columns(by_title=True))

        dlg = ProcessingDialog(str(prop), sel_runs, var_ids_titles, parent=self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            submitter = ExtractionSubmitter(self.context_dir, self.db)

            try:
                reqs = dlg.extraction_requests()
                submitted = submitter.submit_multi(reqs)
            except Exception as e:
                log.error("Error launching processing", exc_info=True)
                self.show_status_message(f"Error launching processing: {e}",
                                         10_000, stylesheet=StatusbarStylesheet.ERROR)
            else:
                self.show_status_message(
                    f"Launched processing for {len(reqs)} runs", 10_000
                )
                if self._connect_to_kafka:
                    for req, (job_id, cluster) in zip(reqs, submitted):
                        self.update_agent.processing_submitted(
                            req.submitted_info(cluster, job_id)
                        )

    adeqt_window = None

    def show_adeqt(self):
        from adeqt import AdeqtWindow
        if self.adeqt_window is None:
            ns = {'window': self, 'table': self.table, 'editor': self._editor}
            self.adeqt_window = AdeqtWindow(ns, parent=self)
        self.adeqt_window.show()

    def _toggle_theme(self, checked):
        """Toggle between light and dark themes."""
        new_theme = Theme.DARK if checked else Theme.LIGHT
        self.apply_theme(new_theme)

    def apply_theme(self, theme: Theme):
        """Apply the selected theme to the application."""
        self.current_theme = theme
        self._save_theme(theme)

        app = QtWidgets.QApplication.instance()

        # Apply palette
        app.setPalette(ThemeManager.get_theme_palette(theme))

        # Apply stylesheet
        app.setStyleSheet(ThemeManager.get_theme_stylesheet(theme))

        # Update status bar style
        self._status_bar.setStyleSheet("QStatusBar::item {border: None;}")

        # Update editor theme
        if hasattr(self, '_editor'):
            self._editor.update_theme(theme)

        # Update error widget lexer theme
        if hasattr(self, '_error_widget_lexer'):
            set_lexer_theme(self._error_widget_lexer, self.current_theme)

        # Update plot windows
        if hasattr(self, '_canvas_inspect'):
            for window in self._canvas_inspect:
                if window.isVisible():
                    window.update_theme(theme)

    def _load_theme(self):
        """Load theme setting from shelve file."""
        if self._settings_db_path.parent.is_dir():
            with shelve.open(str(self._settings_db_path)) as settings:
                try:
                    theme_name = settings.get(Settings.THEME.value)
                    if theme_name is not None:
                        return Theme(theme_name)
                except (ValueError, KeyError):
                    pass
        return Theme.LIGHT

    def _save_theme(self, theme: Theme):
        """Save theme setting to shelve file."""
        if self._settings_db_path.parent.is_dir():
            with shelve.open(str(self._settings_db_path)) as settings:
                settings[Settings.THEME.value] = theme.value


class TableViewStyle(QtWidgets.QProxyStyle):
    """
    Subclass that enables instant tooltips for widgets in a TableView.
    """
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QtWidgets.QStyle.SH_ToolTip_WakeUpDelay \
           and isinstance(widget.parent(), TableView):
            return 0
        else:
            return super().styleHint(hint, option, widget, returnData)


class TabBarStyle(QtWidgets.QProxyStyle):
    """
    Subclass that enables bold tab text for tab 1 (the editor tab).
    """
    def __init__(self):
        super().__init__()
        self.enable_bold = False

    def drawControl(self, element, option, painter, widget=None):
        if self.enable_bold and \
           element == QtWidgets.QStyle.CE_TabBarTab and \
           widget.tabRect(1) == option.rect:
            font = widget.font()
            font.setBold(True)
            painter.save()
            painter.setFont(font)
            super().drawControl(element, option, painter, widget)
            painter.restore()
        else:
            super().drawControl(element, option, painter, widget)


class LogViewWindow(QtWidgets.QMainWindow):
    def __init__(self, file_path: Path, parent=None, polling_interval_ms: int = 1500):
        super().__init__(parent)
        self.file_path = file_path
        self.text_edit = QtWidgets.QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        font = self.text_edit.document().defaultFont()
        font.setFamily('monospace')
        self.text_edit.document().setDefaultFont(font)
        self.setCentralWidget(self.text_edit)
        self.resize(1000, 800)

        # Read initial file content and keep track of the file size to only append new content
        self.text_edit.setPlainText(self.file_path.read_text(encoding='utf-8', errors='replace'))
        stat_res = self.file_path.stat()
        self._last_mtime = stat_res.st_mtime
        self._last_size = stat_res.st_size

        # Poll for file changes using a timer
        self._timer = QTimer(self)
        self._timer.setInterval(polling_interval_ms)
        self._timer.timeout.connect(self._check_log_file_update)
        self._timer.start()

    def _check_log_file_update(self):
        """Poll the file for changes in mtime or size."""
        try:
            stat_res = self.file_path.stat()
        except FileNotFoundError:
            # File might have been deleted
            if self._last_size is not None:  # Avoid repeated logging
                log.warning(f"Log file {self.file_path} not found.", exc_info=True)
                self.text_edit.setPlainText(f"[Log file {self.file_path} not found or deleted]")
                self._last_size = None
                self._last_mtime = None
            return

        if stat_res.st_mtime != self._last_mtime or stat_res.st_size != self._last_size:
            self._read_and_update_log(stat_res)

    def _read_and_update_log(self, current_stat):
        """Read changes from the log file and update the text edit."""
        current_size = current_stat.st_size
        self._last_mtime = current_stat.st_mtime

        if self._last_size is None or current_size < self._last_size:
            # File might have been truncated, replaced, or reappeared
            try:
                new_text = self.file_path.read_text(encoding='utf-8', errors='replace')
                self.text_edit.setPlainText(new_text)
                self._last_size = current_size
            except Exception:
                log.error(f"Error reading log file {self.file_path}", exc_info=True)
                self.text_edit.setPlainText(f"[Error reading log file {self.file_path}]")
                self._last_size = None
                self._last_mtime = None

        elif current_size > self._last_size:
            # Append new content
            try:
                with self.file_path.open("rb") as f:
                    f.seek(self._last_size)
                    new_content = f.read()
                self.text_edit.appendPlainText(new_content.decode('utf-8', errors='replace'))
                self._last_size = current_size
            except Exception:
                log.error(f"Error reading log file {self.file_path}", exc_info=True)

    def closeEvent(self, event):
        # Stop watching the file when the window closes
        self._timer.stop()
        super().closeEvent(event)


def prompt_setup_db(context_dir: Path, prop_no=None, parent=None):
    if not db_path(context_dir).is_file():

        button = QMessageBox.question(
            parent, "Database not found",
            f"{context_dir} does not contain a DAMNIT database, "
            "would you like to create one?"
        )
        if button != QMessageBox.Yes:
            return False

        if not (context_dir / 'context.py').is_file():
            new_ctx_dialog = NewContextFileDialog(context_dir, parent)
            context_file_src, user_vars_src = new_ctx_dialog.run_get_result()
            if context_file_src is None:
                return False
        else:
            context_file_src = user_vars_src = None

        if prop_no is None:
            prop_no, ok = QtWidgets.QInputDialog.getInt(
                parent, "Select proposal", "Which proposal is this for?"
            )
            if not ok:
                return False
        initialize_proposal(context_dir, prop_no, context_file_src, user_vars_src)
        return True

    return True


def run_app(context_dir, software_opengl=False, connect_to_kafka=True):
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar,
    )

    # Required for the WebViewer to load pages
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--no-sandbox'

    if software_opengl or re.match(r'^max-exfl\d{3}.desy.de$', gethostname()):
        log.info('Use software OpenGL.')
        QtWidgets.QApplication.setAttribute(
            Qt.AA_UseSoftwareOpenGL
        )
        QQuickWindow.setSceneGraphBackend(QSGRendererInterface.Software)

    application = QtWidgets.QApplication(sys.argv)
    application.setStyle(TableViewStyle())

    if context_dir is None:
        open_dialog = OpenDBDialog()
        context_dir, prop_no = open_dialog.run_get_result()
        if context_dir is None:
            return 0
        if not prompt_setup_db(context_dir, prop_no):
            # User said no to setting up a new database
            return 0

    # configure webviewer url engine
    scheme_handler = UrlSchemeHandler(parent=application)
    profile = QWebEngineProfile.defaultProfile()
    scheme_handler.install(profile)

    window = MainWindow(context_dir=context_dir, connect_to_kafka=connect_to_kafka)
    window.show()
    return application.exec()


def main():
    ap = ArgumentParser()
    ap.add_argument(
        "context_dir", type=Path, nargs="?", help="Directory storing summarised results"
    )
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    sys.exit(run_app(args.context_dir))


if __name__ == "__main__":
    main()
