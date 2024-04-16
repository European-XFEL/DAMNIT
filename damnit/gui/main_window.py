import logging
import shelve
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from enum import Enum
from pathlib import Path
from socket import gethostname

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from pandas.api.types import infer_dtype
from PyQt5 import QtCore, QtGui, QtSvg, QtWidgets
from PyQt5.Qsci import QsciLexerPython, QsciScintilla
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTabWidget

from kafka.errors import NoBrokersAvailable

from ..backend import backend_is_running, initialize_and_start_backend
from ..api import RunVariables
from ..backend.db import BlobTypes, DamnitDB, MsgKind, ReducedData, db_path
from ..backend.extract_data import get_context_file, process_log_path
from ..backend.user_variables import UserEditableVariable
from ..definitions import UPDATE_BROKERS
from ..util import StatusbarStylesheet, fix_data_for_plotting, icon_path
from .editor import ContextTestResult, Editor
from .kafka import UpdateAgent
from .open_dialog import OpenDBDialog
from .plot import Canvas, Plot
from .table import DamnitTableModel, TableView, prettify_notation
from .user_variables import AddUserVariableDialog
from .widgets import CollapsibleWidget
from .zulip_messenger import ZulipMessenger

log = logging.getLogger(__name__)

class Settings(Enum):
    COLUMNS = "columns"

class MainWindow(QtWidgets.QMainWindow):

    context_dir_changed = QtCore.pyqtSignal(str)

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

        self._settings_db_path = Path.home() / ".local" / "state" / "damnit" / "settings.db"

        self.setWindowTitle("Data And Metadata iNspection Interactive Thing")
        self.setWindowIcon(QtGui.QIcon(icon_path("AMORE.png")))
        self._create_status_bar()
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

        self.table = None
        
        self.zulip_messenger = None

        self._create_view()
        self.configure_editor()
        self.center_window()

        if context_dir is not None:
            self.autoconfigure(context_dir)

        self._canvas_inspect = []

    def on_tab_changed(self, index):
        if index == 0:
            self._status_bar.showMessage("Double-click on a cell to inspect results.")
        elif index == 1:
            self._status_bar.showMessage(self._editor_status_message)

    def closeEvent(self, event):
        if not self._context_is_saved:
            dialog = QMessageBox(QMessageBox.Warning,
                                 "Warning - unsaved changes",
                                 "There are unsaved changes to the context, do you want to save before exiting?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            result = dialog.exec()

            if result == QMessageBox.Save:
                self.save_context()
            elif result == QMessageBox.Cancel:
                event.ignore()
                return

        self.stop_update_listener_thread()
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

        self._status_bar.messageChanged.connect(lambda m: self.show_default_status_message() if m == "" else m)

        self._status_bar.setStyleSheet("QStatusBar::item {border: None;}")
        self._status_bar.showMessage("Autoconfigure AMORE.")
        self.setStatusBar(self._status_bar)

        self._status_bar_connection_status = QtWidgets.QLabel()
        self._status_bar.addPermanentWidget(self._status_bar_connection_status)

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
        if not prompt_setup_db_and_backend(context_dir, prop_no, parent=self):
            # User said no to setting up a new database
            return

        self.autoconfigure(context_dir)

    def save_settings(self):
        self._settings_db_path.parent.mkdir(parents=True, exist_ok=True)

        with shelve.open(str(self._settings_db_path)) as db:
            settings = { Settings.COLUMNS.value: self.table_view.get_column_states() }
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

    def launch_update_computed_vars(self):
        log.debug("Launching subprocess to read variables from context file")
        proc = QtCore.QProcess(parent=self)
        # Show stdout & stderr with the parent process
        proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.ForwardedChannels)
        proc.finished.connect(proc.deleteLater)
        proc.setWorkingDirectory(str(self.context_dir))
        proc.start(sys.executable, ['-m', 'damnit.cli', 'read-context'])
        proc.closeWriteChannel()

    def add_variable(self, name, title, variable_type, description="", before=None):
        n_static_cols = self.table_view.get_static_columns_count()
        before_pos = n_static_cols + 1
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

        action_create_var = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("accessories-text-editor"),
            "&Create user variable",
            self
        )
        action_create_var.setShortcut("Shift+U")
        action_create_var.setStatusTip("Create user editable variable")
        action_create_var.triggered.connect(self._menu_create_user_var)
        action_create_var.setEnabled(False)
        self.context_dir_changed.connect(lambda _: action_create_var.setEnabled(True))

        action_export = QtWidgets.QAction(QtGui.QIcon(icon_path("export.png")), "&Export", self)
        action_export.setStatusTip("Export to Excel/CSV")
        action_export.setEnabled(False)
        self.context_dir_changed.connect(lambda _: action_export.setEnabled(True))
        action_export.triggered.connect(self.export_table)

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
        fileMenu.addAction(action_create_var)
        fileMenu.addAction(action_export)
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

    def _set_comment_date(self):
        self.comment_time.setText(
            time.strftime("%H:%M %d/%m/%Y", time.localtime(time.time()))
        )

    def _comment_button_clicked(self):
        ts = datetime.strptime(self.comment_time.text(), "%H:%M %d/%m/%Y").timestamp()
        text = self.comment.text()
        comment_id = self.db.add_standalone_comment(ts, text)
        self.table.insert_comment_row(comment_id, text, ts)
        self.comment.clear()

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

        cell_data = self.table.get_value_at(index)
        is_image = self.table.itemFromIndex(index).data(Qt.DecorationRole) is not None

        if not (is_image or isinstance(cell_data, (int, float))):
            QMessageBox.warning(self, "Can't inspect variable",
                                f"'{quantity}' has type '{type(cell_data).__name__}', cannot inspect.")
            return

        try:
            variable = RunVariables(self._context_path.parent, run)[quantity]
        except KeyError:
            log.warning(f"Unrecognized variable: '{quantity}'")
            return

        try:
            data = xr.DataArray(variable.read())
        except KeyError:
            log.warning(f'"{quantity}" not found in {variable.file}...')
            return

        if data.ndim == 2 or (data.ndim == 3 and data.shape[-1] in (3, 4)):
            canvas = Canvas(
                self,
                image=data.data,
                title=f"{quantity_title} (run {run})",
            )
        else:
            if data.ndim == 0:
                # If this is a scalar value, then we can't plot it
                QMessageBox.warning(self, "Can't inspect variable",
                                    f"'{quantity}' is a scalar, there's nothing more to plot.")
                return
            if data.ndim > 2:
                QMessageBox.warning(self, "Can't inspect variable",
                                    f"'{quantity}' with {data.ndim} dimensions (not supported).")
                return

            # Use the train ID if it's been saved, otherwise generate an X axis
            if "trainId" in data.coords:
                x = data.trainId
            else:
                x = np.arange(len(data))

            canvas = Canvas(
                self,
                x=[fix_data_for_plotting(x)],
                y=[fix_data_for_plotting(data)],
                xlabel=f"Event (run {run})",
                ylabel=quantity_title,
                fmt="o",
            )


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
        table.time_comment_changed.connect(self.save_time_comment)
        table.run_visibility_changed.connect(lambda row, state: self.plot.update())
        table.rowsInserted.connect(self.on_rows_inserted)
        return table

    def _create_view(self) -> None:
        vertical_layout = QtWidgets.QVBoxLayout()
        comment_horizontal_layout = QtWidgets.QHBoxLayout()

        # the table
        self.table_view = TableView()

        self.table_view.doubleClicked.connect(self._inspect_data_proxy_idx)
        self.table_view.settings_changed.connect(self.save_settings)
        self.table_view.zulip_action.triggered.connect(self.export_selection_to_zulip)
        self.table_view.log_view_requested.connect(self.show_run_logs)

        vertical_layout.addWidget(self.table_view)

        # add all other widgets on a collapsible layout
        collapsible = CollapsibleWidget()
        vertical_layout.addWidget(collapsible)

        # comments
        self.comment = QtWidgets.QLineEdit(self)
        self.comment.setText("Time can be edited in the field on the right.")

        self.comment_time = QtWidgets.QLineEdit(self)
        self.comment_time.setStyleSheet("width: 25px;")

        comment_button = QtWidgets.QPushButton("Additional comment")
        comment_button.setEnabled(True)
        comment_button.clicked.connect(self._comment_button_clicked)

        comment_horizontal_layout.addWidget(comment_button)
        comment_horizontal_layout.addWidget(self.comment, stretch=3)
        comment_horizontal_layout.addWidget(QtWidgets.QLabel("at"))
        comment_horizontal_layout.addWidget(self.comment_time, stretch=1)

        collapsible.add_layout(comment_horizontal_layout)

        comment_timer = QtCore.QTimer()
        self._set_comment_date()
        comment_timer.setInterval(30000)
        comment_timer.timeout.connect(self._set_comment_date)
        comment_timer.start()

        # plotting control
        self.plot = Plot(self)
        plotting_group = QtWidgets.QGroupBox("Plotting controls")
        plot_vertical_layout = QtWidgets.QVBoxLayout()
        plot_horizontal_layout = QtWidgets.QHBoxLayout()
        plot_parameters_horizontal_layout = QtWidgets.QHBoxLayout()

        plot_horizontal_layout.addWidget(self.plot._button_plot)
        self.plot._button_plot_runs.setMinimumWidth(200)
        plot_horizontal_layout.addStretch()

        plot_horizontal_layout.addWidget(QtWidgets.QLabel("Y:"))
        plot_horizontal_layout.addWidget(self.plot._combo_box_y_axis)
        plot_horizontal_layout.addWidget(self.plot.vs_button)
        plot_horizontal_layout.addWidget(QtWidgets.QLabel("X:"))
        plot_horizontal_layout.addWidget(self.plot._combo_box_x_axis)

        plot_vertical_layout.addLayout(plot_horizontal_layout)

        plot_parameters_horizontal_layout.addWidget(self.plot._button_plot_runs)
        self.plot._button_plot.setMinimumWidth(200)
        plot_parameters_horizontal_layout.addStretch()

        plot_parameters_horizontal_layout.addWidget(
            self.plot._toggle_probability_density
        )

        plot_vertical_layout.addLayout(plot_parameters_horizontal_layout)

        plotting_group.setLayout(plot_vertical_layout)
        
        collapsible.add_widget(plotting_group)

        vertical_layout.setSpacing(0)
        vertical_layout.setContentsMargins(0, 0, 0, 0)
        self._view_widget.setLayout(vertical_layout)

    def configure_editor(self):
        test_widget = QtWidgets.QWidget()

        self._editor.textChanged.connect(self.on_context_changed)

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

    def reload_context(self):
        if not self._context_path.is_file():
            self.show_status_message("No context.py file found")
            return
        self._editor.setText(self._context_path.read_text())
        self.test_context()
        self.mark_context_saved()

    def test_context(self):
        test_result, output = self._editor.test_context(self.db, self._context_path.parent)

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
        return test_result

    def set_error_icon(self, icon):
        self._context_status_icon.load(icon_path(f"{icon}_circle.svg"))
        self._context_status_icon.renderer().setAspectRatioMode(Qt.KeepAspectRatio)

    def set_error_widget_text(self, text):
        # Clear the widget and wait for a bit to visually indicate to the
        # user that something happened.
        self._error_widget.setText("")
        QtCore.QTimer.singleShot(100, lambda: self._error_widget.setText(text))

    def save_context(self):
        if self.test_context() == ContextTestResult.ERROR:
            return

        self._context_path.write_text(self._editor.text())
        self.mark_context_saved()
        self._editor.setFocus()

    def mark_context_saved(self):
        self._context_is_saved = True
        self._tabbar_style.enable_bold = False
        self._tab_widget.setTabText(1, "Context file")
        self._tab_widget.tabBar().setTabTextColor(1, QtGui.QColor("black"))
        self._editor_status_message = str(self._context_path.resolve())
        self.on_tab_changed(self._tab_widget.currentIndex())

    def save_value(self, prop, run, name, value):
        if self.db is None:
            log.warning("No SQLite database in use, value not saved")
            return

        log.debug("Saving data for variable %s for prop %d run %d", name, prop, run)
        self.db.set_variable(prop, run, name, ReducedData(value))
        if self._connect_to_kafka:
            self.update_agent.run_values_updated(prop, run, name, value)

    def save_time_comment(self, comment_id, value):
        if self.db is None:
            log.warning("No SQLite database in use, comment not saved")
            return

        log.debug("Saving time-based comment ID %d", comment_id)
        self.db.change_standalone_comment(comment_id, value)
        
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

        df = df.applymap(prettify_notation)
        df.replace(["None", '<NA>', 'nan'], '', inplace=True)
        self.zulip_messenger.send_table(df)

    adeqt_window = None

    def show_adeqt(self):
        from adeqt import AdeqtWindow
        if self.adeqt_window is None:
            ns = {'window': self, 'table': self.table}
            self.adeqt_window = AdeqtWindow(ns, parent=self)
        self.adeqt_window.show()

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
    def __init__(self, file_path: Path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.text_edit = QtWidgets.QPlainTextEdit(file_path.read_text())
        self.text_edit.setReadOnly(True)
        font = self.text_edit.document().defaultFont()
        font.setFamily('monospace')
        self.text_edit.document().setDefaultFont(font)
        self.setCentralWidget(self.text_edit)
        self.resize(1000, 800)


def prompt_setup_db_and_backend(context_dir: Path, prop_no=None, parent=None):
    if not db_path(context_dir).is_file():

        button = QMessageBox.question(
            parent, "Database not found",
            f"{context_dir} does not contain a DAMNIT database, "
            "would you like to create one and start the backend?"
        )
        if button != QMessageBox.Yes:
            return False


        if prop_no is None:
            prop_no, ok = QtWidgets.QInputDialog.getInt(
                parent, "Select proposal", "Which proposal is this for?"
            )
            if not ok:
                return False
        initialize_and_start_backend(context_dir, prop_no)

    # Check if the backend is running
    elif not backend_is_running(context_dir):
        button = QMessageBox.question(
            parent, "Backend not running",
            "The DAMNIT backend is not running, would you like to start it? "
            "This is only necessary if new runs are expected."
        )
        if button == QMessageBox.Yes:
            initialize_and_start_backend(context_dir, prop_no)

    return True


def run_app(context_dir, connect_to_kafka=True):
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar
    )
    application = QtWidgets.QApplication(sys.argv)
    application.setStyle(TableViewStyle())

    if context_dir is None:
        open_dialog = OpenDBDialog()
        context_dir, prop_no = open_dialog.run_get_result()
        if context_dir is None:
            return 0
        if not prompt_setup_db_and_backend(context_dir, prop_no):
            # User said no to setting up a new database
            return 0

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
