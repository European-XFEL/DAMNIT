import pickle
import os
import logging
import shutil
import sys
import time
import traceback
from argparse import ArgumentParser
from datetime import datetime
from io import StringIO
from pathlib import Path
from subprocess import Popen

import libtmux
import pandas as pd
import numpy as np
import h5py

from pyflakes.reporter import Reporter
from pyflakes.api import check as pyflakes_check

from kafka.errors import NoBrokersAvailable
from extra_data.read_machinery import find_proposal

from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTabWidget, QShortcut
from PyQt5.Qsci import QsciScintilla, QsciLexerPython, QsciCommand

from ..backend.db import open_db, get_meta, set_meta
from ..context import ContextFile
from ..definitions import UPDATE_BROKERS
from .kafka import UpdateReceiver
from .table import TableView, Table
from .plot import Canvas, Plot


log = logging.getLogger(__name__)
pd.options.mode.use_inf_as_na = True


class MainWindow(QtWidgets.QMainWindow):
    context_path = None
    db = None
    db_id = None

    context_dir_changed = QtCore.pyqtSignal(str)

    def __init__(self, context_dir: Path = None, connect_to_kafka: bool = True):
        super().__init__()

        self.data = None
        self._connect_to_kafka = connect_to_kafka
        self._updates_thread = None
        self._updates_thread = None
        self._received_update = False
        self._context_path = None
        self._context_is_saved = True
        self._attributi = {}

        self.setWindowTitle("Data And Metadata iNspection Interactive Thing")
        self.setWindowIcon(QtGui.QIcon("amore_mid_prototype/gui/ico/AMORE.png"))
        self._create_status_bar()
        self._create_menu_bar()

        self._view_widget = QtWidgets.QWidget(self)
        self._editor = QsciScintilla()
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
                                 QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            result = dialog.exec()

            if result == QMessageBox.Yes:
                self.save_context()
            elif result == QMessageBox.Cancel:
                event.ignore()
                return

        self.stop_update_listener_thread()
        super().closeEvent(event)

    def stop_update_listener_thread(self):
        if self._updates_thread is not None:
            self.update_receiver.stop()
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

        self._status_bar.setStyleSheet("QStatusBar::item {border: None;}")
        self._status_bar.showMessage("Autoconfigure AMORE.")
        self.setStatusBar(self._status_bar)

        self._status_bar_connection_status = QtWidgets.QLabel()
        self._status_bar.addPermanentWidget(self._status_bar_connection_status)

    def _menu_bar_edit_context(self):
        Popen(['xdg-open', self._context_path])

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
        proposal_dir = ""

        # If we're on a system with access to GPFS, prompt for the proposal
        # number so we can preset the prompt for the AMORE directory.
        if self.gpfs_accessible():
            prompt = True
            while prompt:
                prop_no, prompt = QtWidgets.QInputDialog.getInt(self, "Select proposal",
                                                                "Which proposal is this for?")
                if not prompt:
                    break

                proposal = f"p{prop_no:06}"
                try:
                    proposal_dir = find_proposal(proposal)
                    prompt = False
                except Exception:
                    button = QtWidgets.QMessageBox.warning(self, "Bad proposal number",
                                                           "Could not find a proposal with this number, try again?",
                                                           buttons=QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                    if button != QtWidgets.QMessageBox.Yes:
                        prompt = False
        else:
            prop_no = None

        # By convention the AMORE directory is often stored at usr/Shared/amore,
        # so if this directory exists, then we use it.
        standard_path = Path(proposal_dir) / "usr/Shared/amore"
        if standard_path.is_dir() and self.db_path(standard_path).is_file():
            path = standard_path
        else:
            # Helper lambda to open a prompt for the user
            prompt_for_path = lambda: QFileDialog.getExistingDirectory(self,
                                                                       "Select context directory",
                                                                       proposal_dir)

            if self.gpfs_accessible() and prop_no is not None:
                button = QMessageBox.question(self, "Database not found",
                                              f"Proposal {prop_no} does not have an AMORE database, " \
                                              "would you like to create one and start the backend?")
                if button == QMessageBox.Yes:
                    self.initialize_database(standard_path, prop_no)
                    path = standard_path
                else:
                    # Otherwise, we prompt the user
                    path = prompt_for_path()
            else:
                path = prompt_for_path()

        # If we found a database, make sure we're working with a Path object
        if path:
            path = Path(path)
        else:
            # Otherwise just return
            return

        # Check if the backend is running
        tmux_socket_path = self.get_tmux_socket_path(path)
        if not tmux_socket_path.exists():
            button = QMessageBox.question(self, "Backend not running",
                                          "The AMORE backend is not running, would you like to start it? " \
                                          "This is only necessary if new runs are expected.")
            if button == QMessageBox.Yes:
                self.start_backend(path)

        self.autoconfigure(Path(path), proposal=prop_no)

    def gpfs_accessible(self):
        return os.path.isdir("/gpfs/exfel/exp")

    def db_path(self, root_path: Path):
        return root_path / "runs.sqlite"

    def get_tmux_socket_path(self, root_path: Path):
        return root_path / "amore-tmux.sock"

    def initialize_database(self, path, proposal):
        # Ensure the directory exists
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o777)

        # Initialize database
        db = open_db(self.db_path(path))
        set_meta(db, 'proposal', proposal)

        # Copy initial context file
        context_path = path / "context.py"
        shutil.copyfile(Path(__file__).parents[1] / "base_context_file.py", context_path)
        os.chmod(context_path, 0o666)

        self.start_backend(path)

    def start_backend(self, path: Path):
        # Create tmux session
        server = libtmux.Server(socket_path=self.get_tmux_socket_path(path))
        server.new_session(session_name="AMORE",
                           window_name="listen",
                           # Unfortunately Maxwell's default tmux is too old to
                           # support '-c' to change directory, so we have to
                           # manually change directory instead of using the
                           # 'start_directory' argument.
                           window_command=f"cd {str(path)}; amore-proto listen .")

    def autoconfigure(self, path: Path, proposal=None):
        self._context_path = path / "context.py"
        if self._context_path.is_file():
            log.info("Reading context file %s", self._context_path)
            ctx_file = ContextFile.from_py_file(self._context_path)
            self._attributi = ctx_file.vars

            self._editor.setText(ctx_file.code)
            self.test_context()
            self.mark_context_saved()

        self.extracted_data_template = str(path / "extracted_data/p{}_r{}.h5")

        sqlite_path = self.db_path(path)
        log.info("Reading data from database")
        self.db = open_db(sqlite_path)

        self.db_id = get_meta(self.db, 'db_id')
        self.stop_update_listener_thread()
        self._updates_thread_launcher()

        df = pd.read_sql_query("SELECT * FROM runs", self.db)
        df.insert(0, "Status", True)
        df.insert(len(df.columns), "comment_id", pd.NA)
        df.pop("added_at")

        # Unpickle serialized objects. First we select all columns that
        # might need deserializing.
        object_cols = df.select_dtypes(include=["object"]).drop(["comment", "comment_id"], axis=1)
        # Then we check each element and unpickle it if necessary, and
        # finally update the main DataFrame.
        unpickled_cols = object_cols.applymap(lambda x: pickle.loads(x) if isinstance(x, bytes) else x)
        df.update(unpickled_cols)

        # Read the comments and prepare them for merging with the main data
        comments_df = pd.read_sql_query(
            "SELECT rowid as comment_id, * FROM time_comments", self.db
        )
        comments_df.insert(0, "Run", pd.NA)
        comments_df.insert(1, "Proposal", pd.NA)
        # Don't try to plot comments
        comments_df.insert(2, "Status", False)

        self.data = pd.concat(
            [
                df.rename(
                    columns={
                        "runnr": "Run",
                        "proposal": "Proposal",
                        "start_time": "Timestamp",
                        "comment": "Comment",
                        **self.column_renames(),
                    }
                ),
                comments_df.rename(
                    columns={"timestamp": "Timestamp", "comment": "Comment",}
                ),
            ]
        )

        self.table_view.setModel(self.table)
        self.table_view.sortByColumn(self.data.columns.get_loc("Timestamp"),
                                     Qt.SortOrder.AscendingOrder)

        # Always keep these columns as small as possible to save space
        header = self.table_view.horizontalHeader()
        for column in ["Status", "Proposal", "Run", "Timestamp"]:
            column_index = self.data.columns.get_loc(column)
            header.setSectionResizeMode(column_index, QtWidgets.QHeaderView.ResizeToContents)

        # Update the column widget and plotting controls with the new columns
        self.table_view.set_columns([self.column_title(c) for c in self.data.columns],
                                    [True for _ in self.data.columns])
        self.plot.update_columns()

        # Hide the comment_id column
        comment_id_col = self.data.columns.get_loc("comment_id")
        self.table_view.setColumnHidden(comment_id_col, True)

        self._tab_widget.setEnabled(True)
        self.context_dir_changed.emit(str(path))

    def column_renames(self):
        return {name: v.title for name, v in self._attributi.items() if v.title}

    def column_title(self, name):
        if name in self._attributi:
            return self._attributi[name].title or name
        return name

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

        action_edit_ctx = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("accessories-text-editor"), "Edit context file", self
        )
        action_edit_ctx.setStatusTip("Open the Python context file in a text editor")
        action_edit_ctx.triggered.connect(self._menu_bar_edit_context)
        action_edit_ctx.setEnabled(False)
        self.context_dir_changed.connect(lambda _: action_edit_ctx.setEnabled(True))

        action_help = QtWidgets.QAction(QtGui.QIcon("help.png"), "&Help", self)
        action_help.setShortcut("Shift+H")
        action_help.setStatusTip("Get help.")
        action_help.triggered.connect(self._menu_bar_help)

        action_exit = QtWidgets.QAction(QtGui.QIcon("exit.png"), "&Exit", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.setStatusTip("Exit AMORE GUI.")
        action_exit.triggered.connect(QtWidgets.QApplication.instance().quit)

        fileMenu = menu_bar.addMenu(
            QtGui.QIcon("amore_mid_prototype/gui/ico/AMORE.png"), "&AMORE"
        )
        fileMenu.addAction(action_autoconfigure)
        fileMenu.addAction(action_edit_ctx)
        fileMenu.addAction(action_help)
        fileMenu.addAction(action_exit)

    def handle_update(self, message):

        # is the message OK?
        if "Run" not in message.keys():
            raise ValueError("Malformed message.")

        # log.info("Updating for message: %s", message)

        # Rename:
        #  start_time -> Timestamp
        renames = {
            "start_time": "Timestamp",
            **self.column_renames(),
        }
        message = {
            "Status": True,
            **{renames.get(k, k): v for (k, v) in message.items()},
        }

        # initialize the view
        if not self._received_update:
            self._received_update = True
            self._status_bar_connection_status.setStyleSheet(
                "color:green;font-weight:bold;"
            )
            self._status_bar_connection_status.setText(
                f"Getting updates ({self.db_id})"
            )

        if self.data is None:
            # ingest data
            self.data = pd.DataFrame({**message, **{"Comment": ""}}, index=[0])

            # build the table
            self._create_view()

        else:
            # is the run already in?
            row = self.data.loc[self.data["Run"] == message["Run"]]

            # additional columns?
            new_cols = set(message) - set(self.data.columns)

            if new_cols:
                log.info("New columns for table: %s", new_cols)
                ncols_before = self.table.columnCount()
                self.table.beginInsertColumns(
                    QtCore.QModelIndex(), ncols_before, ncols_before + len(new_cols) - 1
                )
                for col_name in new_cols:
                    self.data.insert(len(self.data.columns), col_name, np.nan)
                self.table.endInsertColumns()

                self.table_view.add_new_columns(
                    list(new_cols), [True for _ in list(new_cols)]
                )

            if row.size:
                log.debug(
                    "Update existing row %s for run %s", row.index, message["Run"]
                )
                for ki, vi in message.items():
                    self.data.at[row.index[0], ki] = vi

                    index = self.table.index(
                        row.index[0], self.data.columns.get_loc(ki)
                    )
                    self.table.dataChanged.emit(index, index)

            else:
                sort_col = self.table.is_sorted_by
                if self.table.is_sorted_by and message.get(sort_col):
                    newval = message[sort_col]
                    if self.table.is_sorted_order == Qt.SortOrder.AscendingOrder:
                        ix = self.data[sort_col].searchsorted(newval)
                    else:
                        ix_back = self.data[sort_col][::-1].searchsorted(newval)
                        ix = len(self.data) - ix_back
                else:
                    ix = len(self.data)
                log.debug("New row in table at index %d", ix)

                # Extract the high-rank arrays from the messages, because
                # DataFrames can only handle 1D cell elements by default. The
                # way around this is to create a column manually with a dtype of
                # 'object'.
                ndarray_cols = { }
                for key, value in message.copy().items():
                    if isinstance(value, np.ndarray) and value.ndim > 1:
                        ndarray_cols[key] = value
                        del message[key]

                # Create a DataFrame with the new data to insert into the main table
                new_entries = pd.DataFrame({**message, **{"Comment": ""}},
                                           index=[self.table.rowCount()])

                # Insert columns with 'object' dtype for the special columns
                # with arrays that are >1D.
                for col_name, value in ndarray_cols.items():
                    col = pd.Series([value], index=[self.table.rowCount()], dtype="object")
                    new_entries.insert(len(new_entries.columns), col_name, col)

                new_df = pd.concat(
                    [
                        self.data.iloc[:ix],
                        new_entries,
                        self.data.iloc[ix:],
                    ],
                    ignore_index=True,
                )

                self.table.beginInsertRows(QtCore.QModelIndex(), ix, ix)
                self.data = new_df
                self.table.endInsertRows()

        # update plots and plotting controls
        self.plot.update_columns()
        self.plot.update()

        # (over)write down metadata
        self.data.to_json("AMORE.json")

    def _updates_thread_launcher(self) -> None:
        if not self._connect_to_kafka:
            return

        assert self.db_id is not None

        try:
            self.update_receiver = UpdateReceiver(self.db_id)
        except NoBrokersAvailable:
            QtWidgets.QMessageBox.warning(self, "Broker connection failed",
                                          f"Could not connect to any Kafka brokers at: {' '.join(UPDATE_BROKERS)}\n\n" +
                                          "DAMNIT can operate offline, but it will not receive any updates from new or reprocessed runs.")
            return

        self._updates_thread = QtCore.QThread()
        self.update_receiver.moveToThread(self._updates_thread)

        self._updates_thread.started.connect(self.update_receiver.loop)
        self.update_receiver.message.connect(self.handle_update)
        QtCore.QTimer.singleShot(0, self._updates_thread.start)

    def _set_comment_date(self):
        self.comment_time.setText(
            time.strftime("%H:%M %d/%m/%Y", time.localtime(time.time()))
        )

    def _comment_button_clicked(self):
        ts = datetime.strptime(self.comment_time.text(), "%H:%M %d/%m/%Y").timestamp()
        text = self.comment.text()
        with self.db:
            cur = self.db.execute("INSERT INTO time_comments VALUES (?, ?)", (ts, text))
        comment_id = cur.lastrowid
        self.data = pd.concat(
            [
                self.data,
                pd.DataFrame(
                    {
                        "Status": False,
                        "Timestamp": ts,
                        "Run": pd.NA,
                        "Proposal": pd.NA,
                        "Comment": text,
                        "comment_id": comment_id,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

        # this block is ugly
        if len(self.table.is_sorted_by):
            self.data.sort_values(
                self.table.is_sorted_by,
                ascending=self.table.is_sorted_order == QtCore.Qt.AscendingOrder,
                inplace=True,
            )
            self.data.reset_index(inplace=True, drop=True)
        self.table.insertRows(self.table.rowCount())

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

    def ds_name(self, quantity):
        # a LUT would be better
        for ki, vi in self._attributi.items():
            if vi.title == quantity:
                return ki

        return quantity

    def make_finite(self, data):
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        return data.fillna(np.nan)

    def inspect_data(self, index):
        proposal = self.data["Proposal"][index.row()]
        run = self.data["Run"][index.row()]

        quantity_title = self.data.columns[index.column()]
        quantity = self.ds_name(quantity_title)

        # Don't try to plot comments
        if quantity in ["Comment", "Status"]:
            return

        log.info(
            "Selected proposal {} run {}, property {}".format(
                proposal, run, quantity_title
            )
        )

        cell_data = self.data.iloc[index.row(), index.column()]
        if not pd.api.types.is_number(cell_data) and not isinstance(cell_data, np.ndarray):
            QMessageBox.warning(self, "Can't inspect variable",
                                f"'{quantity}' has type '{type(cell_data).__name__}', cannot inspect.")
            return

        is_image = (isinstance(cell_data, np.ndarray) and cell_data.ndim == 2)

        try:
            file_name, dataset = self.get_run_file(proposal, run)
        except:
            return

        try:
            if is_image:
                image = dataset[quantity]["data"][:]
            else:
                y_ds = dataset[quantity]["data"]
                if len(y_ds.shape) == 0:
                    # If this is a scalar value, then we can't plot it
                    QMessageBox.warning(self, "Can't inspect variable",
                                        f"'{quantity}' is a scalar, there's nothing more to plot.")
                    return
                else:
                    y = y_ds[:]

                # Use the train ID if it's been saved, otherwise generate an X axis
                if "trainId" in dataset[quantity]:
                    x = dataset[quantity]["trainId"][:]
                else:
                    x = np.arange(len(y))
        except KeyError:
            log.warning("'{}' not found in {}...".format(quantity, file_name))
            return
        finally:
            dataset.close()

        self._canvas_inspect.append(
            Canvas(
                self,
                x=[self.make_finite(x)] if not is_image else [],
                y=[self.make_finite(y)] if not is_image else [],
                image=image if is_image else None,
                xlabel="Event (run {})".format(run),
                ylabel=quantity_title,
                fmt="ro",
                autoscale=False,
                strongly_correlated=True
            )
        )
        self._canvas_inspect[-1].show()

    def _create_view(self) -> None:
        vertical_layout = QtWidgets.QVBoxLayout()
        table_horizontal_layout = QtWidgets.QHBoxLayout()
        comment_horizontal_layout = QtWidgets.QHBoxLayout()

        # the table
        self.table_view = TableView()
        self.table = Table(self)
        self.table.comment_changed.connect(self.save_comment)
        self.table.time_comment_changed.connect(self.save_time_comment)
        self.table.run_visibility_changed.connect(lambda row, state: self.plot.update())

        self.table_view.doubleClicked.connect(self.inspect_data)

        table_horizontal_layout.addWidget(self.table_view, stretch=6)
        table_horizontal_layout.addWidget(self.table_view.create_column_widget(),
                                          stretch=1)

        vertical_layout.addLayout(table_horizontal_layout)

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

        vertical_layout.addLayout(comment_horizontal_layout)

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

        plot_horizontal_layout.addWidget(self.plot._combo_box_x_axis)
        plot_horizontal_layout.addWidget(self.plot.vs_button)
        plot_horizontal_layout.addWidget(self.plot._combo_box_y_axis)

        plot_vertical_layout.addLayout(plot_horizontal_layout)

        plot_parameters_horizontal_layout.addWidget(self.plot._button_plot_runs)
        self.plot._button_plot.setMinimumWidth(200)
        plot_parameters_horizontal_layout.addStretch()

        plot_parameters_horizontal_layout.addWidget(
            self.plot._toggle_probability_density
        )

        plot_vertical_layout.addLayout(plot_parameters_horizontal_layout)

        plotting_group.setLayout(plot_vertical_layout)

        vertical_layout.addWidget(plotting_group)

        self._view_widget.setLayout(vertical_layout)

    def configure_editor(self):
        test_widget = QtWidgets.QWidget()

        font = QtGui.QFont("Monospace", pointSize=12)
        self._editor_lexer = QsciLexerPython()
        self._editor_lexer.setDefaultFont(font)

        self._editor.setLexer(self._editor_lexer)
        self._editor.setAutoCompletionSource(QsciScintilla.AutoCompletionSource.AcsAll)
        self._editor.setAutoCompletionThreshold(3)
        self._editor.setIndentationsUseTabs(False)
        self._editor.setTabWidth(4)
        self._editor.setAutoIndent(True)
        self._editor.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
        self._editor.setCaretLineVisible(True)
        self._editor.setCaretLineBackgroundColor(QtGui.QColor('lightgray'))
        self._editor.setMarginWidth(0, "0000")
        self._editor.setMarginLineNumbers(0, True)
        self._editor.textChanged.connect(self.on_context_changed)

        # Set Ctrl + D to delete a line
        commands = self._editor.standardCommands()
        ctrl_d = commands.boundTo(Qt.ControlModifier | Qt.Key_D)
        ctrl_d.setKey(0)

        line_del = commands.find(QsciCommand.LineDelete)
        line_del.setKey(Qt.ControlModifier | Qt.Key_D)

        # And Ctrl + S to save the context file
        self._save_shortcut = QShortcut(Qt.ControlModifier | Qt.Key_S, self._editor)
        self._save_shortcut.activated.connect(self.save_context)

        vbox = QtWidgets.QGridLayout()
        test_widget.setLayout(vbox)

        test_btn = QtWidgets.QPushButton("Test")
        test_btn.clicked.connect(self.test_context)

        self._context_status_icon = QtSvg.QSvgWidget()

        font = QtGui.QFont("Monospace", pointSize=9)
        self._error_widget_lexer = QsciLexerPython()
        self._error_widget_lexer.setDefaultFont(font)

        self._error_widget.setReadOnly(True)
        self._error_widget.setCaretWidth(0)
        self._error_widget.setLexer(self._error_widget_lexer)

        vbox.addWidget(test_btn, 0, 0)
        vbox.addWidget(self._context_status_icon, 1, 0)
        vbox.addWidget(self._error_widget, 0, 1, 2, 1)
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

    def test_context(self):
        try:
            ContextFile.from_str(self._editor.text())
        except:
            self.set_error_widget_text(traceback.format_exc())
            self.set_error_icon("red")

            # Resize the error window
            height = self._editor_parent_widget.height()
            self._editor_parent_widget.setSizes([2 * height // 3, height // 3])

            # Extract the line number of the error
            exc_type, e, tb = sys.exc_info()
            lineno = -1
            offset = 0
            if exc_type == SyntaxError:
                # SyntaxError's are special, their tracebacks don't include the
                # line number.
                lineno = e.lineno
                offset = e.offset - 1
            else:
                # Look for the frame with a filename matching the context
                for frame in traceback.extract_tb(tb):
                    if frame.filename == "<string>":
                        lineno = frame.lineno
                        break

            if lineno != -1:
                # The line numbers reported by Python are 1-indexed so we
                # decrement before passing them to scintilla.
                lineno -= 1

                self._editor.ensureLineVisible(lineno)

                # If we're already at the line, don't move the cursor unnecessarily
                if lineno != self._editor.getCursorPosition()[0]:
                    self._editor.setCursorPosition(lineno, offset)

            return False

        # If that worked, try pyflakes
        out_buffer = StringIO()
        reporter = Reporter(out_buffer, out_buffer)
        num_warnings = pyflakes_check(self._editor.text(), "<ctx>", reporter)

        # Move the error widget down
        height = self.height()
        self._editor_parent_widget.setSizes([height, 1])

        if num_warnings > 0:
            # We don't treat pyflakes warnings as fatal errors because sometimes
            # pyflakes reports valid but harmless problems, like unused
            # variables or unused imports.
            self.set_error_widget_text(out_buffer.getvalue())
            self.set_error_icon("yellow")
            return True
        else:
            self.set_error_widget_text("Valid context.")
            self.set_error_icon("green")

            return True

    def set_error_icon(self, icon):
        self._context_status_icon.load(f"amore_mid_prototype/gui/ico/{icon}_circle.svg")
        self._context_status_icon.renderer().setAspectRatioMode(Qt.KeepAspectRatio)

    def set_error_widget_text(self, text):
        # Clear the widget and wait for a bit to visually indicate to the
        # user that something happened.
        self._error_widget.setText("")
        QtWidgets.QApplication.instance().processEvents()
        time.sleep(0.1)

        self._error_widget.setText(text)

    def save_context(self):
        if not self.test_context():
            return

        self._context_path.write_text(self._editor.text())
        self.mark_context_saved()

    def mark_context_saved(self):
        self._context_is_saved = True
        self._tabbar_style.enable_bold = False
        self._tab_widget.setTabText(1, "Context file")
        self._tab_widget.tabBar().setTabTextColor(1, QtGui.QColor("black"))
        self._editor_status_message = str(self._context_path.resolve())
        self.on_tab_changed(self._tab_widget.currentIndex())

    def save_comment(self, prop, run, value):
        if self.db is None:
            log.warning("No SQLite database in use, comment not saved")
            return

        log.debug("Saving comment for prop %d run %d", prop, run)
        with self.db:
            self.db.execute(
                """
                UPDATE runs set comment=? WHERE proposal=? AND runnr=?
                """,
                (value, int(prop), int(run)),
            )

    def save_time_comment(self, comment_id, value):
        if self.db is None:
            log.warning("No SQLite database in use, comment not saved")
            return

        log.debug("Saving time-based comment ID %d", comment_id)
        with self.db:
            self.db.execute(
                """UPDATE time_comments set comment=? WHERE rowid=?""",
                (value, comment_id),
            )


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


def run_app(context_dir, connect_to_kafka=True):
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar
    )
    application = QtWidgets.QApplication(sys.argv)
    application.setStyle(TableViewStyle())

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
