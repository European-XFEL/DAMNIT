import pickle
import os
import logging
import shelve
import sys
import time

from operator import itemgetter
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np
import h5py

from kafka.errors import NoBrokersAvailable
from extra_data.read_machinery import find_proposal

from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg
from PyQt5.QtCore import Qt, QObject, QFileSystemWatcher
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTabWidget
from PyQt5.Qsci import QsciScintilla, QsciLexerPython

from ..backend.db import db_path, open_db, get_meta
from ..backend import initialize_and_start_backend, backend_is_running
from ..context import ContextFile
from ..definitions import UPDATE_BROKERS, LOG_FORMAT
from .kafka import UpdateReceiver
from .table import TableView, Table
from .plot import Canvas, Plot
from .editor import Editor, ContextTestResult
from ..log import DictHandler


log = logging.getLogger(__name__)
pd.options.mode.use_inf_as_na = True

class Settings(Enum):
    COLUMNS = "columns"


class MainWindow(QtWidgets.QMainWindow):
    db = None
    db_id = None
    log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    log_html_format = """<span style=" font-size:10pt;
font-weight:400; color:{}; display : inline;" >
<pre style = "white-space: pre-wrap;">{}<br></pre></span>"""

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

        self._settings_db_path = Path.home() / ".local" / "state" / "damnit" / "settings.db"
        self.be_log_path = 'amore.log'
        self.fe_log_path = str(Path.home() / '.amore.log')

        self.setWindowTitle("Data And Metadata iNspection Interactive Thing")
        self.setWindowIcon(QtGui.QIcon(self.icon_path("AMORE.png")))
        self._create_status_bar()
        self._create_menu_bar()

        self._view_widget = QtWidgets.QWidget(self)
        self._editor = Editor()
        self._error_widget = QsciScintilla()
        self._editor_parent_widget = QtWidgets.QSplitter(Qt.Vertical)
        self._log_view_widget = QtWidgets.QWidget(self)

        self._tab_widget = QTabWidget()
        self._tabbar_style = TabBarStyle()
        self._tab_widget.tabBar().setStyle(self._tabbar_style)
        self._tab_widget.addTab(self._view_widget, "Run table")
        self._tab_widget.addTab(self._editor_parent_widget, "Context file")
        self._tab_widget.addTab(self._log_view_widget, "Logs")
        self._tab_widget.currentChanged.connect(self.on_tab_changed)

        # Disable the main window at first since we haven't loaded any database yet
        self._tab_widget.setEnabled(False)
        self.setCentralWidget(self._tab_widget)

        # Local handler streams directly into the general log file
        self.log_pos = 0
        self.log_list = []
        self.d_handler = DictHandler(self.log_list)
        self.f_handler = logging.FileHandler(self.fe_log_path)
        formatter = logging.Formatter(LOG_FORMAT)
        self.f_handler.setFormatter(formatter)
        root_logger = logging.root 
        root_logger.addHandler(self.d_handler)
        root_logger.addHandler(self.f_handler)

        self.log_widget = QtWidgets.QTextEdit(self)
        self.log_widget.setReadOnly(True)
        self._create_log_view()

        self._create_view()
        self.configure_editor()
        self.center_window()

        if context_dir is not None:
            self.autoconfigure(context_dir)

        self._canvas_inspect = []

    def icon_path(self, name):
        """
        Helper function to get the path to an icon file stored under ico/.
        """
        return str(Path(__file__).parent / "ico" / name)

    def on_tab_changed(self, index):
        if index == 0:
            self._status_bar.showMessage("Double-click on a cell to inspect results.")
        elif index == 1:
            self._status_bar.showMessage(self._editor_status_message)
        elif index == 2:
            self._tab_widget.tabBar().setTabTextColor(2, QtGui.QColor("black"))

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
        if standard_path.is_dir() and db_path(standard_path).is_file():
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
                    initialize_and_start_backend(standard_path, prop_no)
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
        if not backend_is_running(path):
            button = QMessageBox.question(self, "Backend not running",
                                          "The AMORE backend is not running, would you like to start it? " \
                                          "This is only necessary if new runs are expected.")
            if button == QMessageBox.Yes:
                initialize_and_start_backend(path)

        self.autoconfigure(Path(path), proposal=prop_no)

    def gpfs_accessible(self):
        return os.path.isdir("/gpfs/exfel/exp")

    def save_settings(self):
        self._settings_db_path.parent.mkdir(parents=True, exist_ok=True)

        with shelve.open(str(self._settings_db_path)) as db:
            settings = { Settings.COLUMNS.value: self.table_view.get_column_states() }
            db[str(self._context_path)] = settings

    def autoconfigure(self, path: Path, proposal=None):
        context_path = path / "context.py"
        if not context_path.is_file():
            QMessageBox.critical(self, "No context file",
                                 "This database is missing a context file, it cannot be opened.")
            return
        else:
            self._context_path = context_path

        log.info("Reading context file %s", self._context_path)
        ctx_file = ContextFile.from_py_file(self._context_path)
        self._attributi = ctx_file.vars

        self._editor.setText(ctx_file.code)
        self.test_context()
        self.mark_context_saved()

        self.extracted_data_template = str(path / "extracted_data/p{}_r{}.h5")

        sqlite_path = db_path(path)
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
        object_cols = df.select_dtypes(include=["object"]).drop(columns=["comment", "comment_id"],
                                                                errors="ignore")
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

        # Load the users settings
        if self._settings_db_path.parent.is_dir():
            with shelve.open(str(self._settings_db_path)) as db:
                key = str(self._context_path)
                col_settings = db[key][Settings.COLUMNS.value]
        else:
            col_settings = { }

        saved_cols = list(col_settings.keys())
        df_cols = self.data.columns.tolist()

        # Strip missing columns
        saved_cols = [col for col in saved_cols if col in df_cols]

        # Sort columns such that every column not saved is pushed to the
        # beginning, and all saved columns are inserted afterwards.
        sorted_cols = [col for col in df_cols if col not in saved_cols]
        sorted_cols.extend(saved_cols)
        self.data = self.data[sorted_cols]

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

        # Hide the comment_id column and all columns hidden by the user
        hidden_columns = ["comment_id"] + [col for col in saved_cols if not col_settings[col]]
        for col in hidden_columns:
            self.table_view.set_column_visibility(col, False, for_restore=True)

        self._tab_widget.setEnabled(True)

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

        action_help = QtWidgets.QAction(QtGui.QIcon("help.png"), "&Help", self)
        action_help.setShortcut("Shift+H")
        action_help.setStatusTip("Get help.")
        action_help.triggered.connect(self._menu_bar_help)

        action_exit = QtWidgets.QAction(QtGui.QIcon("exit.png"), "&Exit", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.setStatusTip("Exit AMORE GUI.")
        action_exit.triggered.connect(QtWidgets.QApplication.instance().quit)

        fileMenu = menu_bar.addMenu(
            QtGui.QIcon(self.icon_path("AMORE.png")), "&AMORE"
        )
        fileMenu.addAction(action_autoconfigure)
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

    def _create_log_view(self):
        # get current last line of BackEnd(be) log file and watches it.
        # The same is done FortEnd(fe)
        if not(os.path.exists(self.be_log_path)):
            Path(self.be_log_path).touch()
        with open(self.be_log_path, 'rb') as log_file:
            self.line_positions_be = self.lines_pos(log_file)
            log_file.seek(0,2)
            self.be_log_pos = log_file.tell()
        self.watcher_backend = QFileSystemWatcher([self.be_log_path])
        self.watcher_backend.fileChanged.connect(self.add_log_entry)

        with open(self.fe_log_path, 'rb') as log_file:
            log_file.seek(0,2)
            self.fe_log_pos = log_file.tell()
        self.watcher_fe = QFileSystemWatcher([self.fe_log_path])
        self.watcher_fe.fileChanged.connect(self.display_log_lines)
    
        self.create_log_cboxes()
 
        horizontal_layout = QtWidgets.QHBoxLayout()
        horizontal_layout.setAlignment(QtCore.Qt.AlignRight)
        self.old_logs_button = QtWidgets.QPushButton('Load past logs')
        horizontal_layout.addWidget(self.old_logs_button)

        self.old_logs_button.adjustSize()

        self.be_pos_relative = len(self.line_positions_be)-1
        self.old_logs_button.clicked.connect(self.fetch_logs)
        vertical_layout = QtWidgets.QVBoxLayout()
        vertical_layout.addLayout(horizontal_layout) 
        vertical_layout.addWidget(self.log_widget) 
        vertical_layout.addLayout(self.log_cboxes) 
        self._log_view_widget.setLayout(vertical_layout)

    def lines_pos(self, file):
        line = file.readline()
        list_pos = [0, file.tell()]
        while line:
            line = file.readline()
            list_pos.append(file.tell())

        return list_pos

    def fetch_logs(self):
        # N is the number of lines to be fetched, while i is
        # the number of exception lines found.
        N = 30
        i = 0
        if self.be_pos_relative == 0:
            return

        with open(self.be_log_path, 'r') as log_file:
            if self.be_pos_relative <= N:
                pos = 0

            else:
                i=1
                pos = self.line_positions_be[self.be_pos_relative - N]
                log_file.seek(pos)
                new_line = log_file.readline()

                while new_line[0] != '2':
                    pos = self.line_positions_be[self.be_pos_relative - N - i]
                    log_file.seek(pos)
                    new_line = log_file.readline()
                    i+=1

                pos = self.line_positions_be[self.be_pos_relative - N - i - 1]

            log_file.seek(pos)
            new_line = log_file.readline()
            log_list_tba = []

            while new_line and \
                    (log_file.tell() != self.line_positions_be[self.be_pos_relative]):
                self.plain_to_dict(new_line, log_list_tba)
                new_line = log_file.readline()

        self.log_list.extend(log_list_tba)
        self.log_list = sorted(self.log_list, key=itemgetter('time')) 
        self.d_handler.update_list(self.log_list)
        self.refresh_log()
        self.log_widget.moveCursor(QtGui.QTextCursor.Start)

        self.be_pos_relative = (self.be_pos_relative -N -i)*int(pos != 0)
        i = 0

    def display_log_lines(self):
        while self.log_pos < len(self.log_list):
            self.display_log_line(self.log_pos)
            self.log_pos += 1

    def display_log_line(self, k = -1):
        self.log_widget.moveCursor(QtGui.QTextCursor.End)
        new_line = self.log_list[k]
        _time, _level, _name, _message = self.log_list[k].values()
        if not(_level.strip() in self.log_levels):
            return

        # miliseconds should not be displayed, but must be in 
        # the list for further sortting 
        if len(_time) >= 19:
            _time = _time[0:19].ljust(20,' ')

        _message = _message.replace('\n', '<br>')
        _level = _level.ljust(8,' ')
        _name = _name.ljust(40,' ')
        log_line = ''.join([_time, _level, _name, _message])
        html_new_line = self.plain_to_html(log_line)

        # only indicate an error occurence if the current tab
        # is not the log tab
        if new_line['level'].strip() == 'ERROR' and self._tab_widget.currentIndex() != 2:
            self._tab_widget.tabBar().setTabTextColor(2, QtGui.QColor("red"))
            self._status_bar.showMessage("An error occurred, check the log tab.")

        # add separation if the date has changed
        if new_line['time'][0:11] != self.log_list[k-1]['time'][0:11]\
                and k>0:
            html_separation =  '''<div style="text-align: right; display : inline;">
<hr />{} &#8595; | &#8593; {} <hr /></div>'''
            html_separation =  html_separation.format(
                    new_line['time'][0:11], self.log_list[k-1]['time'][0:11])
 
            html_separation = self.log_html_format.format('gray', html_separation)
            self.log_widget.insertHtml(html_separation)

        self.log_widget.insertHtml(html_new_line)

    def add_log_entry(self):
        # appends log entry from file to log_list 
        # and displays it
        log_list_tba = []
        with open(self.be_log_path, 'r') as f:
            f.seek(self.be_log_pos)
            newline = f.readline()
            while newline:
                self.plain_to_dict(newline, log_list_tba)
                newline = f.readline()
            self.be_log_pos = f.tell()

        if len(log_list_tba) > 0 and \
                all(['time' in l.keys() for l in log_list_tba]):
            self.log_list.extend(log_list_tba)
            self.display_log_lines()

    def plain_to_dict(self, newline, log_list_tba):
        # transforms newline (plain text) into a dic
        # please take note that record itens must be separated with |
        keys = ['time', 'level', 'name', 'message']
        if '|' in newline:
            values = [i.strip() for i in newline.split('|')]
            log_list_tba.append(dict(zip(keys, values)))

        else:
            # ERROR's execptions message lines should be added to
            # the upper ERROR  message. 
            try:
                log_list_tba[-1]['message'] = \
                        ''.join([log_list_tba[-1]['message'], newline])
            except:
                return

    def plain_to_html(self, logline):
        log_level = ''
        html_log = None

        # any log line without a level keyword is treated as an error
        for level in self.log_levels:
            log_level_found = level in logline
            if log_level_found:
                log_level = level
                break

        match log_level:
            case 'DEBUG':
                html_log = self.log_html_format.format('black', logline)
            case 'INFO':
                html_log = self.log_html_format.format('black', logline)
            case 'WARNING':
                html_log = self.log_html_format.format('#f27b1f', logline)
            case 'ERROR':
                html_log = self.log_html_format.format('red', logline)
            case 'CRITICAL':
                html_log = self.log_html_format.format('red', logline)

        html_log = html_log.replace('<br><br>', '<br>')
        return  html_log
 
    def create_log_cboxes(self):
        horizontal_layout = QtWidgets.QHBoxLayout()
        horizontal_layout.setAlignment(QtCore.Qt.AlignLeft)
        horizontal_layout.setSpacing(10)
        log_bar_label = QtWidgets.QLabel('Showing logs for:')
        horizontal_layout.addWidget(log_bar_label)

        self.infoC = QtWidgets.QCheckBox('Infos')
        self.infoC.setChecked(True)
        self.infoC.stateChanged.connect(lambda: \
                self.add_log_level('INFO') if self.infoC.isChecked()\
                else self.remove_log_level('INFO'))

        self.warningC = QtWidgets.QCheckBox('Warnings')
        self.warningC.setChecked(True)
        self.warningC.stateChanged.connect(lambda: \
                self.add_log_level('WARNING') if self.warningC.isChecked()\
                else self.remove_log_level('WARNING'))

        self.errorC = QtWidgets.QCheckBox('Errors')
        self.errorC.setChecked(True)
        self.errorC.stateChanged.connect(lambda: \
                self.add_log_level('ERROR') if self.errorC.isChecked()\
                else self.remove_log_level('ERROR'))

        horizontal_layout.addWidget(self.infoC)
        horizontal_layout.addWidget(self.warningC)
        horizontal_layout.addWidget(self.errorC)

        self.log_cboxes = horizontal_layout

    def add_log_level(self, level = None):
        self.log_levels.append(level)
        self.refresh_log()
 
    def remove_log_level(self, level = None):
        self.log_levels.remove(level)
        self.refresh_log()

    def refresh_log(self):
        self.log_widget.clear()
        self.log_pos = 0
        self.display_log_lines()

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
        self.table_view.settings_changed.connect(self.save_settings)

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

        self._editor.textChanged.connect(self.on_context_changed)
        self._editor.save_requested.connect(self.save_context)

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
        test_result, output = self._editor.test_context()

        if test_result == ContextTestResult.ERROR:
            self.set_error_widget_text(output)
            self.set_error_icon("red")

            # Resize the error window
            height = self._editor_parent_widget.height()
            self._editor_parent_widget.setSizes([2 * height // 3, height // 3])
        else:
            # Move the error widget down
            height = self.height()
            self._editor_parent_widget.setSizes([height, 1])

            if test_result == ContextTestResult.WARNING:
                self.set_error_widget_text(output)

                # We don't treat pyflakes warnings as fatal errors because sometimes
                # pyflakes reports valid but harmless problems, like unused
                # variables or unused imports.
                self.set_error_icon("yellow")
            elif test_result == ContextTestResult.OK:
                self.set_error_widget_text("Valid context.")
                self.set_error_icon("green")

        return test_result

    def set_error_icon(self, icon):
        self._context_status_icon.load(self.icon_path(f"{icon}_circle.svg"))
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

    ging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    sys.exit(run_app(args.context_dir))


if __name__ == "__main__":
    main()
