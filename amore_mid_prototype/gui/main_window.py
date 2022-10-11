import typing
import pickle
import os
import logging
import shutil
import sys
import time
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from subprocess import Popen

import libtmux
import pandas as pd
import numpy as np
import h5py
import zulip
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from kafka.errors import NoBrokersAvailable

from extra_data.read_machinery import find_proposal

from ..backend.db import open_db, get_meta, set_meta
from ..context import ContextFile
from ..definitions import UPDATE_BROKERS
from .kafka import UpdateReceiver
from .table import TableView, Table
from .plot import Canvas, Plot


log = logging.getLogger(__name__)
pd.options.mode.use_inf_as_na = True


class QLogger(logging.Handler):
    # https://stackoverflow.com/questions/28655198/best-way-to-display-logs-in-pyqt
    def __init__(self, parent):
        super().__init__()
        self.widget = QtWidgets.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)
        self.widget.setFixedHeight(75)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)

class MainWindow(QtWidgets.QMainWindow):
    context_path = None
    db = None
    db_id = None

    context_dir_changed = QtCore.pyqtSignal(str)

    def __init__(self, context_dir: Path = None, zulip_config: typing.Optional[str] = None):
        super().__init__()

        self.data = None
        self._updates_thread = None
        self._received_update = False
        self._attributi = {}

        self.zulip_config = zulip_config
        self.zulip_client = None
        self.zulip_streams = None
        if self.zulip_config is not None:
            self.zulip_client = zulip.Client(config_file=zulip_config)

        self.setWindowTitle("Data And Metadata iNspection Interactive Thing")
        self.setWindowIcon(QtGui.QIcon("amore_mid_prototype/gui/ico/AMORE.png"))

        self._create_status_bar()
        self._create_menu_bar()

        # self.tabs = QtWidgets.QTabWidget()
        # self.tab1 = QtWidgets.QWidget()
        # self.tab2 = QtWidgets.QWidget()

        # self.tabs.addTab(self.tab1, "Inspector")
        # self.tabs.addTab(self.tab2, "Editor")

        self._view_widget = QtWidgets.QWidget(self)
        # Disable the main window at first since we haven't loaded any database yet
        self._view_widget.setEnabled(False)
        self.setCentralWidget(self._view_widget)

        # logging
        self.logger = QLogger(self)
        self.logger.setFormatter(
            logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")
        )
        logging.getLogger().addHandler(self.logger)

        self._create_view()
        self.center_window()

        if context_dir is not None:
            self.autoconfigure(context_dir)

        self._canvas_inspect = []

    def closeEvent(self, event):

        for ci in range(len(self.plot._canvas["canvas"])):
            self.plot._canvas["canvas"][ci].close()

        self.stop_update_listener_thread()

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
        Popen(['xdg-open', self.context_path])

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
                prop_no, prompt = QtWidgets.QInputDialog.getInt(
                    self, "Select proposal", "Which proposal is this for?"
                )
                if not prompt:
                    break

                proposal = f"p{prop_no:06}"
                try:
                    proposal_dir = find_proposal(proposal)
                    prompt = False
                except Exception:
                    button = QtWidgets.QMessageBox.warning(
                        self,
                        "Bad proposal number",
                        "Could not find a proposal with this number, try again?",
                        buttons=QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    )
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
        self.context_path = path / "context.py"
        if self.context_path.is_file():
            log.info("Reading context file %s", self.context_path)
            ctx_file = ContextFile.from_py_file(self.context_path)
            self._attributi = ctx_file.vars

        self.extracted_data_template = str(path / "extracted_data/p{}_r{}.h5")

        sqlite_path = self.db_path(path)
        log.info("Reading data from database")
        self.db = open_db(sqlite_path)

        self.db_id = get_meta(self.db, "db_id")
        self.stop_update_listener_thread()
        self._updates_thread_launcher()

        df = pd.read_sql_query("SELECT * FROM runs", self.db)
        df.insert(0, "Use", True)
        df.insert(len(df.columns), "_comment_id", pd.NA)
        df.pop("added_at")

        # Unpickle serialized objects. First we select all columns that
        # might need deserializing.
        object_cols = df.select_dtypes(include=["object"]).drop(["comment", "_comment_id"], axis=1)

        # Then we check each element and unpickle it if necessary, and
        # finally update the main DataFrame.
        unpickled_cols = object_cols.applymap(lambda x: pickle.loads(x) if isinstance(x, bytes) else x)
        df.update(unpickled_cols)

        # Read the comments and prepare them for merging with the main data
        comments_df = pd.read_sql_query(
            "SELECT rowid as _comment_id, * FROM time_comments", self.db
        )
        comments_df.insert(0, "Run", pd.NA)
        comments_df.insert(1, "Proposal", pd.NA)
        # Don't try to plot comments
        comments_df.insert(2, "Use", False)

        self.data = pd.concat(
            [
                df.rename(
                    columns={
                        "runnr": "Run",
                        "proposal": "Proposal",
                        "start_time": "Timestamp",
                        "comment": "Comment",
                        "added_at": "_added_at",
                        **self.column_renames(),
                    }
                ),
                comments_df.rename(
                    columns={
                        "timestamp": "Timestamp",
                        "comment": "Comment",
                    }
                ),
            ]
        )

        # move some columns
        if "Comment" in self.data.columns:
            self.data.insert(1, "Comment", self.data.pop("Comment"))

        if "Timestamp" in self.data.columns:
            self.data.insert(0, "Timestamp", self.data.pop("Timestamp"))

        if "Use" in self.data.columns:
            self.data.insert(2, "Use", self.data.pop("Use"))

        if "Run" in self.data.columns:
            self.data.insert(3, "Run", self.data.pop("Run"))

        self.table_view.setModel(self.table)
        self.table_view.sortByColumn(
            self.data.columns.get_loc("Timestamp"), Qt.SortOrder.AscendingOrder
        )

        # Always keep these columns as small as possible to save space
        header = self.table_view.horizontalHeader()
        for column in ["Use", "Run", "Timestamp"]:
            column_index = self.data.columns.get_loc(column)
            header.setSectionResizeMode(
                column_index, QtWidgets.QHeaderView.ResizeToContents
            )

        self.plot.update_columns()

        # hide some column
        for column in self.data.columns:
            if column.startswith("_"):
                self.table_view.setColumnHidden(
                    self.data.columns.get_loc(column), True
                )

                # to avoid tweaking the sorting, hidden columns should be the last ones
                # self.data.insert(
                #    len(self.data.columns) - 1, column, self.data.pop(column)
                # )

        self._status_bar.showMessage(
            "Select some entries in the table, all of them using Ctrl+A. Double-click on a cell to inspect data."
        )
        self._view_widget.setEnabled(True)
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
            "Use": True,
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
                ndarray_cols = {}
                for key, value in message.copy().items():
                    if isinstance(value, np.ndarray) and value.ndim > 1:
                        ndarray_cols[key] = value
                        del message[key]

                # Create a DataFrame with the new data to insert into the main table
                new_entries = pd.DataFrame(
                    {**message, **{"Comment": ""}}, index=[self.table.rowCount()]
                )

                # Insert columns with 'object' dtype for the special columns
                # with arrays that are >1D.
                for col_name, value in ndarray_cols.items():
                    col = pd.Series(
                        [value], index=[self.table.rowCount()], dtype="object"
                    )
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
        assert self.db_id is not None

        try:
            self.update_receiver = UpdateReceiver(self.db_id)
        except NoBrokersAvailable:
            QtWidgets.QMessageBox.warning(
                self,
                "Broker connection failed",
                f"Could not connect to any Kafka brokers at: {' '.join(UPDATE_BROKERS)}\n\n"
                + "DAMNIT can operate offline, but it will not receive any updates from new or reprocessed runs.",
            )
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

    def _table_settings_button_clicked(self):
        dialog = QtWidgets.QDialog(self)

        layout = self.table_view.set_columns_visibility(
            [self.column_title(c) for c in self.data.columns],
            [True for _ in self.data.columns],
        )

        dialog.setLayout(layout)
        dialog.show()

    def _log_show_button_clicked(self):
        if self.log_show_button.isChecked():
            self.logger.widget.show()
            self.log_show_button.setText("Hide log")
        else:
            self.logger.widget.hide()

    def _comment_button_clicked(self):
        ts = (
            datetime.strptime(self.comment_time.text(), "%H:%M %d/%m/%Y")
            .astimezone(timezone.utc)
            .timestamp()
        )
        text = self.comment.text()
        with self.db:
            cur = self.db.execute("INSERT INTO time_comments VALUES (?, ?)", (ts, text))
        comment_id = cur.lastrowid
        self.data = pd.concat(
            [
                self.data,
                pd.DataFrame(
                    {
                        "Use": False,
                        "Timestamp": ts,
                        "Run": pd.NA,
                        "Proposal": pd.NA,
                        "Comment": text,
                        "_comment_id": comment_id,
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

    def get_run_file(self, proposal, run, write_to_log=True):
        file_name = self.extracted_data_template.format(proposal, run)

        try:
            run_file = h5py.File(file_name)
            return file_name, run_file
        except FileNotFoundError:  # as e:
            if write_to_log:
                log.warning("{} not found...".format(file_name))
            return None, None

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
        if quantity in ["Comment", "Use"]:
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
                fmt="o-",
                color="red",
            )
        )
        self._canvas_inspect[-1].show()

    def _create_view(self) -> None:
        vertical_layout = QtWidgets.QVBoxLayout()
        table_horizontal_layout = QtWidgets.QHBoxLayout()
        # filter_horizontal_layout = QtWidgets.QHBoxLayout()
        settings_horizontal_layout = QtWidgets.QHBoxLayout()
        comment_horizontal_layout = QtWidgets.QHBoxLayout()

        # filter
        # filter_button = QtWidgets.QPushButton("Filter data")
        # filter_button.setEnabled(True)
        # filter_button.setMinimumWidth(175)
        # filter_button.clicked.connect(self._comment_button_clicked)

        # self.filter = QtWidgets.QLineEdit(self)
        # self.filter.setToolTip("Filter results.")

        # filter_horizontal_layout.addWidget(filter_button)
        # filter_horizontal_layout.addWidget(self.filter)

        # vertical_layout.addLayout(filter_horizontal_layout)

        # the table
        self.table_view = TableView()
        self.table = Table(self)
        self.table.comment_changed.connect(self.save_comment)
        self.table.time_comment_changed.connect(self.save_time_comment)
        self.table.run_visibility_changed.connect(lambda row, state: self.plot.update())

        self.table_view.doubleClicked.connect(self.inspect_data)

        table_horizontal_layout.addWidget(self.table_view)

        vertical_layout.addLayout(table_horizontal_layout)

        # groups and table settings
        # table_settings_button = QtWidgets.QPushButton("Groups settings")
        # table_settings_button.setEnabled(True)
        # table_settings_button.setMinimumWidth(175)
        # table_settings_button.clicked.connect(self._table_settings_button_clicked)

        table_settings_button = QtWidgets.QPushButton("Table settings")
        table_settings_button.setEnabled(True)
        table_settings_button.setMinimumWidth(175)
        table_settings_button.clicked.connect(self._table_settings_button_clicked)

        self.log_show_button = QtWidgets.QPushButton("Show log")
        self.log_show_button.setEnabled(True)
        self.log_show_button.setCheckable(True)
        self.log_show_button.setMinimumWidth(175)
        self.log_show_button.clicked.connect(self._log_show_button_clicked)

        settings_horizontal_layout.addWidget(table_settings_button)
        settings_horizontal_layout.addStretch()
        settings_horizontal_layout.addWidget(self.log_show_button)

        # self.filter = QtWidgets.QLineEdit(self)
        # self.filter.setToolTip("Filter results.")

        # filter_horizontal_layout.addWidget(filter_button)
        # filter_horizontal_layout.addWidget(self.filter)

        vertical_layout.addLayout(settings_horizontal_layout)

        vseparator = QtWidgets.QFrame()
        vseparator.setFrameShape(QtWidgets.QFrame.HLine)
        vseparator.setFrameShadow(QtWidgets.QFrame.Sunken)

        vertical_layout.addWidget(vseparator)

        # plotting control
        self.plot = Plot(self)
        # plotting_group = QtWidgets.QGroupBox(
        #    "Plot (double-click on a cell to inspect data)"
        # )

        plot_grid_layout = QtWidgets.QGridLayout()
        # plotting_group.setLayout(plot_grid_layout)

        plot_grid_layout.addWidget(self.plot._button_plot_runs, *(0, 0))
        plot_grid_layout.addWidget(self.plot._toggle_plot_summary_table, *(0, 1))
        plot_grid_layout.setColumnStretch(2, 1)

        plot_grid_layout.addWidget(self.plot._toggle_probability_density, *(0, 3))

        self.plot._combo_box_x_axis.setFixedWidth(300)
        self.plot._combo_box_y_axis.setFixedWidth(300)
        plot_grid_layout.addWidget(self.plot._combo_box_y_axis, *(0, 4))
        plot_grid_layout.addWidget(self.plot.vs_label, *(0, 5))
        plot_grid_layout.addWidget(self.plot._combo_box_x_axis, *(0, 6))

        vertical_layout.addLayout(plot_grid_layout)

        vseparator = QtWidgets.QFrame()
        vseparator.setFrameShape(QtWidgets.QFrame.HLine)
        vseparator.setFrameShadow(QtWidgets.QFrame.Sunken)

        vertical_layout.addWidget(vseparator)

        # comments
        self.comment = QtWidgets.QLineEdit(self)
        self.comment.setToolTip("Time can be edited in the field on the left.")

        self.comment_time = QtWidgets.QLineEdit(self)
        self.comment_time.setStyleSheet("width: 25px;")
        self.comment_time.setAlignment(QtCore.Qt.AlignRight)
        self.comment_time.setFixedWidth(175)

        comment_button = QtWidgets.QPushButton("Additional comment")
        comment_button.setEnabled(True)
        comment_button.setMinimumWidth(175)
        comment_button.clicked.connect(self._comment_button_clicked)

        comment_horizontal_layout.addWidget(comment_button)
        comment_horizontal_layout.addWidget(self.comment_time)
        comment_horizontal_layout.addWidget(self.comment)

        vertical_layout.addLayout(comment_horizontal_layout)

        comment_timer = QtCore.QTimer()
        self._set_comment_date()
        comment_timer.setInterval(30000)
        comment_timer.timeout.connect(self._set_comment_date)
        comment_timer.start()

        self.logger.widget.hide()
        vertical_layout.addWidget(self.logger.widget)

        self._view_widget.setLayout(vertical_layout)

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
        if hint == QtWidgets.QStyle.SH_ToolTip_WakeUpDelay and isinstance(
            widget.parent(), TableView
        ):
            return 0
        else:
            return super().styleHint(hint, option, widget, returnData)


def run_app(context_dir, zulip_config=None):
    # to avoid Mac messing everything
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar
    )
    
    application = QtWidgets.QApplication(sys.argv)
    application.setStyle(TableViewStyle())

    window = MainWindow(context_dir=context_dir, zulip_config=zulip_config)
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
