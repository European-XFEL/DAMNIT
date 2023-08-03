import pickle
import os
import logging
import shelve
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np
import h5py
from pandas.api.types import infer_dtype

from kafka.errors import NoBrokersAvailable

from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QTabWidget
from PyQt5.Qsci import QsciScintilla, QsciLexerPython

from ..backend.db import db_path, DamnitDB
from ..backend import initialize_and_start_backend, backend_is_running
from ..context import ContextFile
from ..ctxsupport.damnit_ctx import UserEditableVariable
from ..ctxsupport.ctxrunner import get_user_variables
from ..definitions import UPDATE_BROKERS
from .kafka import UpdateReceiver
from .table import TableView, Table
from .plot import Canvas, Plot
from .user_variables import AddUserVariableDialog
from .editor import Editor, ContextTestResult
from .open_dialog import OpenDBDialog
from .zulip_messenger import ZulipMessenger


log = logging.getLogger(__name__)
pd.options.mode.use_inf_as_na = True

class Settings(Enum):
    COLUMNS = "columns"

class MainWindow(QtWidgets.QMainWindow):

    context_dir_changed = QtCore.pyqtSignal(str)

    db = None
    db_id = None
    _columns_dialog = None

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
        self._title_to_name = {}
        self._name_to_title = {}

        self._settings_db_path = Path.home() / ".local" / "state" / "damnit" / "settings.db"

        self.setWindowTitle("Data And Metadata iNspection Interactive Thing")
        self.setWindowIcon(QtGui.QIcon(self.icon_path("AMORE.png")))
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
        
        self.zulip_messenger = None

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

        self._status_bar.messageChanged.connect(lambda m: self.show_default_status_message() if m == "" else m)

        self._status_bar.setStyleSheet("QStatusBar::item {border: None;}")
        self._status_bar.showMessage("Autoconfigure AMORE.")
        self.setStatusBar(self._status_bar)

        self._status_bar_connection_status = QtWidgets.QLabel()
        self._status_bar.addPermanentWidget(self._status_bar_connection_status)

    def show_status_message(self, message, timeout = 0, stylesheet = ''):
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

        self.autoconfigure(context_dir, proposal=prop_no)

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

        sqlite_path = db_path(path)
        log.info("Reading data from database")
        self.db = DamnitDB(sqlite_path)

        user_variables = get_user_variables(self.db.conn)

        log.info("Reading context file %s", self._context_path)
        ctx_file = ContextFile.from_py_file(self._context_path, external_vars = user_variables)

        for kk, vv in user_variables.items():
            self.table.add_editable_column(vv.title or vv.name)

        self._attributi = ctx_file.vars

        self._title_to_name = { "Comment" : "comment"} | {
            (aa.title or kk) : kk for kk, aa in self._attributi.items()
        }
        self._name_to_title = {vv : kk for kk, vv in self._title_to_name.items()}

        self._editor.setText(ctx_file.code)
        self.test_context()
        self.mark_context_saved()

        self.extracted_data_template = str(path / "extracted_data/p{}_r{}.h5")

        self.db_id = self.db.metameta['db_id']
        self.stop_update_listener_thread()
        self._updates_thread_launcher()

        df = pd.read_sql_query("SELECT * FROM runs", self.db.conn)
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
            "SELECT rowid as comment_id, * FROM time_comments", self.db.conn
        )
        comments_df.insert(0, "Run", pd.NA)
        comments_df.insert(1, "Proposal", pd.NA)
        # Don't try to plot comments
        comments_df.insert(2, "Status", False)

        # Unset the table_view model before we start changing it
        self.table_view.setModel(None)

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
        col_settings = { }
        if self._settings_db_path.parent.is_dir():
            with shelve.open(str(self._settings_db_path)) as db:
                key = str(self._context_path)
                if key in db:
                    col_settings = db[key][Settings.COLUMNS.value]

        saved_cols = list(col_settings.keys())
        df_cols = self.data.columns.tolist()

        # Strip missing columns
        saved_cols = [col for col in saved_cols if col in df_cols]

        # Sort columns such that all static columns (proposal, run, etc) are at
        # the beginning, followed by all the columns that have saved settings,
        # followed by all the other columns (i.e. comment_id and any new columns
        # added in between the last save and now).
        static_cols = df_cols[:5]
        non_static_cols = df_cols[5:]
        sorted_cols = static_cols
        # Static columns are saved too to store their visibility, but we filter
        # them out here because they've already been added to the list.
        sorted_cols.extend([col for col in saved_cols if col not in sorted_cols])
        # Add all other unsaved columns
        sorted_cols.extend([col for col in non_static_cols if col not in saved_cols])

        self.data = self.data[sorted_cols]

        for cc in self.data:
            col_name = self.col_title_to_name(cc)
            if col_name in self._attributi and hasattr(self._attributi[col_name], 'variable_type'):
                var_type_class = self._attributi[col_name].get_type_class()
                self.data[cc] = var_type_class.convert(self.data[cc])

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
                                    [col_settings.get(col, True) for col in self.data.columns])
        self.plot.update_columns()

        # Hide the comment_id column
        self.table_view.set_column_visibility("comment_id", False, for_restore=True)

        self._tab_widget.setEnabled(True)
        self.show_default_status_message()
        self.context_dir_changed.emit(str(path))

    def column_renames(self):
        return {name: v.title for name, v in self._attributi.items() if v.title}

    def has_variable(self, name, by_title=False):
        if by_title:
            haystack = set(self._title_to_name.keys()) | set(self.data.columns.values)
        else:
            haystack = set(self._name_to_title.keys())

        return name in haystack

    def add_variable(self, name, title, variable_type, description="", before=None):
        n_static_cols = self.table_view.get_static_columns_count()
        before_pos = n_static_cols + 1
        if before == None:
            before_pos += self.table_view.get_movable_columns_count()
        else:
            before_pos += before
        variable = UserEditableVariable(name, title=title, variable_type=variable_type, description=description)
        self.db.add_user_variable(variable)
        self._attributi[name] = variable
        self._name_to_title[name] = title
        self._title_to_name[title] = name
        self.data.insert(before_pos, title, variable.get_type_class().convert(pd.Series(index=self.data.index, dtype='object')))
        self.table.insertColumn(before_pos)
        self.table_view.add_new_columns([title], [True], [before_pos - n_static_cols - 1])
        self.table.add_editable_column(title)


    def column_title(self, name):
        if name in self._attributi:
            return self._attributi[name].title or name
        return name

    def open_column_dialog(self):
        if self._columns_dialog is None:
            self._columns_dialog = QtWidgets.QDialog(self)
            self._columns_dialog.setWindowTitle("Column settings")
            layout = QtWidgets.QVBoxLayout()

            layout.addWidget(QtWidgets.QLabel("These columns can be hidden but not reordered:"))
            layout.addWidget(self.table_view._static_columns_widget)
            layout.addWidget(QtWidgets.QLabel("Drag these columns to reorder them:"))
            layout.addWidget(self.table_view._columns_widget)
            self._columns_dialog.setLayout(layout)

        self._columns_dialog.show()

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
        fileMenu.addAction(action_create_var)
        fileMenu.addAction(action_help)
        fileMenu.addAction(action_exit)

        # Table menu
        action_columns = QtWidgets.QAction("Select && reorder columns", self)
        action_columns.triggered.connect(self.open_column_dialog)
        tableMenu = menu_bar.addMenu("Table")
        tableMenu.addAction(action_columns)

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
                    if isinstance(message[col_name], np.ndarray):
                        self.data[col_name] = self.data[col_name].astype('object')

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
        comment_id = self.db.add_standalone_comment(ts, text)
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

    def col_name_to_title(self, quantity):
        res = quantity

        if quantity in self._name_to_title:
            res = self._name_to_title[quantity]

        return res

    def col_title_to_name(self, title):
        res = title

        if title in self._title_to_name:
            res = self._title_to_name[title]

        return res

    def get_variable_from_name(self, name):
        if name in self._attributi:
            return self._attributi[name]
        else:
            raise RuntimeError(f"Couldn't find variable with name '{name}'")

    def make_finite(self, data):
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        return data.astype('object').fillna(np.nan)

    def bool_to_numeric(self, data):
        if infer_dtype(data) == 'boolean':
            data = data.astype('float')

        return data

    def fix_data_for_plotting(self, data):
        return self.bool_to_numeric(self.make_finite(data))

    def inspect_data(self, index):
        proposal = self.data["Proposal"][index.row()]
        run = self.data["Run"][index.row()]

        quantity_title = self.data.columns[index.column()]
        quantity = self.col_title_to_name(quantity_title)

        # Don't try to plot strings
        if quantity_title in { "Status" } | self.table.editable_columns:
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
                x=[self.bool_to_numeric(self.fix_data_for_plotting(x))] if not is_image else [],
                y=[self.bool_to_numeric(self.fix_data_for_plotting(y))] if not is_image else [],
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
        comment_horizontal_layout = QtWidgets.QHBoxLayout()

        # the table
        self.table_view = TableView()
        self.table = Table(self)
        self.table.value_changed.connect(self.save_value)
        self.table.time_comment_changed.connect(self.save_time_comment)
        self.table.run_visibility_changed.connect(lambda row, state: self.plot.update())

        self.table_view.doubleClicked.connect(self.inspect_data)
        self.table_view.settings_changed.connect(self.save_settings)

        vertical_layout.addWidget(self.table_view)

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

        vbox = QtWidgets.QGridLayout()
        test_widget.setLayout(vbox)

        self._save_btn = QtWidgets.QPushButton("Save")
        self._save_btn.clicked.connect(self.save_context)
        self._save_btn.setToolTip("Ctrl + S")
        self._save_btn.setShortcut(QtGui.QKeySequence(Qt.ControlModifier | Qt.Key_S))

        self._check_btn = QtWidgets.QPushButton("Validate")
        self._check_btn.clicked.connect(self.test_context)

        self._context_status_icon = QtSvg.QSvgWidget()

        font = QtGui.QFont("Monospace", pointSize=9)
        self._error_widget_lexer = QsciLexerPython()
        self._error_widget_lexer.setDefaultFont(font)

        self._error_widget.setReadOnly(True)
        self._error_widget.setCaretWidth(0)
        self._error_widget.setLexer(self._error_widget_lexer)

        vbox.addWidget(self._save_btn, 0, 0)
        vbox.addWidget(self._check_btn, 1, 0)
        vbox.addWidget(self._context_status_icon, 2, 0)
        vbox.addWidget(self._error_widget, 0, 1, 3, 1)
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
        self._editor.setFocus()

    def mark_context_saved(self):
        self._context_is_saved = True
        self._tabbar_style.enable_bold = False
        self._tab_widget.setTabText(1, "Context file")
        self._tab_widget.tabBar().setTabTextColor(1, QtGui.QColor("black"))
        self._editor_status_message = str(self._context_path.resolve())
        self.on_tab_changed(self._tab_widget.currentIndex())

    def save_value(self, prop, run, column_name, value):
        if self.db is None:
            log.warning("No SQLite database in use, value not saved")
            return

        log.debug("Saving data for column %s for prop %d run %d", column_name, prop, run)
        column_title = self.col_name_to_title(column_name)
        if column_name in self.table.editable_columns or column_title in self.table.editable_columns:
            with self.db.conn:
                self.db.conn.execute(
                    "UPDATE runs set {}=? WHERE proposal=? AND runnr=?".format(column_name),
                    (value, int(prop), int(run)),
                )

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
