import os
import logging
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import h5py
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from extra_data.read_machinery import find_proposal

from ..backend.db import open_db
from ..context import ContextFile
from .zmq import ZmqStreamReceiver
from .table import TableView, Table
from .plot import Canvas, Plot


log = logging.getLogger(__name__)
pd.options.mode.use_inf_as_na = True


class MainWindow(QtWidgets.QMainWindow):
    db = None

    def __init__(
        self, zmq_endpoint: str = None, context_dir: Path = None,
    ):
        super().__init__()

        self.data = None
        self._zmq_thread = None
        self.zmq_endpoint = zmq_endpoint
        self._zmq_thread = None
        self._is_zmq_receiving_data = False
        self._attributi = {}

        self.setWindowTitle("Data And Metadata iNspection Interactive Thing")
        self.setWindowIcon(QtGui.QIcon("amore_mid_prototype/gui/ico/AMORE.png"))
        self._create_status_bar()
        self._create_menu_bar()

        self._view_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self._view_widget)

        if context_dir is not None:
            self.autoconfigure(context_dir)
        elif self.zmq_endpoint is not None:
            self._zmq_thread_launcher()

        self._canvas_inspect = []

        self.center_window()

    def closeEvent(self, event):
        if self._zmq_thread is not None:
            self._zmq_thread.exit()

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
        if os.path.isdir("/gpfs/exfel/exp"):
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
                except Exception as e:
                    button = QtWidgets.QMessageBox.warning(self, "Bad proposal number",
                                                           "Could not find a proposal with this number, try again?",
                                                           buttons=QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                    if button != QtWidgets.QMessageBox.Yes:
                        prompt = False

        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select context directory", proposal_dir
        )
        if path:
            self.autoconfigure(Path(path))

    def autoconfigure(self, path: Path):
        context_path = path / "context.py"
        if context_path.is_file():
            log.info("Reading context file %s", context_path)
            ctx_file = ContextFile.from_py_file(context_path)
            self._attributi = ctx_file.vars

        zmq_addr_path = path / ".zmq_extraction_events"
        if zmq_addr_path.is_file():
            self.zmq_endpoint = zmq_addr_path.read_text().strip()
            log.info("Connecting to %s (ZMQ)", self.zmq_endpoint)
            self._zmq_thread_launcher()
        else:
            log.warning("No .zmq_extraction_events file in context folder")
            self._status_bar_connection_status.setStyleSheet(
                "color:red;font-weight:bold;"
            )
            self._status_bar_connection_status.setText("No ZMQ socket found in folder")

        self.extracted_data_template = str(path / "extracted_data/p{}_r{}.h5")

        sqlite_path = path / "runs.sqlite"
        if sqlite_path.is_file():
            log.info("Reading data from database")
            self.db = open_db(sqlite_path)
            df = pd.read_sql_query("SELECT * FROM runs", self.db)
            df.insert(0, "Status", True)
            df.insert(len(df.columns), "_comment_id", pd.NA)

            # Read the comments and prepare them for merging with the main data
            comments_df = pd.read_sql_query(
                "SELECT rowid as _comment_id, * FROM time_comments", self.db
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
                            "added_at": "_added_at",
                            **self.column_renames(),
                        }
                    ),
                    comments_df.rename(
                        columns={"timestamp": "Timestamp", "comment": "Comment",}
                    ),
                ]
            )

            # move some columns
            if "Timestamp" in self.data.columns:
                self.data.insert(0, "Timestamp", self.data.pop("Timestamp"))

            if "Status" in self.data.columns:
                self.data.insert(1, "Status", self.data.pop("Status"))
            
            if "Run" in self.data.columns:
                self.data.insert(2, "Run", self.data.pop("Run"))
            
            if "Comment" in self.data.columns:
                self.data.insert(3, "Comment", self.data.pop("Comment"))

            self._create_view()
            self.table_view.sortByColumn(self.data.columns.get_loc("Timestamp"),
                                         Qt.SortOrder.AscendingOrder)
            
            # hide some column
            for column in self.data.columns:
                if column.startswith("_"):
                    self.table_view.setColumnHidden(self.data.columns.get_loc(column), True)

        self._status_bar.showMessage("Double-click on a cell to inspect results.")

    def column_renames(self):
        return {name: v.title for name, v in self._attributi.items() if v.title}

    def column_title(self, name):
        if name in self._attributi:
            return self._attributi[name].title or name
        return name

    def _menu_bar_connect(self) -> None:
        text, status_ok = QtWidgets.QInputDialog.getText(
            self, "Connect to AMORE backend", "Endpoint (e.g. tcp://localhost:5555):"
        )
        if status_ok:
            self.zmq_endpoint = str(text)
            self._zmq_thread_launcher()

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

        action_connect = QtWidgets.QAction(
            QtGui.QIcon("connect.png"), "Connect with &endpoint", self
        )
        action_connect.setShortcut("Shift+E")
        action_connect.setStatusTip("Connect to AMORE server using a 0mq endpoint.")
        action_connect.triggered.connect(self._menu_bar_connect)

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
        fileMenu.addAction(action_connect)
        fileMenu.addAction(action_help)
        fileMenu.addAction(action_exit)

    def zmq_get_data_and_update(self, message):

        # is the message OK?
        if "Run" not in message.keys():
            raise ValueError("Malformed message.")

        # log.info("Updating for ZMQ message: %s", message)

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
        if not self._is_zmq_receiving_data:
            self._is_zmq_receiving_data = True
            self._status_bar_connection_status.setStyleSheet(
                "color:green;font-weight:bold;"
            )
            self._status_bar_connection_status.setText(self.zmq_endpoint)

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

                self.plot.update_combo_box(new_cols)

                self.table_view.set_item_columns_visibility(
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

                new_df = pd.concat(
                    [
                        self.data.iloc[:ix],
                        pd.DataFrame(
                            {**message, **{"Comment": ""}},
                            index=[self.table.rowCount()],
                        ),
                        self.data.iloc[ix:],
                    ],
                    ignore_index=True,
                )

                self.table.beginInsertRows(QtCore.QModelIndex(), ix, ix)
                self.data = new_df
                self.table.endInsertRows()

        # update plots
        self.plot.update()

        # (over)write down metadata
        self.data.to_json("AMORE.json")

    def _zmq_thread_launcher(self) -> None:
        self._zmq_thread = QtCore.QThread()
        self.zeromq_listener = ZmqStreamReceiver(self.zmq_endpoint)
        self.zeromq_listener.moveToThread(self._zmq_thread)

        self._zmq_thread.started.connect(self.zeromq_listener.loop)
        self.zeromq_listener.message.connect(self.zmq_get_data_and_update)
        QtCore.QTimer.singleShot(0, self._zmq_thread.start)

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

    def get_run_file(self, proposal, run):
        file_name = self.extracted_data_template.format(proposal, run)

        try:
            run_file = h5py.File(file_name)
            return file_name, run_file
        except FileNotFoundError as e:
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

        try:
            file_name, dataset = self.get_run_file(proposal, run)
        except:
            return

        try:
            x, y = dataset[quantity]["trainId"][:], dataset[quantity]["data"][:]
        except KeyError:
            log.warning("'{}' not found in {}...".format(quantity, file_name))
            return
        finally:
            dataset.close()

        self._canvas_inspect.append(
            Canvas(
                self,
                x=[self.make_finite(x)],
                y=[self.make_finite(y)],
                xlabel="Event (run {})".format(run),
                ylabel=quantity_title,
                fmt="ro",
                autoscale=False,
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
        self.table_view.setModel(self.table)

        # Always keep these columns as small as possible to save space
        header = self.table_view.horizontalHeader()
        for column in ["Status", "Run", "Timestamp"]:
            column_index = self.data.columns.get_loc(column)
            header.setSectionResizeMode(column_index, QtWidgets.QHeaderView.ResizeToContents)

        self.table_view.doubleClicked.connect(self.inspect_data)

        table_horizontal_layout.addWidget(self.table_view, stretch=6)
        table_horizontal_layout.addWidget(
            self.table_view.set_columns_visibility(
                [self.column_title(c) for c in self.data.columns],
                [True for _ in self.data.columns],
            ),
            stretch=1,
        )

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


def run_app(context_dir):
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar
    )
    application = QtWidgets.QApplication(sys.argv)
    application.setStyle(TableViewStyle())

    window = MainWindow(context_dir=context_dir)
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
