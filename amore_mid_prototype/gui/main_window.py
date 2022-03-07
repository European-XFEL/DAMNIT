import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from .zmq import ZmqStreamReceiver
from .table import TableView, Table
from .plot import Plot


log = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self, zmq_endpoint: str = None, filename_import_metadata: str = None,
    ):
        super().__init__()

        self.data = None
        self.zmq_endpoint = zmq_endpoint
        self.filename_import_metadata = filename_import_metadata
        self._is_zmq_receiving_data = False

        self.setWindowTitle("~ AMORE ~")
        self.resize(600, 1000)
        self._create_status_bar()
        self._create_menu_bar()

        if self.zmq_endpoint is not None:
            self._zmq_thread_launcher()

        self._view_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self._view_widget)

    def _create_status_bar(self) -> None:
        self._status_bar = QtWidgets.QStatusBar()

        self._status_bar.setStyleSheet("QStatusBar::item {border: None;}")
        self._status_bar.showMessage("Autoconfigure AMORE.")
        self.setStatusBar(self._status_bar)

        self._status_bar_connection_status = QtWidgets.QLabel()
        self._status_bar_connection_status.setStyleSheet(
            "color:green;font-weight:bold;"
        )
        self._status_bar.addPermanentWidget(self._status_bar_connection_status)

    def _menu_bar_import_file(self):

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "All files (*);;JSON files (*.json)",
        )

        if filename:
            self.filename_import_metadata = filename

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
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select context directory",
        )
        if not path:
            return

        zmq_addr_path = Path(path, ".zmq_extraction_events")
        if zmq_addr_path.is_file():
            self.zmq_endpoint = zmq_addr_path.read_text().strip()
            log.info("Connecting to %s (ZMQ)", self.zmq_endpoint)
            self._zmq_thread_launcher()
        else:
            log.warning("No .zmq_extraction_events file in context folder")
            self._status_bar_connection_status.setText("No ZMQ socket found in folder")

        sqlite_path = Path(path, "runs.sqlite")
        if sqlite_path.is_file():
            log.info("Reading data from database")
            db = sqlite3.connect(sqlite_path)
            df = pd.read_sql_query("SELECT * FROM runs", db)
            df.migrated_at /= 1000  # ms -> s
            self.data = df.rename(
                columns={
                    "runnr": "Run",
                    "proposal": "Proposal",
                    "migrated_at": "Timestamp",
                    "comment": "Comment",
                }
            )
            self._create_view()

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

        fileMenu = menu_bar.addMenu("&AMORE")
        fileMenu.addAction(action_autoconfigure)
        fileMenu.addAction(action_connect)
        fileMenu.addAction(action_help)
        fileMenu.addAction(action_exit)

    def zmq_get_data_and_update(self, message):

        # is the message OK?
        if "Run" not in message.keys():
            raise ValueError("Malformed message.")

        # log.info("Updating for ZMQ message: %s", message)

        # initialize the view
        if not self._is_zmq_receiving_data:
            self._is_zmq_receiving_data = True
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
                log.debug("New row in table")
                self.data = pd.concat(
                    [
                        self.data,
                        pd.DataFrame(
                            {**message, **{"Comment": ""}},
                            index=[self.table.rowCount()],
                        ),
                    ]
                )
                self.table.data = self.data
                if len(self.table.is_sorted_by):
                    self.table.data.sort_values(
                        self.table.is_sorted_by,
                        ascending=self.table.is_sorted_order
                        == QtCore.Qt.AscendingOrder,
                        inplace=True,
                    )
                    self.table.data.reset_index(inplace=True, drop=True)

                self.table.insertRows(self.table.rowCount())

            if new_cols:
                log.info("New columns for table: %s", new_cols)
                self.table.insertColumns(self.table.columnCount() - 1)

                self.plot.update_combo_box(new_cols)

                self.table_view.set_item_columns_visibility(
                    list(new_cols), [True for _ in list(new_cols)]
                )

        # update plots
        self.plot.data = self.data
        self.plot.update()

    def _zmq_thread_launcher(self) -> None:
        self._zmq_thread = QtCore.QThread()
        self.zeromq_listener = ZmqStreamReceiver(self.zmq_endpoint)
        self.zeromq_listener.moveToThread(self._zmq_thread)

        self._zmq_thread.started.connect(self.zeromq_listener.loop)
        self.zeromq_listener.message.connect(self.zmq_get_data_and_update)
        QtCore.QTimer.singleShot(0, self._zmq_thread.start)

    def _create_view(self) -> None:
        vertical_layout = QtWidgets.QVBoxLayout()
        table_horizontal_layout = QtWidgets.QHBoxLayout()
        comment_horizontal_layout = QtWidgets.QHBoxLayout()
        plot_horizontal_layout = QtWidgets.QHBoxLayout()

        # the table
        self.table_view = TableView()
        self.table = Table(self.data)
        self.table_view.setModel(self.table)

        table_horizontal_layout.addWidget(self.table_view, stretch=6)
        table_horizontal_layout.addLayout(
            self.table_view.set_columns_visibility(
                self.table.data.columns, [True for _ in self.table.data.columns]
            ),
            stretch=1,
        )

        vertical_layout.addLayout(table_horizontal_layout)

        # comments
        # self.nameLabel = QLabel(self)
        # self.nameLabel.setText('Name:')

        # comment_horizontal_layout.addWidget(self.nameLabel)

        # vertical_layout.addLayout(comment_horizontal_layout)

        # plotting control
        self.plot = Plot(self.data)

        plot_horizontal_layout.addWidget(self.plot._button_plot)
        plot_horizontal_layout.addStretch()

        plot_horizontal_layout.addWidget(self.plot._combo_box_x_axis)
        plot_horizontal_layout.addWidget(QtWidgets.QLabel("vs."))
        plot_horizontal_layout.addWidget(self.plot._combo_box_y_axis)

        vertical_layout.addLayout(plot_horizontal_layout)

        self._view_widget.setLayout(vertical_layout)


def main():
    logging.basicConfig(level=logging.DEBUG)
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar
    )
    application = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()
