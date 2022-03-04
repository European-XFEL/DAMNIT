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
        self,
        zmq_endpoint: str = None,
        filename_import_metadata: str = None,
        filename_export_metadata: str = "AMORE.csv",
    ):
        super().__init__()

        self.data = None
        self.zmq_endpoint = zmq_endpoint
        self.filename_import_metadata = filename_import_metadata
        self.filename_export_metadata = filename_export_metadata
        self._is_zmq_receiving_data = False

        self.setWindowTitle("~ AMORE ~")
        self.resize(800, 1000)
        self._create_status_bar()
        self._create_menu_bar()

        if self.zmq_endpoint is not None:
            self._zmq_thread_launcher()

        self._view_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self._view_widget)

    def _create_status_bar(self) -> None:
        self._status_bar = QtWidgets.QStatusBar()

        self._status_bar.setStyleSheet("QStatusBar::item {border: None;}")
        self._status_bar.showMessage(
            "Connect to AMORE backend or import existing metadata."
        )
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

    def _menu_bar_export_file(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "QFileDialog.getSaveFileName()",
            "",
            "All files (*);;JSON files (*.json)",
        )

        if filename:
            self.filename_export_metadata = filename

    def _menu_bar_help(self) -> None:
        dialog = QtWidgets.QMessageBox(self)

        font = QtGui.QFont()
        font.setBold(False)
        dialog.setFont(font)

        dialog.setWindowTitle("Getting help!")
        dialog.setText(
            """To start inspecting experimental results,
connect to the AMORE backend or import existing metadata.
    
If you experience any issue, please contact us at:
da-dev@xfel.eu"""
        )
        dialog.exec()

    def _menu_bar_connect(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select context directory",
        )
        if not path:
            return

        zmq_addr_path = Path(path, '.zmq_extraction_events')
        if zmq_addr_path.is_file():
            self.zmq_endpoint = zmq_addr_path.read_text().strip()
            log.info("Connecting to %s (ZMQ)", self.zmq_endpoint)
            self._zmq_thread_launcher()
        else:
            log.warning("No .zmq_extraction_events file in context folder")
            self._status_bar_connection_status.setText("No ZMQ socket found in folder")

        sqlite_path = Path(path, 'runs.sqlite')
        if sqlite_path.is_file():
            log.info("Reading data from database")
            db = sqlite3.connect(sqlite_path)
            df = pd.read_sql_query('SELECT * FROM runs', db)
            df.migrated_at /= 1000  # ms -> s
            self.data = df.rename(columns={
                'runnr': 'Run', 'proposal': 'Proposal',
                'migrated_at': 'Timestamp', 'comment': 'Comment'
            })
            self._create_view()

    def _create_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)

        action_connect = QtWidgets.QAction(QtGui.QIcon("connect.png"), "&Connect", self)
        action_connect.setShortcut("Shift+C")
        action_connect.setStatusTip("Connect to AMORE server.")
        action_connect.triggered.connect(self._menu_bar_connect)

        action_import = QtWidgets.QAction(QtGui.QIcon("import.png"), "&Import", self)
        action_import.setShortcut("Shift+I")
        action_import.setStatusTip("Import metadata.")
        action_import.triggered.connect(self._menu_bar_import_file)

        action_export = QtWidgets.QAction(QtGui.QIcon("export.png"), "&Export", self)
        action_export.setShortcut("Shift+E")
        action_export.setStatusTip("Export metadata.")
        action_export.triggered.connect(self._menu_bar_export_file)

        action_help = QtWidgets.QAction(QtGui.QIcon("help.png"), "&Help", self)
        action_help.setShortcut("Shift+H")
        action_help.setStatusTip("Get help.")
        action_help.triggered.connect(self._menu_bar_help)

        action_exit = QtWidgets.QAction(QtGui.QIcon("exit.png"), "&Exit", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.setStatusTip("Exit AMORE GUI.")
        action_exit.triggered.connect(QtWidgets.QApplication.instance().quit)

        fileMenu = menu_bar.addMenu("&AMORE")
        fileMenu.addAction(action_connect)
        fileMenu.addAction(action_import)
        fileMenu.addAction(action_export)
        fileMenu.addAction(action_help)
        fileMenu.addAction(action_exit)

        fileMenu = menu_bar.addMenu("&View")
        # fileMenu.addAction(action_set_columns)

    def zmq_get_data_and_update(self, message):

        # is the message OK?
        if "Run" not in message.keys():
            raise ValueError("Malformed message.")

        log.info("Updating for ZMQ message: %s", message)

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

            if row.size:
                log.debug("Update existing row %s for run %s",
                          row.index, message['Run'])
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
                self.table.insertRows(self.table.rowCount())

            new_cols = set(message) - set(self.data.columns)
            if new_cols:
                log.info("New columns for table: %s", new_cols)
                self.table.insertColumns(self.table.columnCount() - 1, len(new_cols))
                self.plot.update_combo_box(new_cols)

        # update plots
        self.plot.data = self.data
        self.plot.update()

        # (over)write down metadata
        self.data.to_json(self.filename_export_metadata)

    def _zmq_thread_launcher(self) -> None:
        self._zmq_thread = QtCore.QThread()
        self.zeromq_listener = ZmqStreamReceiver(self.zmq_endpoint)
        self.zeromq_listener.moveToThread(self._zmq_thread)

        self._zmq_thread.started.connect(self.zeromq_listener.loop)
        self.zeromq_listener.message.connect(self.zmq_get_data_and_update)
        QtCore.QTimer.singleShot(0, self._zmq_thread.start)

    def _create_view(self) -> None:
        vertical_layout = QtWidgets.QVBoxLayout()
        horizontal_layout = QtWidgets.QHBoxLayout()

        # the table
        self.table_view = TableView()
        self.table = Table(self.data)
        self.table_view.setModel(self.table)

        vertical_layout.addWidget(self.table_view)

        # comments
        # self.

        # plotting control
        self.plot = Plot(self.data)

        horizontal_layout.addWidget(self.plot._button_plot)
        horizontal_layout.addStretch()

        horizontal_layout.addWidget(self.plot._combo_box_x_axis)
        horizontal_layout.addWidget(QtWidgets.QLabel("vs."))
        horizontal_layout.addWidget(self.plot._combo_box_y_axis)

        vertical_layout.addLayout(horizontal_layout)

        self._view_widget.setLayout(vertical_layout)

def main():
    logging.basicConfig(level=logging.INFO)
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar
    )
    application = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(application.exec())

if __name__ == "__main__":
    main()
