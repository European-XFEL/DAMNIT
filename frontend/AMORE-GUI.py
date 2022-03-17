import sys
import time
import pandas as pd
import numpy as np
from PyQt6 import Qt, QtCore, QtGui, QtWidgets

from IO.zmq import ZmqStreamReceiver
from GUI.table import TableView, Table
from GUI.plot import Plot


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
        text, status_ok = QtWidgets.QInputDialog.getText(
            self, "Connect to AMORE backend", "Endpoint (e.g. tcp://localhost:5555):"
        )
        if status_ok:
            self.zmq_endpoint = str(text)
            self._zmq_thread_launcher()

    def _create_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)

        action_connect = QtGui.QAction(QtGui.QIcon("connect.png"), "&Connect", self)
        action_connect.setShortcut("Shift+C")
        action_connect.setStatusTip("Connect to AMORE server.")
        action_connect.triggered.connect(self._menu_bar_connect)

        action_import = QtGui.QAction(QtGui.QIcon("import.png"), "&Import", self)
        action_import.setShortcut("Shift+I")
        action_import.setStatusTip("Import metadata.")
        action_import.triggered.connect(self._menu_bar_import_file)

        action_export = QtGui.QAction(QtGui.QIcon("export.png"), "&Export", self)
        action_export.setShortcut("Shift+E")
        action_export.setStatusTip("Export metadata.")
        action_export.triggered.connect(self._menu_bar_export_file)

        action_help = QtGui.QAction(QtGui.QIcon("help.png"), "&Help", self)
        action_help.setShortcut("Shift+H")
        action_help.setStatusTip("Get help.")
        action_help.triggered.connect(self._menu_bar_help)

        action_exit = QtGui.QAction(QtGui.QIcon("exit.png"), "&Exit", self)
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

        # initialize the view
        if not self._is_zmq_receiving_data:
            self._is_zmq_receiving_data = True
            self._status_bar_connection_status.setText(self.zmq_endpoint)

            # ingest data
            self.data = pd.DataFrame({**message, **{"Comment": ""}}, index=[0])

            # build the table
            self._create_view()

        else:

            for ki in message.keys():
                if ki not in self.data.columns:
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

                    self.table.insertColumns(self.table.columnCount() - 1)
                    self.plot.update_combo_box((ki,))

            # is the run already in?
            row = self.data.loc[self.data["Run"] == message["Run"]]

            if row.size:
                for ki, vi in message.items():
                    self.data.at[row.index[0], ki] = vi

                    index = self.table.index(
                        row.index[0], self.data.columns.get_loc(ki)
                    )
                    self.table.dataChanged.emit(index, index)

            else:
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


if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar
    )
    application = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(application.exec())
