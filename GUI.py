import sys
from PyQt6 import QtCore, QtGui, QtWidgets

from IO.zmq import ZmqReceiver


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.filename_import_metadata = None
        self.filename_export_metadata = None

        self.setWindowTitle("~ AMORE ~")
        self.resize(500, 800)
        self._create_status_bar()
        self._create_menu_bar()

        # main_layout = QtWidgets.QWidget(self)
        # self.setCentralWidget(main_layout)

        # vertical_layout = QtWidgets.QVBoxLayout()
        # main_layout.setLayout(vertical_layout)

        # main_layout.addWidget(self._set_menu_bar())

        # self.setLayout(vertical_layout)

    def _create_status_bar(self) -> None:
        self._status_bar = QtWidgets.QStatusBar()

        self._status_bar.setStyleSheet("QStatusBar::item {border: None;}")
        self._status_bar.showMessage(
            "Connect to AMORE backend or import existing metadata."
        )
        self.setStatusBar(self._status_bar)

        self._status_bar_connection_status = QtWidgets.QLabel()
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
connect to the AMORE backend (e.g. tcp://localhost:5555) or import existing metadata.
    
If you experience any issue, please contact us at:
da-dev@xfel.eu"""
        )
        dialog.exec()

    def _menu_bar_connect(self) -> None:
        text, status_ok = QtWidgets.QInputDialog.getText(
            self, "Connect to AMORE backend", "Endpoint:"
        )
        if status_ok:
            self.zmq_endpoint = str(text)

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
        #fileMenu.addAction(action_set_columns)


if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar
    )
    application = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(application.exec())
