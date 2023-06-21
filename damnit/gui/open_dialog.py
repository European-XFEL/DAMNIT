from pathlib import Path

from extra_data.read_machinery import find_proposal
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog, QFileDialog, QDialogButtonBox

from .open_dialog_ui import Ui_Dialog

class ProposalFinder(QObject):
    find_result = pyqtSignal(str, str)

    def find_proposal(self, propnum: str):
        if propnum.isdecimal() and len(propnum) >= 4:
            try:
                dir = find_proposal(f"p{int(propnum):06}")
            except:
                dir = ''
        else:
            dir = ''
        self.find_result.emit(propnum, dir)

class OpenDBDialog(QDialog):
    proposal_num_changed = pyqtSignal(str)
    proposal_dir = ''

    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.ui.proposal_rb.toggled.connect(self.update_ok)
        self.ui.folder_edit.textChanged.connect(self.update_ok)
        self.ui.browse_button.clicked.connect(self.browse_for_folder)
        self.ui.proposal_edit.setFocus()

        self.proposal_finder_thread = QThread()
        self.proposal_finder = ProposalFinder()
        self.proposal_finder.moveToThread(self.proposal_finder_thread)
        self.ui.proposal_edit.textChanged.connect(self.proposal_finder.find_proposal)
        self.proposal_finder.find_result.connect(self.proposal_dir_result)
        self.finished.connect(self.proposal_finder_thread.quit)
        self.proposal_finder_thread.start()

    def proposal_dir_result(self, propnum: str, dir: str):
        if propnum != self.ui.proposal_edit.text():
            return  # Text field has been changed
        self.proposal_dir = dir
        self.update_ok()

    def update_ok(self):
        if self.ui.proposal_rb.isChecked():
            valid = bool(self.proposal_dir)
        else:
            valid = Path(self.ui.folder_edit.text()).is_dir()
        self.ui.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)

    def browse_for_folder(self):
        path = QFileDialog.getExistingDirectory()
        if path:
            self.ui.folder_edit.setText(path)

    def get_chosen_dir(self):
        if self.ui.proposal_rb.isChecked():
            return Path(self.proposal_dir, "usr/Shared/amore")
        else:
            return Path(self.ui.folder_edit.text())

def select_amore_dir():
    dlg = QDialog()
    ui = Ui_Dialog()
    ui.setupUi(dlg)
    ui.browse_button.clicked.connect()

    dlg.exec()
