import os.path

from extra_data.read_machinery import find_proposal
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QDialog, QFileDialog, QDialogButtonBox

from .open_dialog_ui import Ui_Dialog

class OpenDBDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.ui.proposal_rb.toggled.connect(self.update_ok)
        self.ui.proposal_edit.textChanged(self.update_ok)
        self.ui.folder_edit.textChanged(self.update_ok)
        self.ui.browse_button.clicked.connect(self.browse_for_folder)
        self.ui.proposal_edit.setValidator(QIntValidator(1000, 999999))
        self.ui.proposal_edit.setFocus()

    def get_proposal_dir(self):
        if not self.ui.proposal_edit.hasAcceptableInput():
            return None
        prop_no = int(self.ui.proposal_edit.text())
        return find_proposal(f"p{prop_no:06}")

    def update_ok(self):
        dir = self.get_chosen_dir()
        valid = (dir is not None) and os.path.isdir(dir)
        self.ui.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)

    def browse_for_folder(self):
        path = QFileDialog.getExistingDirectory()
        if path:
            self.ui.folder_edit.setText()

    def get_chosen_dir(self):
        if self.ui.proposal_rb.isChecked():
            return self.get_proposal_dir()
        else:
            return self.ui.folder_edit.text()

def select_amore_dir():
    dlg = QDialog()
    ui = Ui_Dialog()
    ui.setupUi(dlg)
    ui.browse_button.clicked.connect()

    dlg.exec()
