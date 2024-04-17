from pathlib import Path
from socket import gethostname
from typing import Optional, Tuple

from extra_data.read_machinery import find_proposal
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFileDialog

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.ui.proposal_rb.toggled.connect(self.update_ok)
        self.ui.folder_edit.textChanged.connect(self.update_ok)
        self.ui.browse_button.clicked.connect(self.browse_for_folder)
        self.ui.proposal_edit.setFocus()

        self.proposal_finder_thread = QThread(parent=parent)
        self.proposal_finder = ProposalFinder()
        self.proposal_finder.moveToThread(self.proposal_finder_thread)
        self.ui.proposal_edit.textChanged.connect(self.proposal_finder.find_proposal)
        self.proposal_finder.find_result.connect(self.proposal_dir_result)
        self.finished.connect(self.proposal_finder_thread.quit)
        self.proposal_finder_thread.finished.connect(self.proposal_finder_thread.deleteLater)

    def run_get_result(self) -> Tuple[Optional[Path], Optional[int]]:
        self.proposal_finder_thread.start()
        if self.exec() == QDialog.Rejected:
            return None, None
        context_dir = self.get_chosen_dir()
        prop_no = self.get_proposal_num()

        # use separated directory if running online to avoid file corruption
        # during sync between clusters.
        if (
                gethostname().startswith('exflonc')
                and not context_dir.stem.endswith('-online')
        ):
            context_dir = context_dir.absolute().parent / f'{context_dir.stem}-online'

        return context_dir, prop_no

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
            return Path(self.proposal_dir) / "usr/Shared/amore"
        else:
            return Path(self.ui.folder_edit.text())

    def get_proposal_num(self) -> Optional[int]:
        if self.ui.proposal_rb.isChecked():
            return int(self.ui.proposal_edit.text())
        return None
