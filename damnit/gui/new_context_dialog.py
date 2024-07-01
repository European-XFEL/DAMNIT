import re
from pathlib import Path
from typing import Optional

from extra_data.read_machinery import find_proposal
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QFileDialog, QListWidgetItem, QMessageBox

from .new_context_dialog_ui import Ui_Dialog

DAMNIT_PKG = Path(__file__).parent.parent

INST_TO_SASE = {
    'FXE': 'SA1', 'SPB': 'SA1',
    'HED': 'SA2', 'MID': 'SA2',
    'SCS': 'SA3',  'SQS': 'SA3', 'SXP': 'SA3',
}

ALL_GROUPS = set(INST_TO_SASE) | set(INST_TO_SASE.values())

def find_instrument(path: Path) -> str:
    if path.is_relative_to('/gpfs/exfel/exp'):
        return path.parts[4]  # Part after exp/
    elif path.is_relative_to('/gpfs/exfel/u'):
        return path.parts[5]  # Part after usr/ or scratch/
    return ''


class NewContextFileDialog(QDialog):
    def __init__(self, target_path, parent=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        group = find_instrument(target_path)
        if group in ALL_GROUPS:
            self.match_groups = {group, None}
            if group in INST_TO_SASE:
                self.match_groups.add(INST_TO_SASE[group])
        else:
            self.match_groups = ALL_GROUPS | {None}

        self.all_templates = []

        group_re = re.compile(r'([a-zA-Z0-9]+)[ -_]')
        for path in sorted((DAMNIT_PKG / 'ctx-templates').iterdir()):
            if (m := group_re.match(path.name)) and m[1] in ALL_GROUPS:
                group = m[1]
            else:
                group = None
            self.all_templates.append((group, path))

        self.populate_template_list()

        self.ui.template_other_inst_cb.toggled.connect(self.populate_template_list)
        self.ui.browse_button.clicked.connect(self.browse)

    def populate_template_list(self, all_insts=False):
        self.ui.template_list.clear()

        for group, path in self.all_templates:
            if (not all_insts) and (group not in self.match_groups):
                continue

            item = QListWidgetItem(path.stem, parent=self.ui.template_list)
            item.setData(Qt.ItemDataRole.UserRole, str(path))

        if self.ui.template_list.count() > 0:
            self.ui.template_list.setCurrentRow(0)

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Python file", filter="Python files (*.py);;Any (*)"
        )
        if path:
            self.ui.file_edit.setText(path)

    def run_get_result(self) -> Optional[Path]:
        if self.exec() == QDialog.Rejected:
            return None

        if self.ui.template_rb.isChecked():
            item = self.ui.template_list.currentItem()
            return Path(item.data(Qt.ItemDataRole.UserRole))

        elif self.ui.proposal_rb.isChecked():
            propnum = self.ui.proposal_edit.text()
            try:
                prop_dir = find_proposal(f"p{int(propnum):06}")
            except Exception:
                QMessageBox.critical(self, "Proposal not found",
                                     f"Could not find proposal {propnum}")
                return self.run_get_result()
            path = Path(prop_dir) / 'usr/Shared/amore/context.py'
            if not path.is_file():
                QMessageBox.critical(self, "No context file",
                                     f"Proposal {propnum} didn't contain a context file")
                return self.run_get_result()
            return path

        else:  # file_rb
            return Path(self.ui.file_edit.text())
