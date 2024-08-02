import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialogButtonBox

from extra_data.read_machinery import find_proposal
from natsort import natsorted
from superqt.utils import thread_worker
from superqt import QSearchableListWidget

from ..backend.extraction_control import ExtractionRequest
from ..backend.extract_data import get_context_file
from ..context import RunData
from .widgets import QtWaitingSpinner

log = logging.getLogger(__name__)

run_range_re = re.compile(r"(\d+)(-\d+)?$")

RUNS_MSG = "Enter run numbers & ranges e.g. '17, 20-32'"


@dataclass
class VariableInfo:
    name: str = None
    title: str = None
    selected: bool = True


@dataclass
class ContextFileInfo:
    file_path: Path = None
    mtime: float = None
    error: bool = False
    variables: defaultdict = field(default_factory=partial(defaultdict, VariableInfo))

    def sorted_vars(self):
        return [v for v in natsorted(self.variables.values(), key=lambda e: e.title)]

ctx_info = ContextFileInfo()


@thread_worker
def get_context_file_vars(ctx_path: Path, python_path: str = None):
    try:
        ctx, err = get_context_file(ctx_path, python_path)
    except Exception:
        err = True
    else:
        if ctx is not None:
            # remove variables if they are not in the context file
            for var in set(ctx_info.variables).difference(ctx.vars):
                ctx_info.variables.pop(var)

            for var in natsorted(ctx.vars.values(), key=lambda e: e.title):
                # try to get the exising var info, to keep the checkbox state
                vi = ctx_info.variables.setdefault(var.name, VariableInfo(var.name, var.title))
                # update the title in case it has changed
                vi.title = var.title
    ctx_info.error = err


def parse_run_ranges(ranges: str) -> list[int]:
    res = []
    for run_s in ranges.split(","):
        if m := run_range_re.match(run_s.strip()):
            start = int(m[1])
            if m[2] is None:
                end = start
            else:
                end = int(m[2][1:])
                if start > end:
                    return []
            res.extend(range(start, end + 1))
        else:
            return []

    return res


def fmt_run_ranges(run_nums: list[int]) -> str:
    if not run_nums:
        return ""

    range_starts, range_ends = [run_nums[0]], []
    current_range_end = run_nums[0]
    for r in run_nums[1:]:
        if r > current_range_end + 1:
            range_ends.append(current_range_end)
            range_starts.append(r)
        current_range_end = r
    range_ends.append(current_range_end)

    s_pieces = []
    for start, end in zip(range_starts, range_ends):
        if start == end:
            s_pieces.append(str(start))
        else:
            s_pieces.append(f"{start}-{end}")

    return ", ".join(s_pieces)


def find_runs(runs: list[int], propnum: str) -> list[int]:
    try:
        prop_dir = Path(find_proposal(f"p{int(propnum):06}"))
        raw_runs = {p.name for p in (prop_dir / 'raw').iterdir()}
    except:  # E.g. propnum is not numeric or permission denied
        return []

    return [run for run in runs if f'r{run:04}' in raw_runs]


class ProcessingDialog(QtWidgets.QDialog):
    selected_runs = ()
    all_vars_selected = False
    no_vars_selected = False

    def __init__(self, proposal: str, runs: list[int], parent=None):
        super().__init__(parent)

        self.setWindowTitle("Process runs")
        self.setSizeGripEnabled(True)
        self.setMinimumSize(300, 200)

        main_vbox = QtWidgets.QVBoxLayout()
        self.setLayout(main_vbox)

        hbox1 = QtWidgets.QHBoxLayout()
        main_vbox.addLayout(hbox1)

        grid1 = QtWidgets.QGridLayout()
        hbox1.addLayout(grid1)
        vbox2 = QtWidgets.QVBoxLayout()
        hbox1.addLayout(vbox2)

        self.edit_prop = QtWidgets.QLineEdit(proposal)
        #self.edit_prop.setInputMask('999900;_')  # 4-6 digits
        grid1.addWidget(QtWidgets.QLabel("Proposal:"), 0, 0)
        grid1.addWidget(self.edit_prop, 0, 1)

        self.edit_runs = QtWidgets.QLineEdit(fmt_run_ranges(runs))
        self.edit_runs.textChanged.connect(self.validate_runs)
        self.runs_hint = QtWidgets.QLabel(RUNS_MSG)
        self.runs_hint.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.runs_hint.setWordWrap(True)
        grid1.addWidget(QtWidgets.QLabel("Runs:"), 1, 0)
        grid1.addWidget(self.edit_runs, 1, 1)
        grid1.addWidget(self.runs_hint, 2, 0, 1, 2)

        self.vars_list = QSearchableListWidget()
        self.vars_list.filter_widget.setPlaceholderText("Filter variables")
        self.vars_list.layout().setContentsMargins(0, 0, 0, 0)
        vbox2.addWidget(self.vars_list)

        self.btn_select_all = QtWidgets.QPushButton("Select all")
        self.btn_select_all.clicked.connect(self.select_all)
        self.btn_deselect_all = QtWidgets.QPushButton("Deselect all")
        self.btn_deselect_all.clicked.connect(self.deselect_all)
        hbox_select_btns = QtWidgets.QHBoxLayout()
        hbox_select_btns.addWidget(self.btn_select_all)
        hbox_select_btns.addWidget(self.btn_deselect_all)
        vbox2.addLayout(hbox_select_btns)

        self.dlg_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.dlg_buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.dlg_buttons.accepted.connect(self.accept)
        self.dlg_buttons.rejected.connect(self.reject)
        main_vbox.addWidget(self.dlg_buttons)

        context_python = self.parent().db.metameta.get("context_python")
        context_file = Path(self.parent()._context_path)

        vars_getter = None
        if ctx_info.file_path != context_file:
            ctx_info.file_path = context_file
            ctx_info.mtime = context_file.stat().st_mtime
            ctx_info.variables.clear()
            vars_getter = get_context_file_vars(self.parent()._context_path, context_python)
        elif ctx_info.mtime != context_file.stat().st_mtime:
            vars_getter = get_context_file_vars(self.parent()._context_path, context_python)

        self.spinner = QtWaitingSpinner(self.vars_list.list_widget, modality=Qt.ApplicationModal)
        if vars_getter is not None:
            vars_getter.returned.connect(self.update_list_items)
            vars_getter.start()
            self.spinner.start()
        else:
            self.update_list_items()

        self.validate_runs()

        self.edit_runs.setFocus()

    def _stop_spinner(self):
        if self.spinner.isSpinning():
            self.spinner.stop()
            self.spinner.hide()

    def update_list_items(self):
        self._stop_spinner()

        self.vars_list.clear()

        for var in ctx_info.sorted_vars():
            item = QtWidgets.QListWidgetItem(var.title)
            item.setData(Qt.UserRole, var.name)
            item.setCheckState(Qt.Checked if var.selected else Qt.Unchecked)
            self.vars_list.addItem(item)

        self.validate_vars()

        if ctx_info.error:
            self.runs_hint.setText(
                f'{self.runs_hint.text()}\n\n'
                '/!\\ context file contains error! Check the editor'
            )

    def validate_runs(self):
        runs = parse_run_ranges(self.edit_runs.text())
        self.selected_runs = find_runs(runs, self.edit_prop.text())
        if runs:
            msg = f"{len(self.selected_runs)} runs selected"
            if nmissing := len(runs) - len(self.selected_runs):
                msg += f" - {nmissing} more run numbers don't exist or aren't accessible"
            self.runs_hint.setText(msg)
        else:
            self.runs_hint.setText(RUNS_MSG)
        self.validate()

    def validate_vars(self):
        checks = [itm.checkState() == Qt.Checked for itm in self._var_list_items()]
        self.all_vars_selected = all_sel = all(checks)
        self.no_vars_selected = none_sel = not any(checks)
        self.btn_select_all.setEnabled(not all_sel)
        self.btn_deselect_all.setEnabled(not none_sel)
        self.validate()

    def validate(self):
        valid = bool(self.selected_runs) and not self.no_vars_selected
        self.dlg_buttons.button(QDialogButtonBox.Ok).setEnabled(valid)

    def _var_list_items(self):
        for i in range(self.vars_list.count()):
            yield self.vars_list.item(i)

    def select_all(self):
        for itm in self._var_list_items():
            itm.setCheckState(Qt.Checked)

    def deselect_all(self):
        for itm in self._var_list_items():
            itm.setCheckState(Qt.Unchecked)

    def save_vars_selection(self):
        for item, var in zip(self._var_list_items(), ctx_info.sorted_vars()):
            var.selected = item.checkState() == Qt.Checked

    def accept(self):
        self.save_vars_selection()
        super().accept()

    def reject(self):
        self.save_vars_selection()
        super().reject()

    # Results (along with .selected_runs)
    def proposal_num(self) -> str:
        return self.edit_prop.text()

    def selected_vars(self):
        return [itm.data(Qt.UserRole) for itm in self._var_list_items()
                if itm.checkState() == Qt.Checked]

    def extraction_requests(self):
        prop = int(self.proposal_num())
        # If all variables are selected, don't specify them explicitly, so that
        # newly added functions will also be executed.
        if self.all_vars_selected:
            var_ids = ()
        else:
            var_ids = tuple(self.selected_vars())
        l = [ExtractionRequest(r, prop, RunData.ALL, variables=var_ids)
             for r in self.selected_runs]
        for req in l[1:]:
            req.update_vars = False
        return l



if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    dlg = ProcessingDialog("1234", [3, 4, 5, 6, 10],
        [("test_var", "Test variable"), ("n_trains", "Trains")]
    )
    if dlg.exec() == QtWidgets.QDialog.Accepted:
        print("Proposal:", dlg.proposal_num())
        print("Runs:", dlg.selected_runs)
        print("Variables:", dlg.selected_vars())
