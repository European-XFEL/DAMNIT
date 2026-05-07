import logging
import re
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialogButtonBox

from extra_data.read_machinery import find_proposal
from superqt import QSearchableListWidget

from damnit.backend.db import DamnitDB
from ..context import RunData
from ..backend.extraction_control import ExtractionRequest
from ..definitions import VariableAttributes

log = logging.getLogger(__name__)

run_range_re = re.compile(r"(\d+)(-\d+)?$")

RUNS_MSG = "Enter run numbers & ranges e.g. '17, 20-32'"

INT_MIN = -2147483648


deselected_vars = set()


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

    def __init__(self, proposal: int, runs: list[int], var_ids_titles,
                 db: DamnitDB, parent=None):
        self.db = db
        super().__init__(parent)

        self.setWindowTitle("Process runs")

        main_vbox = QtWidgets.QVBoxLayout()
        self.setLayout(main_vbox)

        hbox1 = QtWidgets.QHBoxLayout()
        main_vbox.addLayout(hbox1)

        grid1 = QtWidgets.QGridLayout()
        hbox1.addLayout(grid1)
        vbox2 = QtWidgets.QVBoxLayout()
        hbox1.addLayout(vbox2)
        self.parameters = db.get_parameters()
        self.params_form = None
        if self.parameters:
            self.params_box = QtWidgets.QGroupBox("Parameters", self)
            self.params_box.setLayout(QtWidgets.QVBoxLayout())
            hbox1.addWidget(self.params_box)
        else:
            self.params_box = None

        self.edit_prop = QtWidgets.QLineEdit(str(proposal))
        #self.edit_prop.setInputMask('999900;_')  # 4-6 digits
        self.edit_prop.textChanged.connect(self.validate_runs)
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
        self.vars_list.filter_widget.setPlaceholderText("Search variable")
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

        for var_id, title in var_ids_titles:
            itm = QtWidgets.QListWidgetItem(title)
            itm.setData(Qt.UserRole, var_id)
            itm.setCheckState(Qt.Unchecked if var_id in deselected_vars else Qt.Checked)
            self.vars_list.addItem(itm)

        self.vars_list.itemChanged.connect(self.validate_vars)

        self.validate_runs()
        self.validate_vars()

        self.edit_runs.setFocus()

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
        self.set_params_form()
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

    def set_params_form(self):
        if self.params_box is None:
            return

        param_value_sets = self.db.get_parameter_values([
            (self.proposal_num(), r) for r in self.selected_runs
        ])
        # Use ... as a marker for varying
        param_values = {
            n: s.pop() if len(s) == 1 else ... for (n, s) in param_value_sets.items()
        }
        param_layout = self.params_box.layout()
        while (child := param_layout.takeAt(0)) is not None:
            if w := child.widget():
                w.deleteLater()

        new_form = ParametersForm(self.parameters, param_values, parent=self)
        param_layout.addWidget(new_form)
        self.params_form = new_form

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
        # We save the deselected variables, so new variables are selected
        global deselected_vars
        deselected_vars = {
            itm.data(Qt.UserRole) for itm in self._var_list_items()
            if itm.checkState() == Qt.Unchecked
        }

    def accept(self):
        self.save_vars_selection()
        super().accept()

    def reject(self):
        self.save_vars_selection()
        super().reject()

    # Results (along with .selected_runs)
    def proposal_num(self) -> int:
        return int(self.edit_prop.text())

    def selected_vars(self):
        return [itm.data(Qt.UserRole) for itm in self._var_list_items()
                if itm.checkState() == Qt.Checked]

    def extraction_requests(self) -> list[ExtractionRequest]:
        prop = self.proposal_num()
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


class ParametersForm(QtWidgets.QWidget):
    def __init__(self, parameters: dict, values, parent=None):
        super().__init__(parent)
        form_layout = QtWidgets.QFormLayout()
        self.setLayout(form_layout)
        self.widgets_by_name = {}
        self.initial_values = {}
        for name, param in parameters.items():
            value = values.get(name)
            if value is None:
                value = param.attributes[VariableAttributes.PARAM_DEFAULT]
            self.initial_values[name] = value
            w = self.widgets_by_name[name] = self.widget_for(param, value)
            form_layout.addRow(param.title or name, w)

    def widget_for(self, param, value=None):
        match param.variable_type:
            case 'integer':
                w = QtWidgets.QSpinBox()
                w.setMinimum(INT_MIN)
                changed_sig = w.valueChanged
                if value is ...:
                    w.setSpecialValueText("Multiple values")
                    w.setValue(INT_MIN)  # Special value
                else:
                    w.setValue(value or 0)
            case 'number':
                w = QtWidgets.QDoubleSpinBox()
                w.setMinimum(-np.inf)
                changed_sig = w.valueChanged
                if value is ...:
                    w.setSpecialValueText("Multiple values")
                    w.setValue(-np.inf)  # Special value
                else:
                    w.setValue(value or 0.)
            case 'string':
                w = QtWidgets.QLineEdit()
                changed_sig = w.textChanged
                if value is ...:
                    w.setPlaceholderText("Multiple values")
                else:
                    w.setText(value or '')
            case 'boolean':
                w = QtWidgets.QCheckBox()
                changed_sig = w.stateChanged
                if value is ...:
                    w.setTristate(True)
                    w.setCheckState(Qt.CheckState.PartiallyChecked)
                else:
                    w.setCheckState(
                        Qt.CheckState.Checked if value else Qt.CheckState.Unchecked
                    )
            case x:
                raise ValueError(f"Unexpected parameter type {x!r}")

        changed_sig.connect(self.mark_changed)
        if param.description:
            w.setToolTip(param.description)
        return w

    def mark_changed(self):
        layout = self.layout()
        for name, widget in self.widgets_by_name.items():
            label = layout.labelForField(widget)
            changed = self.get_value(name) != self.initial_values[name]
            label.setStyleSheet("QLabel {font-weight: bold}" if changed else "")

    def get_value(self, name):
        w = self.widgets_by_name[name]
        if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            val = w.value()
            if w.specialValueText() and val == w.minimum():
                return ...
            return val
        elif isinstance(w, QtWidgets.QLineEdit):
            val = w.text()
            if (not val) and w.placeholderText():
                return ...
        elif isinstance(w, QtWidgets.QCheckBox):
            state = w.checkState()
            if state == Qt.CheckState.PartiallyChecked:
                return ...
            return state == Qt.CheckState.Checked
        else:
            raise ValueError(f"Unexpected widget type {w!r}")

    def get_values(self):
        return {n: self.get_value(n) for n in self.widgets_by_name}

    def get_modified_values(self):
        return {n: v for (n, v) in self.get_values().items()
                if v != self.initial_values[n]}


class ParamsNewRunsDialog(QtWidgets.QDialog):
    def __init__(self, db: DamnitDB, parent=None):
        self.db = db
        super().__init__(parent)

        self.setWindowTitle("Parameters for new runs")
        self.setMinimumSize(200, 200)
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)

        params_dict = db.get_parameters()
        values = {
            n: v for (n, p) in params_dict.items()
            if (v := p.attributes.get(VariableAttributes.PARAM_VALUE_NEW_RUN)) is not None
        }
        self.form = ParametersForm(params_dict, values)
        vbox.addWidget(self.form)

        info = QtWidgets.QLabel(
            "The values here will be used for new runs."
            "They will not affect reprocessing runs already in DAMNIT."
        )
        info.setWordWrap(True)
        vbox.addWidget(info)

        dlg_buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        dlg_buttons.accepted.connect(self.accept)
        dlg_buttons.rejected.connect(self.reject)
        vbox.addWidget(dlg_buttons)

    def accept(self):
        self.db.update_variables_attrs({
            n: {VariableAttributes.PARAM_VALUE_NEW_RUN: v}
            for (n, v) in self.form.get_modified_values().items()
        })
        super().accept()



if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    dlg = ProcessingDialog("1234", [3, 4, 5, 6, 10],
        [("test_var", "Test variable"), ("n_trains", "Trains")]
    )
    if dlg.exec() == QtWidgets.QDialog.Accepted:
        print("Proposal:", dlg.proposal_num())
        print("Runs:", dlg.selected_runs)
        print("Variables:", dlg.selected_vars())
