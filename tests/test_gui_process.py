from unittest.mock import patch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialogButtonBox

from damnit.backend.db import VariableAttributes
from damnit.gui.process import ParamsNewRunsDialog, ProcessingDialog


def test_processing_dialog(mock_db, qtbot):
    db_dir, db = mock_db
    # Mock find_runs() to skip check for run folders
    with patch("damnit.gui.process.find_runs", return_value=[1]):
        dlg = ProcessingDialog(1234, [1], db)
    qtbot.addWidget(dlg)
    ok_btn = dlg.dlg_buttons.button(QDialogButtonBox.StandardButton.Ok)

    all_vars = set()
    for i in range(dlg.vars_list.count()):
        all_vars.add(dlg.vars_list.item(i).data(Qt.ItemDataRole.UserRole))

    # All variables should be selected initially
    assert set(dlg.selected_vars()) == all_vars
    assert ok_btn.isEnabled()

    print(f"{dlg.parameters=}")

    # Deselecting should affect all variables
    dlg.deselect_all()
    assert set(dlg.selected_vars()) == set()
    assert not ok_btn.isEnabled()

    # Modifying a parameter forces affected variables to be selected
    scale_spinbox = dlg.params_form.widgets_by_name['scale_factor']
    assert scale_spinbox.value() == 1
    scale_spinbox.setValue(3)
    assert set(dlg.selected_vars()) == {'scalar1', 'scalar2', 'array', 'meta_array'}
    # Force-selected variables can't be deselected
    dlg.deselect_all()
    assert set(dlg.selected_vars()) == {'scalar1', 'scalar2', 'array', 'meta_array'}

    # Restoring the parameter's original value returns the deselected variables
    scale_spinbox.setValue(1)
    assert set(dlg.selected_vars()) == set()
    assert not ok_btn.isEnabled()


def test_params_new_runs_dialog(mock_db, qtbot):
    db_dir, db = mock_db
    dlg = ParamsNewRunsDialog(db)
    qtbot.addWidget(dlg)

    scale_spinbox = dlg.form.widgets_by_name['scale_factor']
    assert scale_spinbox.value() == 1

    scale_spinbox.setValue(7)
    assert dlg.form.get_modified_values() == {'scale_factor': 7}
    dlg.accept()

    sfvi = db.get_parameters()['scale_factor']
    assert sfvi.is_param
    assert sfvi.attributes[VariableAttributes.PARAM_VALUE_NEW_RUN] == 7
