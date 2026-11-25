from unittest.mock import patch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialogButtonBox

from damnit.backend.db import VariableAttributes
from damnit.gui.process import ParamsNewRunsDialog, ProcessingDialog


def test_processing_dialog(mock_db, qtbot):
    db_dir, db = mock_db
    # Mock find_runs() to skip check for run folders
    run_nums = [1, 3, 4]
    with patch("damnit.gui.process.find_runs", return_value=run_nums):
        dlg = ProcessingDialog(1234, run_nums, db)
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
    scale_le = dlg.params_form.widgets_by_name['scale_factor']
    assert scale_le.text() == "1"
    scale_le.setText("3")
    affected_vars = {'scalar1', 'scalar2', 'array', 'meta_array'}
    assert set(dlg.selected_vars()) == affected_vars
    # Force-selected variables can't be deselected
    dlg.deselect_all()
    assert set(dlg.selected_vars()) == affected_vars
    ers = dlg.extraction_requests()
    assert [er.run for er in ers] == run_nums
    assert set(ers[0].variables) == affected_vars
    assert ers[0].params == {'scale_factor': 3}

    # Restoring the parameter's original value returns the deselected variables
    scale_le.setText("1")
    assert set(dlg.selected_vars()) == set()
    assert not ok_btn.isEnabled()


def test_params_new_runs_dialog(mock_db, qtbot):
    db_dir, db = mock_db
    dlg = ParamsNewRunsDialog(db)
    qtbot.addWidget(dlg)

    scale_le = dlg.form.widgets_by_name['scale_factor']
    assert scale_le.text() == "1"

    scale_le.setText("7")
    assert dlg.form.get_modified_values() == {'scale_factor': 7}
    dlg.accept()

    sfvi = db.get_parameters()['scale_factor']
    assert sfvi.is_param
    assert sfvi.attributes[VariableAttributes.PARAM_VALUE_NEW_RUN] == 7
