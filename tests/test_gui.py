import json
import os
import re
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import Qt, QPoint, QEvent
from PyQt6.QtGui import QColor, QPalette, QPixmap
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtWidgets import (QApplication, QDialog, QFileDialog, QInputDialog,
                             QLineEdit, QMessageBox, QStyledItemDelegate)

import damnit
from damnit.backend.db import DamnitDB, MsgKind, ReducedData
from damnit.backend.extraction_control import SlurmCancelResult
from damnit.backend.extract_data import add_to_db
from damnit.context import Pipeline
from damnit.gui.editor import ContextTestResult
from damnit.gui.main_window import AddUserVariableDialog, MainWindow, prompt_setup_db
from damnit.gui.open_dialog import OpenDBDialog
from damnit.gui.plot import HistogramPlotWindow, ScatterPlotWindow
from damnit.gui.standalone_comments import TimeComment
from damnit.gui.roles import LINE_DATA_ROLE, PROVENANCE_ROLE, UNITS_ROLE
from damnit.gui.table import (
    DamnitTableModel,
    STATIC_COLUMNS,
    TableView,
    best_text_color,
)
from damnit.gui.table_filter import (CategoricalFilter, FilterProxy,
                                     CategoricalFilterWidget, FilterMenu,
                                     NumericFilter, NumericFilterWidget,
                                     ThumbnailFilter, ThumbnailFilterWidget)
from damnit.gui.theme import Theme
from damnit.gui.web_viewer import PlotlyPlot
from damnit.gui.zulip_messenger import ZulipConfig

from .helpers import (
    amore_proto, extract_mock_run, mkcontext, reduced_data_from_dict,
    make_table_with_headers)


# Check if a PID exists by using `kill -0`
def pid_dead(pid):
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True


@contextmanager
def assert_sends_update(win, broker):
    win.update_agent.kafka_prd.flush(timeout=3)  # Flush old messages before capture
    with broker.assert_produces(win.db.kafka_topic) as new_records:
        l = []
        yield l
        win.update_agent.kafka_prd.flush(timeout=3)  # Flush new messages

    l.extend([json.loads(r.value) for r in new_records])


def test_connect_to_kafka(mock_db, qtbot, monkeypatch):
    db_dir, db = mock_db

    with patch(f"kafka.KafkaConsumer") as kafka_cns, \
         patch(f"kafka.KafkaProducer") as kafka_prd:
        win = MainWindow(db_dir, background_activity=False)
        win.close()
        qtbot.addWidget(win)
        kafka_cns.assert_called_once()
        kafka_prd.assert_called_once()

    monkeypatch.setenv("AMORE_BROKER", "none")
    with patch(f"kafka.KafkaConsumer") as kafka_cns, \
         patch(f"kafka.KafkaProducer") as kafka_prd:
        win = MainWindow(db_dir, background_activity=False)
        win.close()
        qtbot.addWidget(win)
        kafka_cns.assert_not_called()
        kafka_prd.assert_not_called()

def test_editor(mock_db, mock_ctx, qtbot):
    db_dir, db = mock_db
    ctx_path = db_dir / "context.py"
    ctx_path.write_text(mock_ctx.code)

    win = MainWindow(db_dir, False)
    # If the context file is not saved, the window will prompt the user about
    # it. This makes the tests hang, so before closing the window we manually
    # mark the context as saved. Useful if a test fails for some reason while
    # the context is changed.
    qtbot.addWidget(win, before_close_func=lambda win: win.mark_context_saved())
    win.show()
    qtbot.waitExposed(win)

    editor = win._editor
    status_bar = win._status_bar

    # Loading a database should also load the context file
    assert editor.text() == mock_ctx.code
    assert "inspect results" in status_bar.currentMessage()

    # Changing to the editor tab should show the full context file path in the
    # status bar.
    win._tab_widget.setCurrentIndex(1)
    assert status_bar.currentMessage() == str(ctx_path.resolve())

    # When the text is changed, instructions to save the context file should be shown
    old_code = "x = 1"
    editor.setText(old_code)
    assert "Ctrl + S" in status_bar.currentMessage()

    # Saving OK code should work
    with qtbot.waitSignal(win.save_context_finished):
        win._save_btn.clicked.emit()
    assert ctx_path.read_text() == old_code

    with qtbot.waitSignal(editor.check_result) as sig:
        editor.launch_test_context(db)
    assert sig.args[0] == ContextTestResult.OK

    assert ctx_path.read_text() == old_code
    assert status_bar.currentMessage() == str(ctx_path.resolve())

    # A valid context that cannot be written should show a warning instead of
    # crashing, and the editor should remain dirty.
    unwritable_code = "x = 10"
    editor.setText(unwritable_code)

    ctx_path.chmod(0o444)
    try:
        with patch("damnit.gui.main_window.QMessageBox.warning") as warning:
            with qtbot.waitSignal(win.save_context_finished):
                win.save_context()
    finally:
        ctx_path.chmod(0o644)
    warning.assert_called_once()
    assert warning.call_args.args[1] == "Could not save context file"
    assert str(ctx_path) in warning.call_args.args[2]
    assert ctx_path.read_text() == old_code
    assert not win._context_is_saved

    # The Validate button should trigger validation. Note that we mock
    # editor.test_context() function instead of MainWindow.test_context()
    # because the win._check_btn.clicked has already been connected to the
    # original function, so mocking it will not make Qt call the mock object.
    with patch.object(editor, "launch_test_context") as test_context:
        win._check_btn.clicked.emit()
        test_context.assert_called_once()

    # Change the context again
    new_code = "x = 2"
    editor.setText(new_code)

    # Cancelling should do nothing
    with patch.object(QMessageBox, "exec", return_value=QMessageBox.StandardButton.Cancel):
        win.close()
        assert win.isVisible()
        assert ctx_path.read_text() == old_code

    # 'Discard' should close the window but not save
    with patch.object(QMessageBox, "exec", return_value=QMessageBox.StandardButton.Discard):
        win.close()
        assert win.isHidden()
        assert ctx_path.read_text() == old_code

    # Show the window again
    win.show()
    qtbot.waitExposed(win)
    assert win.isVisible()

    # Save the valid code
    with qtbot.waitSignal(win.save_context_finished):
        win.save_context()
    assert ctx_path.read_text() == new_code

    # Attempting to save ERROR'ing code should not save anything
    editor.setText("123 = 456")
    with qtbot.waitSignal(win.save_context_finished):
        win.save_context()
    assert ctx_path.read_text() == new_code

    # But saving WARNING code should work
    warning_code = textwrap.dedent("""
    import numpy as np
    x = 1
    """)
    editor.setText(warning_code)
    with qtbot.waitSignal(editor.check_result) as sig:
        editor.launch_test_context(db)
    assert sig.args[0] == ContextTestResult.WARNING
    with qtbot.waitSignal(editor.check_result):
        win.save_context()
    assert ctx_path.read_text() == warning_code

    # Throwing an exception when evaluating the context file in a different
    # environment should be handled gracefully. This can happen if running the
    # ctxrunner itself fails, e.g. because of a missing dependency.
    db.metameta["context_python"] = sys.executable
    with qtbot.waitSignal(editor.check_result) as sig, \
         patch("damnit.gui.editor.get_context_file", side_effect=Exception("foo")):
        editor.launch_test_context(db)
    assert sig.args[0] == ContextTestResult.ERROR

def test_settings(mock_db_with_data, mock_ctx, tmp_path, monkeypatch, qtbot):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)

    # Create the window with a mocked Path so that it saves the settings in the
    # home directory.
    with patch("pathlib.Path.home", return_value=tmp_path):
        win = MainWindow(db_dir, False)
    qtbot.addWidget(win)

    # Helper function to show the currently visible headers
    def visible_headers():
        header = win.table_view.horizontalHeader()
        headers = [header.model().headerData(header.logicalIndex(i), Qt.Orientation.Horizontal) for i in range(header.count())
                   if not header.isSectionHidden(i)]
        return headers

    # Helper function to get the width of a column by its title
    def column_width(title):
        logical_index = win.table.find_column(title, by_title=True)
        return win.table_view.horizontalHeader().sectionSize(logical_index)

    # Helper function to move a column from `from_idx` to `to_idx`. Column moves
    # are triggered by the user drag-and-dropping items in a QListWidget, but
    # unfortunately it seems well-nigh impossible to either simulate a
    # drag-and-drop with QTest or programmatically move a row in a QListWidget
    # (such that its models' rowsMoved() signal is emitted). Hence, this ugly
    # hack.
    def move_columns_manually(from_idx, to_idx):
        headers = visible_headers()
        col_one = headers[from_idx]
        col_two = headers[to_idx]
        columns = win.table_view._columns_widget
        col_one_item = columns.findItems(col_one, Qt.MatchFlag.MatchExactly)[0]
        col_two_item = columns.findItems(col_two, Qt.MatchFlag.MatchExactly)[0]
        old_idx = columns.row(col_two_item)

        from_row = columns.row(col_one_item)
        columns.takeItem(columns.row(col_one_item))
        columns.insertItem(old_idx, col_one_item)
        columns.setCurrentItem(col_one_item)
        win.table_view.item_moved(None, from_row, from_row,
                                  None, columns.row(col_one_item))

        return col_one, col_two

    # Opening a database and adding columns should have triggered a save, so the
    # settings directory should now exist.
    settings_db_path = tmp_path / ".local/state/damnit/settings.db"
    assert settings_db_path.parent.is_dir()

    # Resize a column and ensure its width is persisted
    resize_col = next(col for col in visible_headers() if col not in STATIC_COLUMNS)
    resize_logical = win.table.find_column(resize_col, by_title=True)
    original_width = column_width(resize_col)
    target_width = 240 if original_width != 240 else 260
    win.table_view.setColumnWidth(resize_logical, target_width)
    qtbot.wait(300)
    win.autoconfigure(db_dir)
    assert column_width(resize_col) == target_width

    columns_widget = win.table_view._columns_widget
    static_columns_widget = win.table_view._static_columns_widget

    last_col_item = columns_widget.item(columns_widget.count() - 1)
    last_col = last_col_item.text()

    # Hide a column
    assert last_col in visible_headers()
    last_col_item.setCheckState(Qt.CheckState.Unchecked)
    headers = visible_headers()
    assert last_col not in headers

    # Reload the database to make sure the changes were saved
    win.autoconfigure(db_dir)
    assert headers == visible_headers()

    # Move a column
    headers = visible_headers()
    col_one, col_two = move_columns_manually(-1, -2)
    new_headers = visible_headers()
    assert new_headers.index(col_one) == headers.index(col_two)

    # Reconfigure to reload the settings
    win.autoconfigure(db_dir)

    # Check that they loaded correctly
    assert last_col not in visible_headers()
    assert visible_headers() == new_headers

    # Swapping the same columns again should restore the table to the state
    # before the first move.
    move_columns_manually(-1, -2)
    assert visible_headers() == headers

    # Shift by more than one column
    headers = visible_headers()
    col_one, col_two = move_columns_manually(5, -1)
    assert visible_headers().index(col_one) == headers.index(col_two)

    # Simulate adding a new column while the GUI is running
    msg = {
        "msg_kind": MsgKind.run_values_updated.value,
        "data": {
            "proposal": 1234,
            "run": 1,
            "values": {"new_var": None},
        }
    }
    msg_data = msg['data']
    db.set_variable(
        msg_data["proposal"],
        msg_data["run"],
        "new_var",
        ReducedData(42),
        provenance="test",
    )
    win.handle_update(msg)

    # The new column should be at the end
    headers = visible_headers()
    assert "new_var" == headers[-1]

    # And after reloading the database, the ordering should be the same
    win.autoconfigure(db_dir)
    assert headers == visible_headers()

    # Simulate adding a new column while the GUI is *not* running
    db.set_variable(msg_data["proposal"], msg_data["run"], "newer_var", ReducedData("foo"), provenance = "test")

    # Reload the database
    headers = visible_headers()
    win.autoconfigure(db_dir)

    # The new column should be at the end
    new_headers = visible_headers()
    assert headers == new_headers[:-1]
    assert "newer_var" == new_headers[-1]

    # Test hiding a static column
    static_columns_widget = win.table_view._static_columns_widget
    last_static_col_item = static_columns_widget.item(static_columns_widget.count() - 1)
    last_static_col = last_static_col_item.text()

    assert last_static_col in visible_headers()
    last_static_col_item.setCheckState(Qt.CheckState.Unchecked)
    assert last_static_col not in visible_headers()

    # Reload the database
    headers = visible_headers()
    win.autoconfigure(db_dir)
    assert headers == visible_headers()

def test_handle_update(mock_db, qtbot):
    db_dir, db = mock_db

    win = MainWindow(db_dir, False)
    qtbot.addWidget(win)

    # Helper lambdas
    model = lambda: win.table_view.model()
    get_headers = lambda: [win.table_view.model().headerData(i, Qt.Orientation.Horizontal)
                           for i in range(win.table_view.horizontalHeader().count())]

    # Sending an update should add a row to the table
    msg = {
        "msg_kind": MsgKind.run_values_updated.value,
        "data": {
            "proposal": 1234,
            "run": 1,
            "values": {
                "scalar1": None,
                "string": None,
            },
        }
    }
    msg_data = msg['data']

    assert win.table.rowCount() == 0
    # we need to update the database first as the values are read from there upon table update
    db.set_variable(msg_data["proposal"], msg_data["run"], "string", ReducedData("foo"), provenance="test")
    db.set_variable(msg_data["proposal"], msg_data["run"], "scalar1", ReducedData(42), provenance="test")
    win.handle_update(msg)
    assert win.table.rowCount() == 1

    # Columns should be added for the new variables
    headers = get_headers()
    assert "Scalar1" in headers
    assert "string" in headers

    # Send an update for an existing row
    msg["data"]["values"] = {"scalar1": None}
    db.set_variable(msg_data["proposal"], msg_data["run"], "scalar1", ReducedData(43), provenance="test")
    win.handle_update(msg)
    assert model().data(model().index(0, headers.index("Scalar1"))) == str(43)

    # Add a new column to an existing row
    msg["data"]["values"] = {"unexpected_var": None}
    db.set_variable(msg_data["proposal"], msg_data["run"], "unexpected_var", ReducedData(7), provenance="test")
    win.handle_update(msg)
    assert len(headers) + 1 == len(get_headers())
    assert "unexpected_var" in get_headers()


def test_header_tooltip(mock_db, qtbot):
    db_dir, db = mock_db

    win = MainWindow(db_dir, False)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)

    description = "Primary scalar value for GUI description tests."
    col_ix = win.table.find_column("Scalar1", by_title=True)
    assert win.table_view.model().headerData(col_ix, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole) == description
    col_ix = win.table.find_column("Scalar2", by_title=True)
    assert win.table_view.model().headerData(col_ix, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole) == None


def test_handle_update_plots(mock_db_with_data, monkeypatch, qtbot):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)

    win = MainWindow(db_dir, False)
    qtbot.addWidget(win)

    # Open 1 scatter plot
    win.plot._plot_summaries_clicked()
    assert len(win.plot._plot_windows) == 1

    # Open 1 histogram
    win.plot._toggle_probability_density.setChecked(True)
    win.plot._plot_summaries_clicked()
    assert len(win.plot._plot_windows) == 2
    assert [type(pw) for pw in win.plot._plot_windows] == [
        ScatterPlotWindow, HistogramPlotWindow
    ]

    extract_mock_run(2)
    msg = {
        "msg_kind": MsgKind.run_values_updated.value,
        "data": {
            "proposal": 1234,
            "run": 1,
            "values": {
                "scalar1": None,
                "string": None,
            },
        }
    }
    msg_data = msg['data']
    db.set_variable(msg_data["proposal"], msg_data["run"], "string", ReducedData("foo"), provenance="test")
    db.set_variable(msg_data["proposal"], msg_data["run"], "scalar1", ReducedData(42), provenance="test")
    win.handle_update(msg)

def test_prompt_setup_db(tmp_path, bound_port, request, qtbot):
    db_dir = tmp_path / "usr/Shared/amore"
    pkg = "damnit.gui.main_window"
    template_path = Path(damnit.__file__).parent / 'ctx-templates' / 'SA1_base.py'

    @contextmanager
    def helper_patch():
        # Patch things such that the GUI thinks we're on GPFS trying to open
        # p1234, and the user always wants to create a database.
        with (patch(f"{pkg}.NewContextFileDialog.run_get_result", return_value=(template_path, None)),
              patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes),
              patch(f"{pkg}.initialize_proposal") as initialize_proposal):
            yield initialize_proposal

    # Set up a new DAMNIT directory
    with helper_patch() as initialize_proposal:
        prompt_setup_db(db_dir, prop_no=1234)
        initialize_proposal.assert_called_once_with(db_dir, 1234, template_path, None)

    # Create the directory and database file to fake the database already existing
    db_dir.mkdir(parents=True)
    DamnitDB.from_dir(db_dir, create=True).close()

    # Autoconfigure with database present
    with helper_patch() as initialize_proposal:
        prompt_setup_db(db_dir, prop_no=1234)
        initialize_proposal.assert_not_called()

def test_user_vars(mock_ctx_user, mock_user_vars, mock_db, mock_kafka_broker, qtbot):

    proposal = 1234
    run_number = 1000

    db_dir, db = mock_db
    ctx_path = db_dir / "context.py"
    ctx_path.write_text(mock_ctx_user.code)

    reduced_data = reduced_data_from_dict({
        "user_integer": 12,
        "user_number": 10.2,
        "user_boolean": True,
        "user_string": "foo",
        "dep_integer": 13,
        "dep_number": 10.2,
        "dep_boolean": False,
        "dep_string": "foofoo"
    })

    with db.conn:
        add_to_db(reduced_data, db, proposal, run_number, provenance="test")

    # Adds the variables to the db
    for vv in mock_user_vars.values():
        db.add_user_variable(vv, exist_ok=True)

    win = MainWindow(db_dir, background_activity=False)
    win.show()
    qtbot.addWidget(win)

    # Find menu for creating variable
    create_user_menu = None
    for aa in win.menuBar().actions()[0].menu().actions():
        if aa.text() == "&Create user variable":
            create_user_menu = aa
            break

    # Check that we found the actual menu
    assert create_user_menu is not None

    # After loading a context file the menu should be enabled
    assert create_user_menu.isEnabled()

    add_var_win = AddUserVariableDialog(win)

    # Check if setting a title with non alphanumeric chars
    #   at beginning and end produces a clean variable name
    add_var_win.variable_title.setText("!!My Cool Var!!#")
    assert add_var_win.variable_name.text() == "my_cool_var"

    # Check if setting a title with non alphanumeric chars
    #   in the middle produces a clean variable name
    add_var_win.variable_title.setText("My DAMN#@! Var")
    assert add_var_win.variable_name.text() == "my_damn_var"

    # Check if setting a title with numbers at the beginning
    #   produces a clean variable name
    add_var_win.variable_title.setText("11Var")
    assert add_var_win.variable_name.text() == "var"

    # Check if setting the title to an empty string sets
    #   the variable name to an empty string
    add_var_win.variable_title.setText("")
    assert add_var_win.variable_name.text() == ""

    # Check if disabling variable name autofill don't touch
    #   the variable name value
    add_var_win.name_action.trigger()
    add_var_win.variable_title.setText("This cool var")
    assert add_var_win.variable_name.text() == ""

    # Check if typing an invalid variable name (i.e. digits at the
    #   beginning) is prevented.
    add_var_win.variable_name.insert("111")
    assert add_var_win.variable_name.text() == ""

    # Check if typing an invalid variable name (i.e. variable including
    #   non-alphanumeric characters) is prevented.
    for c in "var$i a@b!le1":
        add_var_win.variable_name.insert(c)
    assert add_var_win.variable_name.text() == "variable1"

    # Check if reactivating the autofill of the variable name updates
    #   the field to match the current title.
    add_var_win.name_action.trigger()
    assert add_var_win.variable_name.text() == "this_cool_var"

    with patch.object(QMessageBox, "exec", return_value=QMessageBox.StandardButton.Cancel):
        # Check if setting the variable title and name to an already existing
        #   one prevents the creation of the variable.
        add_var_win.variable_title.setText("User integer")
        add_var_win.check_if_variable_is_unique("")
        assert add_var_win.result() == QDialog.Rejected

        # Check if setting the variable name to an already existing
        #   one prevents the creation of the variable.
        add_var_win.name_action.trigger()
        add_var_win.variable_title.setText("My cool integer")
        add_var_win.check_if_variable_is_unique("")
        assert add_var_win.result() == QDialog.Rejected

        # Check if setting the variable title to an already existing
        #   one prevents the creation of the variable.
        add_var_win.variable_title.setText("User integer")
        add_var_win.variable_name.setText("my_cool_integer")
        add_var_win.check_if_variable_is_unique("")
        assert add_var_win.result() == QDialog.Rejected

        # Check if setting the variable title and name to a non-conflicting
        #   one allows the creation of the variable.
        add_var_win.name_action.trigger()
        add_var_win.variable_title.setText("My integer")
        add_var_win.variable_type.setCurrentText("integer")
        add_var_win.variable_description.setPlainText("My cool description")
        add_var_win.check_if_variable_is_unique("")
        assert add_var_win.result() == QDialog.DialogCode.Accepted

    # Check that the variable with the already existing name is not created
    assert db.conn.execute("SELECT COUNT(*) FROM variables WHERE title = 'My cool integer'").fetchone()[0] == 0

    # Check that the variable with the already existing title is not created
    assert db.conn.execute("SELECT COUNT(*) FROM variables WHERE name = 'my_cool_integer'").fetchone()[0] == 0

    row = db.conn.execute("SELECT name, type, title, description FROM variables WHERE name = 'my_integer'").fetchone()

    # Check that the variable added via the dialog is actually created
    assert row is not None

    # Check that the fields of the variable added via the dialog was created
    assert tuple(row) == ("my_integer", "integer", "My integer", "My cool description")

    class CaptureEditorDelegate(QStyledItemDelegate):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.widget = None

        def createEditor(self, parent, option, index):
            self.widget = QLineEdit(parent=parent)
            return self.widget

        def destroyEditor(self, editor, index):
            self.widget = None

    table_view = win.table_view
    table_view.setItemDelegate(CaptureEditorDelegate())
    table_model = win.table
    sortproxy = table_view.model()

    def open_editor_and_get_delegate(field_name, row_number = 0):
        col_num = table_model.find_column(field_name)
        table_view.edit(sortproxy.mapFromSource(table_model.index(row_number, col_num)))
        return table_view.itemDelegate()

    def change_to_value_and_close(value):
        delegate = table_view.itemDelegate()
        delegate.widget.insert(value)
        table_view.commitData(delegate.widget)
        table_view.closeEditor(
            delegate.widget,
            QStyledItemDelegate.EndEditHint.NoHint,
        )

    def get_value_from_field(field_name, row_number = 0):
        col_num = table_model.find_column(field_name)
        return table_model.get_value_at_rc(row_number, col_num)

    def get_value_from_db(field_name):
        if not re.fullmatch(r"[a-zA-Z_]\w+", field_name, flags=re.A):
            raise ValueError(f"Error in field_name: the variable name '{field_name}' is not of the form '[a-zA-Z_]\\w+'")
        return db.conn.execute(f"SELECT {field_name} FROM runs WHERE run = ?", (run_number,)).fetchone()[0]

    @contextmanager
    def check_kafka_send(variable):
        with assert_sends_update(win, mock_kafka_broker) as msgs:
            yield

        assert [m['msg_kind'] for m in msgs] == [MsgKind.run_values_updated.value]
        assert set(msgs[0]['data']['values']) == {variable}


    # Check that editing is prevented when trying to modfiy a non-editable column
    assert open_editor_and_get_delegate("dep_number").widget is None

    # Check that editing is allowed when trying to modify a user editable column
    assert open_editor_and_get_delegate("user_number").widget is not None

    with check_kafka_send("user_number"):
        change_to_value_and_close("15.4")

    # Check that the value in the table is of the correct type and value
    assert abs(get_value_from_field("user_number") - 15.4) < 1e-5

    # Check that the value in the db matches what was typed in the table
    assert abs(get_value_from_db("user_number") - 15.4) < 1e-5

    # Check that editing is allowed when trying to modfiy a user editable column
    assert open_editor_and_get_delegate("user_number").widget is not None

    # Try to assign a value of the wrong type
    change_to_value_and_close("fooo")
    # Check that the value is still the same as before
    assert abs(get_value_from_field("user_number") - 15.4) < 1e-5

    # Check that editing is allowed when trying to modfiy a user editable column
    assert open_editor_and_get_delegate("user_number").widget is not None

    # Try to assign an empty value (i.e. deletes the cell)
    change_to_value_and_close("")
    assert pd.isna(get_value_from_field("user_number"))

    # Check that the value in the db matches what was typed in the table
    assert get_value_from_db("user_number") is None

    # Check that editing is allowed when trying to modfiy a user editable column
    assert open_editor_and_get_delegate("user_integer").widget is not None

    change_to_value_and_close("42")

    # Check that the value in the table is of the correct type and value
    assert get_value_from_field("user_integer") == 42

    # Check that the value in the db matches what was typed in the table
    assert get_value_from_db("user_integer") == 42

    # Check that editing is allowed when trying to modfiy a user editable column
    assert open_editor_and_get_delegate("user_integer").widget is not None

    # Try to assign an empty value (i.e. deletes the cell)
    change_to_value_and_close("")
    assert pd.isna(get_value_from_field("user_integer"))

    # Check that the value in the db matches what was typed in the table
    assert get_value_from_db("user_integer") is None

    # Check that editing is allowed when trying to modfiy a user editable column
    assert open_editor_and_get_delegate("user_string").widget is not None

    change_to_value_and_close("Cool string")
    # Check that the value in the table is of the correct type and value
    assert get_value_from_field("user_string") == "Cool string"

    # Check that the value in the db matches what was typed in the table
    assert get_value_from_db("user_string") == "Cool string"

    # Check that editing is allowed when trying to modfiy a user editable column
    assert open_editor_and_get_delegate("user_string").widget is not None

    # Try to assign an empty value (i.e. deletes the cell)
    change_to_value_and_close("")
    assert pd.isna(get_value_from_field("user_string"))

    # Check that the value in the db matches what was typed in the table
    assert get_value_from_db("user_string") is None

    # Check that editing is allowed when trying to modfiy a user editable column
    assert open_editor_and_get_delegate("user_boolean").widget is not None

    change_to_value_and_close("T")
    # Check that the value in the table is of the correct type and value
    assert get_value_from_field("user_boolean")

    # Check that the value in the db matches what was typed in the table
    assert get_value_from_db("user_boolean")

    # Check that editing is allowed when trying to modfiy a user editable column
    assert open_editor_and_get_delegate("user_boolean").widget is not None

    change_to_value_and_close("no")
    # Check that the value in the table is of the correct type and value
    assert not get_value_from_field("user_boolean")

    # Check that editing is allowed when trying to modfiy a user editable column
    assert open_editor_and_get_delegate("user_boolean").widget is not None

    # Try to assign an empty value (i.e. deletes the cell)
    with check_kafka_send("user_boolean"):
        change_to_value_and_close("")
    assert pd.isna(get_value_from_field("user_boolean"))

    # Check that the value in the db matches what was typed in the table
    assert get_value_from_db("user_boolean") is None

def test_table_and_plotting(mock_db_with_data, mock_ctx, mock_run, mock_kafka_broker, monkeypatch, qtbot):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)

    # Create a context file with relevant variables
    const_array_code = """
    @Variable(title="Constant array", summary="mean")
    def constant_array(run):
        return np.ones(2)

    @Variable(title="Image")
    def image(run):
        return np.random.rand(512, 512)

    @Variable(title="Color image")
    def color_image(run):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot([1, 2, 3, 4], [4, 3, 2, 1])
        return fig

    @Variable(title='2D data with summary', summary='mean')
    def mean_2d(run):
        return np.random.rand(512, 512)

    @Variable(title="2D Complex", summary='max')
    def complex_2d(run):
        return np.array([
            [1+1j, 2+2j, 3+3j, 4+4j],
            [1+1j, 2+2j, 3+3j, 4+4j],
            [1+1j, 2+2j, 3+3j, 4+4j]])

    @Variable(title="1D Xarray Complex", summary='max')
    def complex_xr_1d(run):
        import xarray as xr
        return xr.DataArray(np.array([1+1j, 2+2j, 3+3j, 4+4j]))

    @Variable()
    def error(run):
        1/0

    @Variable(title="2D array for line plot")
    def line_plot_data(run):
        return np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6]])

    @Variable(title="2D xarray for line plot")
    def xarray_line_plot_data(run, data: "var#line_plot_data"):
        import xarray as xr
        return xr.DataArray(data, dims=['line', 'x'], coords={'line': ['A', 'B', 'C'], 'x': [0, 1, 2, 3]})
    """
    ctx_code = mock_ctx.code + "\n\n" + textwrap.dedent(const_array_code)
    (db_dir / "context.py").write_text(ctx_code)
    Pipeline.from_str(ctx_code)
    extract_mock_run(1)

    # Create window
    win = MainWindow(db_dir, background_activity=False)
    qtbot.addWidget(win)

    # Helper function to get a QModelIndex from a variable title
    def get_index(title, row=0):
        col = win.table.find_column(title, by_title=True)
        index = win.table.index(row, col)
        assert index.isValid()
        return index

    # We should be able to plot summaries
    win.plot._combo_box_x_axis.setCurrentText("Array")
    win.plot._combo_box_y_axis.setCurrentText("Constant array")

    with patch.object(QMessageBox, "warning") as warning:
        win.plot._plot_summaries_clicked()
        warning.assert_not_called()

    # And plot an array
    array_index = get_index("Array")
    with patch.object(QMessageBox, "warning") as warning:
        win.inspect_data(array_index)
        warning.assert_not_called()

    # And correlate two array variables
    array_sorted_idx = win.table_view.model().mapFromSource(array_index)
    win.table_view.setCurrentIndex(array_sorted_idx)
    with patch.object(QMessageBox, "warning") as warning:
        win.plot._plot_run_data_clicked()
        warning.assert_not_called()

    # Check that the text for the array that changes is bold
    assert win.table.data(array_index, role=Qt.ItemDataRole.FontRole).bold()

    # But not for the constant array
    const_array_index = get_index("Constant array")
    assert win.table.data(const_array_index, role=Qt.ItemDataRole.FontRole) is None

    # Edit a comment
    comment_index = get_index("Comment")
    with assert_sends_update(win, mock_kafka_broker) as msgs:
        win.table.setData(comment_index, "Foo", Qt.ItemDataRole.EditRole)

    assert [m['msg_kind'] for m in msgs] == [MsgKind.run_values_updated.value]
    assert set(msgs[0]['data']['values']) == {"comment"}

    # Check that 2D arrays are treated as images
    image_index = get_index("Image")
    assert isinstance(win.table.data(image_index, role=Qt.ItemDataRole.DecorationRole), QPixmap)
    with patch.object(QMessageBox, "warning") as warning:
        win.inspect_data(image_index)
        warning.assert_not_called()

    # And that 3D image arrays are also treated as images
    color_image_index = get_index("Color image")
    assert isinstance(win.table.data(color_image_index, role=Qt.ItemDataRole.DecorationRole), QPixmap)
    with patch.object(QMessageBox, "warning") as warning:
        win.inspect_data(color_image_index)
        warning.assert_not_called()

    # Check that 2D arrays with summary are inspectable
    mean_2d_index = get_index('2D data with summary')
    assert win.table.data(mean_2d_index, role=Qt.ItemDataRole.FontRole).bold()
    assert isinstance(win.table.data(mean_2d_index, role=Qt.ItemDataRole.DisplayRole), str)
    with patch.object(QMessageBox, "warning") as warning:
        win.inspect_data(mean_2d_index)
        warning.assert_not_called()

    # xarray of complex data is not inspectable
    complex_2d = get_index('2D Complex')
    with patch.object(QMessageBox, "warning") as warning:
        win.inspect_data(complex_2d)
        warning.assert_called_once()

    complex_xr_1d = get_index('1D Xarray Complex')
    with patch.object(QMessageBox, "warning") as warning:
        win.inspect_data(complex_xr_1d)
        warning.assert_called_once()

    # Check that objects with a preview are inspectable
    plotly_preview = get_index('Plotly preview')
    n_plots_before = len(win._canvas_inspect)
    win.inspect_data(plotly_preview)
    assert len(win._canvas_inspect) == n_plots_before + 1
    assert isinstance(win._canvas_inspect[-1], PlotlyPlot)

    assert isinstance(  # Errors evaluating variables get a coloured decoration
        win.table.data(get_index('error'), role=Qt.ItemDataRole.DecorationRole), QColor
    )

    # Test the line plot functionality for 2D data
    line_data_index = get_index('2D array for line plot')
    with patch.object(QMessageBox, "warning") as warning:
        win.inspect_data(line_data_index)
        plot_window = win._canvas_inspect[-1]
        warning.assert_not_called()

        # Check image mode is the default
        assert isinstance(plot_window._image_artist, plt.Artist)

        # Toggle to line plot mode
        plot_window._plot_as_lines_checkbox.setCheckState(Qt.CheckState.Checked)
        plot_window.figure.canvas.flush_events()

        assert not plot_window._dynamic_aspect_checkbox.isEnabled()
        assert plot_window._axis.get_aspect() == 'auto'

        # Toggle back to image mode
        plot_window._plot_as_lines_checkbox.setCheckState(Qt.CheckState.Unchecked)
        plot_window.figure.canvas.flush_events()

        assert plot_window._dynamic_aspect_checkbox.isEnabled()

    # Test the line plot with xarray data
    xarray_line_data_index = get_index('2D xarray for line plot')
    with patch.object(QMessageBox, "warning") as warning:
        win.inspect_data(xarray_line_data_index)
        xr_plot_window = win._canvas_inspect[-1]
        warning.assert_not_called()

        xr_plot_window._plot_as_lines_checkbox.setCheckState(Qt.CheckState.Checked)
        xr_plot_window.figure.canvas.flush_events()

        assert len(xr_plot_window._axis.get_legend().get_texts()) == 3  # 3 rows in our test data


def test_open_dialog(mock_db, qtbot):
    db_dir, db = mock_db
    dlg = OpenDBDialog()
    qtbot.addWidget(dlg)
    dlg.proposal_finder_thread.start()

    # Test supplying a proposal number:
    with patch("damnit.gui.open_dialog.find_proposal", return_value=str(db_dir)):
        with qtbot.waitSignal(dlg.proposal_finder.find_result):
            dlg.ui.proposal_edit.setText('1234')
    dlg.accept()
    dlg.proposal_finder_thread.wait(2000)

    assert dlg.get_chosen_dir() == db_dir / 'usr/Shared/amore'
    assert dlg.get_proposal_num() == 1234

    # Test selecting a folder:
    dlg = OpenDBDialog()
    qtbot.addWidget(dlg)
    dlg.proposal_finder_thread.start()
    dlg.ui.folder_rb.setChecked(True)
    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(db_dir)):
        dlg.ui.browse_button.click()
    dlg.accept()
    dlg.proposal_finder_thread.wait(2000)

    assert dlg.get_chosen_dir() == db_dir
    assert dlg.get_proposal_num() is None

def test_zulip(mock_db_with_data, monkeypatch, qtbot):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)
    win = MainWindow(db_dir, False)
    qtbot.addWidget(win)
    pkg = 'damnit.gui.zulip_messenger.requests'

    mock_zulip_cfg = """
    [ZULIP]
    key = 1234567890
    url = url
    """
    res_get = SimpleNamespace(status_code = 200,
                              text = '{"stream" : "stream"}')
    res_post = SimpleNamespace(status_code = 200,
                              response = '{"response" : "success"}')

    (db_dir / "zulip.cfg").write_text(mock_zulip_cfg)

    with patch(f'{pkg}.get', return_value =res_get) as mock_get,\
    patch(f'{pkg}.post', return_value = res_post) as mock_post:
        win.check_zulip_messenger()
        mock_get.assert_called_once()
        assert win.zulip_messenger.ok

        # test parsing of configuration file
        assert win.zulip_messenger.key == "1234567890"
        assert win.zulip_messenger.url == "url"

        # test get request for correct set up of the stream name
        mock_get.assert_called_once()
        assert win.zulip_messenger.stream == "stream"

        df = win.table.dataframe_for_export(['Proposal', 'Run', 'Comment'])
        test_dialog = ZulipConfig(win, win.zulip_messenger,
                                  kind='table', table=df)
        test_dialog.handle_form()

        # Check if table was parsed into a list
        assert isinstance(test_dialog.msg, list)

        # Check if post was called
        mock_post.assert_called_once()

    # Simple smoke test to make sure a Zulip message is sent
    with patch.object(win, "zulip_messenger") as messenger, \
         patch.object(win, "check_zulip_messenger", return_value=True):
        win.export_selection_to_zulip()

        messenger.send_table.assert_called_once()


def test_copy_selection_as_markdown(mock_db_with_data, qtbot, monkeypatch):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)
    win = MainWindow(db_dir, False)
    qtbot.addWidget(win)

    win.table_view.selectRow(0)
    win.copy_selection_as_markdown()

    markdown = QApplication.clipboard().text()
    assert "|Run|" in markdown
    assert "|Proposal|" not in markdown
    assert "|Status|" not in markdown
    assert "<image>" not in markdown


@pytest.mark.parametrize("extension", [".xlsx", ".csv", ".md"])
def test_exporting(mock_db_with_data, mock_kafka_broker, qtbot, monkeypatch, extension):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)

    code = """
    import numpy as np
    from damnit_ctx import Variable

    @Variable(title="Number")
    def number(run):
        return 42

    @Variable(title="Image")
    def image(run):
        return np.random.rand(100, 100)
    """
    ctx = mkcontext(code)
    (db_dir / "context.py").write_text(ctx.code)
    extract_mock_run(1)

    win = MainWindow(db_dir)
    qtbot.addWidget(win)

    export_path = db_dir / f"export{extension}"
    filter_str = f"Ext (*{extension})"
    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(export_path.stem), filter_str)):
        win.export_table()

    assert export_path.is_file()

    # Check that images are formatted nicely
    if extension == ".md":
        markdown = export_path.read_text()
        assert "|Proposal|Run|" in markdown
        assert "<image>" in markdown
    else:
        df = pd.read_excel(export_path) if extension == ".xlsx" else pd.read_csv(export_path)
        assert df["Image"][0] == "<image>"

def test_delete_variable(mock_db_with_data, qtbot, monkeypatch, mock_kafka_broker):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)

    # We'll delete the 'array' variable
    assert "array" in db.variable_names()
    win = MainWindow(db_dir)
    qtbot.addWidget(win)
    tbl = win.table
    column_ids_before = [tbl.column_id(i) for i in range(tbl.columnCount())]
    column_titles_before = tbl.column_titles.copy()
    assert "array" in column_ids_before
    col_visibility_before = win.table_view.get_column_states()
    assert "Array" in col_visibility_before  # Keyed by title, not column ID

    # If the user clicks 'No' then we should do nothing
    with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.No) as warning:
        win.table_view.confirm_delete_variable("array")
        warning.assert_called_once()
    assert "array" in db.variable_names()
    assert [tbl.column_id(i) for i in range(tbl.columnCount())] == column_ids_before
    assert tbl.column_titles == column_titles_before
    assert win.table_view.get_column_states() == col_visibility_before

    # Otherwise it should be deleted from the database and HDF5 files
    with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Yes) as warning:
        win.table_view.confirm_delete_variable("array")
        warning.assert_called_once()

    assert "array" not in db.variable_names()
    assert tbl.columnCount() == len(column_ids_before) - 1
    assert "array" not in [tbl.column_id(i) for i in range(tbl.columnCount())]
    assert len(tbl.column_titles) == tbl.columnCount()
    assert "Array" not in win.table_view.get_column_states()

    proposal = db.metameta['proposal']
    with h5py.File(db_dir / f"extracted_data/p{proposal}_r1.h5") as f:
        assert "array" not in f.keys()
        assert "array" not in f[".reduced"].keys()

def test_delete_variable_with_tag_filter(
    mock_db, tmp_path, qtbot, monkeypatch, mock_kafka_broker
):
    def _visible_count(tv):
        count = 0
        for col in range(tv.get_static_columns_count(), table_view.model().columnCount()):
            if not tv.isColumnHidden(col):
                count += 1
        return count

    def _delete_col(tv, col_name):
        with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Yes):
            tv.confirm_delete_variable(col_name)

    db_dir, _db = mock_db
    monkeypatch.chdir(db_dir)

    with patch("pathlib.Path.home", return_value=tmp_path):
        win = MainWindow(db_dir, background_activity=False)
    qtbot.addWidget(win)

    table_view = win.table_view

    initial_states = table_view.get_column_states()
    assert "Array" in initial_states

    # Apply a tag filter that hides non-scalar variables in the view only.
    table_view.apply_tag_filter({"scalar"})
    table_view.apply_tag_filter.flush()
    assert table_view._current_tag_filter == {"scalar"}

    # Delete a non-scalar variable while the filter is active.
    _delete_col(table_view, "array")

    expected_states = dict(initial_states)
    del expected_states["Array"]

    # Deletion should not convert temporarily filter-hidden columns into
    # manually hidden settings.
    assert table_view.get_column_states() == expected_states
    assert table_view._current_tag_filter == {"scalar"}

    # The active filter should still be applied after deleting a column.
    assert _visible_count(table_view) == 2  # Scalar1 and Scalar2

    # delete a scalar variable
    _delete_col(table_view, "scalar2")
    del expected_states["Scalar2"]
    assert table_view.get_column_states() == expected_states
    assert _visible_count(table_view) == 1  # Scalar1

    # Verify persistence across GUI reload.
    win.autoconfigure(db_dir)
    assert win.table_view.get_column_states() == expected_states

    with patch("pathlib.Path.home", return_value=tmp_path):
        win2 = MainWindow(db_dir)
    qtbot.addWidget(win2)
    assert win2.table_view.get_column_states() == expected_states


def test_precreate_runs(mock_db_with_data, qtbot, monkeypatch, mock_kafka_broker):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)

    win = MainWindow(db_dir, background_activity=False)
    qtbot.addWidget(win)
    get_n_runs = lambda: db.conn.execute("SELECT COUNT(run) FROM runs").fetchone()[0]
    n_runs = get_n_runs()

    # The user cancelling should do nothing
    with patch.object(QInputDialog, "getInt", return_value=(1, False)) as dialog:
        win.precreate_runs_dialog()
        dialog.assert_called_once()
        assert get_n_runs() == n_runs

    # But accepting should add a run
    with patch.object(QInputDialog, "getInt", return_value=(1, True)) as dialog:
        win.precreate_runs_dialog()
        dialog.assert_called_once()
        assert get_n_runs() == n_runs + 1


def test_tag_filtering(mock_db_with_data, mock_ctx, qtbot):
    """Test the tag filtering functionality in the table view."""
    db_dir, db = mock_db_with_data

    # Create main window
    win = MainWindow(db_dir, False)
    qtbot.addWidget(win)

    table_view = win.table_view

    # Helper function to count visible variable columns
    def count_visible_vars():
        count = 0
        for col in range(table_view.get_static_columns_count(), table_view.model().columnCount()):
            if not table_view.isColumnHidden(col):
                count += 1
        return count

    # Helper function to count visible static columns
    def count_visible_static():
        count = 0
        for col in range(table_view.get_static_columns_count()):
            if not table_view.isColumnHidden(col):
                count += 1
        return count

    # Hepler function to apply tag filter
    def apply_tag_filter(tags):
        table_view.apply_tag_filter(tags)
        table_view.apply_tag_filter.flush()

    # Test initial state - all columns should be visible
    initial_var_count = count_visible_vars()
    initial_static_count = count_visible_static()
    assert initial_var_count == len(db.variable_names())
    assert initial_static_count == table_view.get_static_columns_count()

    # Test filtering with single tag
    apply_tag_filter({"scalar"})
    assert count_visible_vars() == 2  # scalar1 and scalar2
    assert table_view._tag_filter_button.text() == "Variables: scalar"

    # Test filtering with multiple tags
    apply_tag_filter({"scalar", "text"})
    assert count_visible_vars() == 3  # scalar1, scalar2, and empty_string
    assert table_view._tag_filter_button.text() == "Variables: 2 tags"

    # Test filtering with non-existent tag
    apply_tag_filter({"nonexistent_tag"})
    assert count_visible_vars() == 0
    assert table_view._tag_filter_button.text() == "Variables: nonexistent_tag"

    # Test clearing filters
    apply_tag_filter(set())
    assert count_visible_vars() == initial_var_count
    assert table_view._tag_filter_button.text() == "Variables by Tag"

    # Test internal state of tag filter set
    table_view._toggle_tag_filter("scalar")
    table_view.apply_tag_filter.flush()
    assert "scalar" in table_view._current_tag_filter
    assert count_visible_vars() == 2

    table_view._toggle_tag_filter("scalar")  # toggle off
    table_view.apply_tag_filter.flush()
    assert "scalar" not in table_view._current_tag_filter
    assert count_visible_vars() == initial_var_count

    # Test interaction between column visibility and tag filtering

    # Hide a static column
    static_item = table_view._static_columns_widget.item(0)  # First static column
    static_item.setCheckState(Qt.CheckState.Unchecked)
    assert count_visible_static() == initial_static_count - 1

    # Hide a variable column
    table_view._columns_widget.findItems('Scalar1', Qt.MatchFlag.MatchExactly)[0].setCheckState(Qt.CheckState.Unchecked)
    assert count_visible_vars() == initial_var_count - 1

    # Apply tag filter - should respect column visibility preferences
    apply_tag_filter({"scalar"})
    assert count_visible_static() == initial_static_count - 1  # Static column still hidden
    visible_vars = count_visible_vars()

    assert visible_vars < 2  # Should be less than 2 because we hid one variable

    # Show all columns again - should still respect visibility preferences
    apply_tag_filter(set())
    assert count_visible_static() == initial_static_count - 1
    assert count_visible_vars() == initial_var_count - 1


def test_filter_proxy(mock_db_with_data_2, qtbot):
    db_dir, db = mock_db_with_data_2

    # Create main window
    win = MainWindow(db_dir, False)
    qtbot.addWidget(win)

    table_view = win.table_view

    proxy_model = table_view.model()
    source_model = win.table
    initial_rows = proxy_model.rowCount()

    # Test numeric filtering
    scalar1_col = source_model.find_column("Scalar1", by_title=True)

    # Test with range and selected values
    num_filter = NumericFilter(scalar1_col, min_val=40, max_val=45, selected_values={42})
    proxy_model.set_filter(scalar1_col, num_filter)
    assert proxy_model.rowCount() == 5

    # Test with range but no matching selected values
    num_filter = NumericFilter(scalar1_col, min_val=40, max_val=45, include_nan=False)
    proxy_model.clear_filters()
    proxy_model.set_filter(scalar1_col, num_filter)
    assert proxy_model.rowCount() == 0

    # Test categorical filtering
    status_col = source_model.find_column("Results", by_title=True)

    # Filter to show only rows with the first status value
    cat_filter = CategoricalFilter(status_col, {"Failed"})
    proxy_model.clear_filters()
    proxy_model.set_filter(status_col, cat_filter)
    assert proxy_model.rowCount() == 4

    # Test multiple filters
    num_filter = NumericFilter(scalar1_col, min_val=40, max_val=45, selected_values={42}, include_nan=False)
    proxy_model.set_filter(scalar1_col, num_filter)
    assert proxy_model.rowCount() == 2

    # Clear filters
    proxy_model.clear_filters()
    assert proxy_model.rowCount() == initial_rows

    # Test thumbnail filter
    thumb_col = source_model.find_column("Image", by_title=True)
    thumb_filter = ThumbnailFilter(thumb_col, show_with_thumbnail=True, show_without_thumbnail=False)
    proxy_model.set_filter(thumb_col, thumb_filter)
    assert proxy_model.rowCount() == 1

    # Test clear all filters
    proxy_model.clear_filters()
    assert proxy_model.rowCount() == initial_rows


def test_filters():
    # Test numeric filter with selected values
    num_filter = NumericFilter(column=0, min_val=10, max_val=20, selected_values={15})
    assert num_filter.accepts(15)
    assert not num_filter.accepts(18)
    assert not num_filter.accepts(5)
    assert not num_filter.accepts(25)
    assert num_filter.accepts(None)
    assert num_filter.accepts(np.nan)
    assert not num_filter.accepts("not a number")

    # Test numeric filter with nan handling
    nan_filter = NumericFilter(column=0, min_val=10, max_val=20, selected_values={15}, include_nan=False)
    assert nan_filter.accepts(15)
    assert not nan_filter.accepts(None)
    assert not nan_filter.accepts(np.nan)
    assert not nan_filter.accepts(5)
    assert not nan_filter.accepts(25)
    assert not nan_filter.accepts(18)

    # Test categorical filter with selected values
    cat_filter = CategoricalFilter(column=1, selected_values={"A", "B"})
    assert cat_filter.accepts("A")
    assert cat_filter.accepts("B")
    assert not cat_filter.accepts("C")
    assert cat_filter.accepts(None)
    assert cat_filter.accepts(np.nan)

    # Test categorical filter with nan handling
    nan_cat_filter = CategoricalFilter(column=1, selected_values={"A"}, include_nan=False)
    assert nan_cat_filter.accepts("A")
    assert not nan_cat_filter.accepts(None)
    assert not nan_cat_filter.accepts(np.nan)
    assert not nan_cat_filter.accepts("B")

    # Test empty filters
    empty_filter = CategoricalFilter(column=1)
    assert not empty_filter.accepts("nothing")
    assert empty_filter.accepts(None)
    assert not empty_filter.accepts(42)

    # Test thumbnail filter with both options enabled
    thumb_filter = ThumbnailFilter(column=2)
    assert thumb_filter.accepts(QPixmap)
    assert thumb_filter.accepts(None)
    assert thumb_filter.accepts("not a thumbnail")

    # Test thumbnail filter showing only thumbnails
    thumb_only_filter = ThumbnailFilter(column=2, show_with_thumbnail=True, show_without_thumbnail=False)
    assert thumb_only_filter.accepts(QPixmap)
    assert not thumb_only_filter.accepts(None)
    assert not thumb_only_filter.accepts("not a thumbnail")

    # Test thumbnail filter showing only non-thumbnails
    no_thumb_filter = ThumbnailFilter(column=2, show_with_thumbnail=False, show_without_thumbnail=True)
    assert not no_thumb_filter.accepts(QPixmap)
    assert no_thumb_filter.accepts(None)
    assert no_thumb_filter.accepts("not a thumbnail")

    # Test thumbnail filter with both options disabled
    hidden_thumb_filter = ThumbnailFilter(column=2, show_with_thumbnail=False, show_without_thumbnail=False)
    assert not hidden_thumb_filter.accepts(QPixmap)
    assert not hidden_thumb_filter.accepts(None)
    assert not hidden_thumb_filter.accepts("not a thumbnail")


def test_standalone_comments(mock_db, qtbot):
    db_dir, db = mock_db

    win = MainWindow(db_dir, False)
    win.show()
    qtbot.waitExposed(win)
    qtbot.addWidget(win)

    # Create and show the TimeComment dialog
    dialog = TimeComment(win)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)

    model = dialog.model

    # Test adding a comment
    test_timestamp = 1640995200  # 2022-01-01 00:00:00
    test_comment = "Test comment 1"
    model.addComment(test_timestamp, test_comment)

    # Verify comment was added
    assert model.rowCount() > 0
    index = model.index(0, 2)  # Comment column
    assert model.data(index, Qt.ItemDataRole.DisplayRole) == test_comment

    # Add another comment
    test_timestamp2 = 1641081600  # 2022-01-02 00:00:00
    test_comment2 = "Test comment 2"
    model.addComment(test_timestamp2, test_comment2)

    # Test sorting
    # Sort by timestamp ascending
    model.sort(1, Qt.SortOrder.AscendingOrder)
    index = model.index(0, 2)
    assert model.data(index, Qt.ItemDataRole.DisplayRole) == test_comment

    # Sort by timestamp descending
    model.sort(1, Qt.SortOrder.DescendingOrder)
    index = model.index(0, 2)
    assert model.data(index, Qt.ItemDataRole.DisplayRole) == test_comment2

    # Test comment persistence
    model.load_comments()
    assert model.rowCount() == 2


def test_filter_menu(mock_db_with_data, qtbot):
    """Test FilterMenu initialization and functionality."""
    win = MainWindow(mock_db_with_data[0], False)
    win.show()
    qtbot.addWidget(win)
    qtbot.waitExposed(win)
    model = win.table_view.model()

    # Test numeric column
    scalar1_col = win.table.find_column("Scalar1", by_title=True)
    numeric_menu = FilterMenu(scalar1_col, model)
    qtbot.addWidget(numeric_menu)
    assert isinstance(numeric_menu.filter_widget, NumericFilterWidget)

    # Test categorical column
    results_col = win.table.find_column("Results", by_title=True)
    categorical_menu = FilterMenu(results_col, model)
    qtbot.addWidget(categorical_menu)
    assert isinstance(categorical_menu.filter_widget, CategoricalFilterWidget)

    # Test thumbnail column
    thumb_col = win.table.find_column("Image", by_title=True)
    thumbnail_menu = FilterMenu(thumb_col, model)
    qtbot.addWidget(thumbnail_menu)
    assert isinstance(thumbnail_menu.filter_widget, ThumbnailFilterWidget)

    # Test filter application
    with qtbot.waitSignal(numeric_menu.filter_widget.filterCleared):
        numeric_menu.filter_widget._on_selection_changed()

    numeric_menu.filter_widget.include_nan.setChecked(False)
    with qtbot.waitSignal(numeric_menu.filter_widget.filterChanged):
        numeric_menu.filter_widget._on_selection_changed()

    # Test menu with existing filter
    existing_filter = CategoricalFilter(results_col, selected_values={"OK"})
    model.set_filter(results_col, existing_filter)
    menu_with_filter = FilterMenu(results_col, model)
    qtbot.addWidget(menu_with_filter)
    assert menu_with_filter.model.filters[results_col] == existing_filter

    # Test menu with existing thumbnail filter
    existing_thumb_filter = ThumbnailFilter(thumb_col, show_with_thumbnail=True, show_without_thumbnail=False)
    model.set_filter(thumb_col, existing_thumb_filter)
    menu_with_thumb_filter = FilterMenu(thumb_col, model)
    qtbot.addWidget(menu_with_thumb_filter)
    assert menu_with_thumb_filter.model.filters[thumb_col] == existing_thumb_filter

    # Test thumbnail filter
    thumb_filter = ThumbnailFilter(thumb_col, show_with_thumbnail=True, show_without_thumbnail=False)
    model.set_filter(thumb_col, thumb_filter)
    assert len(model.filters) == 2
    assert thumb_col in model.filters
    assert model.filters[thumb_col] == thumb_filter


def test_processing_status(mock_db_with_data, qtbot, mock_kafka_broker):
    db_dir, db = mock_db_with_data
    win = MainWindow(db_dir, background_activity=False)
    qtbot.addWidget(win)
    tbl = win.table

    def shows_as_processing(run):
        row = tbl.find_row(1234, run)
        runnr_s = tbl.verticalHeaderItem(row).data(Qt.ItemDataRole.DisplayRole)
        return "⚙️" in runnr_s

    d = {'proposal': 1234, 'data': 'all', 'hostname': '', 'username': '',
         'slurm_cluster': '', 'slurm_job_id': '', 'status': 'RUNNING'}

    # Test with an existing run
    prid1, prid2 = str(uuid4()), str(uuid4())
    tbl.handle_processing_state_set(d | {'run': 1, 'processing_id': prid1})
    assert shows_as_processing(1)
    tbl.handle_processing_state_set(d | {'run': 1, 'processing_id': prid2})
    tbl.handle_processing_finished({'processing_id': prid1})
    assert shows_as_processing(1)
    tbl.handle_processing_finished({'processing_id': prid2})
    assert not shows_as_processing(1)

    # Processing starting for a new run should add a row
    assert tbl.rowCount() == 1
    tbl.handle_processing_state_set(d | {'run': 2, 'processing_id': str(uuid4())})
    assert tbl.rowCount() == 2
    assert shows_as_processing(2)


def test_cancel_processing_jobs(mock_db_with_data, qtbot, mock_kafka_broker):
    db_dir, db = mock_db_with_data
    win = MainWindow(db_dir, background_activity=False)
    qtbot.addWidget(win)
    tbl = win.table
    view = win.table_view

    row = tbl.find_row(1234, 1)
    proxy_row = view.model().mapFromSource(tbl.index(row, 0)).row()
    view.selectRow(proxy_row)
    view.selection_changed()
    assert not view.cancel_jobs_action.isEnabled()

    prid = str(uuid4())
    tbl.handle_processing_state_set({
        'proposal': 1234,
        'run': 1,
        'data': 'all',
        'hostname': '',
        'username': '',
        'slurm_cluster': 'maxwell',
        'slurm_job_id': '321',
        'status': 'PENDING',
        'processing_id': prid,
    })
    assert view.cancel_jobs_action.isEnabled()

    with patch("damnit.backend.extraction_control.cancel_slurm_job") as cancel:
        cancel.return_value = SlurmCancelResult(
            cluster="maxwell", job_id="321", cancelled=True, error="",
            already_gone=False, state="PENDING"
        )
        win.cancel_processing_jobs()

    cancel.assert_called_once_with("maxwell", "321")
    assert prid not in tbl.processing_jobs.jobs
    assert not view.cancel_jobs_action.isEnabled()
    assert "job 321" in (
        db_dir / "process_logs" / "r1-p1234.out"
    ).read_text()


def test_theme(mock_db, qtbot, tmp_path):
    """Test theme loading, saving, and application."""
    db_dir, db = mock_db
    settings_path = tmp_path / ".local/state/damnit/settings.db"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    with patch("pathlib.Path.home", return_value=tmp_path):
        # Test default theme
        win = MainWindow(db_dir, False)
        qtbot.addWidget(win)
        assert win.current_theme == Theme.LIGHT

        # Test theme saving and loading
        win._toggle_theme(True)
        assert win.current_theme == Theme.DARK
        win.close()

        # Create new window to test theme persistence
        win2 = MainWindow(db_dir, False)
        qtbot.addWidget(win2)
        assert win2.current_theme == Theme.DARK  # Should load saved dark theme
        assert win2.dark_mode_action.isChecked()  # Action should be checked

        # Test theme application to components
        dark_palette = win2.palette()
        dark_window_color = dark_palette.color(QPalette.ColorRole.Window).name()
        assert dark_window_color == "#353535"  # Dark theme color
        assert dark_palette.color(QPalette.ColorRole.WindowText).name() == "#ffffff"  # White text

        # Test theme application to editor
        assert win2._editor._lexer.defaultPaper(0).name() == "#232323"  # Dark theme editor background

        # Test theme toggle back to light
        win2._toggle_theme(False)
        QApplication.processEvents()
        assert win2.current_theme == Theme.LIGHT
        assert win2.palette().color(QPalette.ColorRole.Window).name() != dark_window_color


def test_filter_header(mock_db_with_data, qtbot, mock_kafka_broker):
    """Test that the filter header shows icons when filters are active."""
    window = MainWindow(mock_db_with_data[0], background_activity=False)
    qtbot.addWidget(window)
    window.show()

    table = window.table_view
    header = table.horizontalHeader()

    # Initially no filters should be active
    for col in range(table.model().columnCount()):
        assert col not in header.filtered_columns

    # Apply a filter to the first column
    col_index = 0
    # filter_menu = FilterMenu(col_index, table.model(), table)
    filter = CategoricalFilter(col_index, {"test_value"})
    table.model().set_filter(col_index, filter)

    # Check that the filter icon appears in the header
    assert col_index in header.filtered_columns

    # Clear the filter
    table.model().set_filter(col_index, None)

    # Check that the filter icon is removed
    assert col_index not in header.filtered_columns


### Test LogViewWindow
LOG_WAIT_MS = 300  # a bit longer than polling interval (200ms)


def test_logview_append_content(log_view_window, qtbot):
    """Test appending new content to the log file."""
    window, log_file_path = log_view_window
    initial_content = log_file_path.read_text()
    initial_stat = log_file_path.stat()
    appended_text = "Line 3"

    assert window.text_edit.toPlainText() == log_file_path.read_text()
    assert initial_stat.st_size > 0
    # Check internal state matches
    assert window._last_size == initial_stat.st_size
    assert window._last_mtime == initial_stat.st_mtime

    with log_file_path.open("a") as f:
        f.write(appended_text)

    # Wait for the timer to fire and process the change
    qtbot.wait(LOG_WAIT_MS)

    expected_content = '\n'.join([initial_content, appended_text])
    final_stat = log_file_path.stat()
    assert window.text_edit.toPlainText() == expected_content
    assert window._last_size == final_stat.st_size
    assert window._last_mtime == final_stat.st_mtime


def test_logview_truncate_content(log_view_window, qtbot):
    """Test truncating the log file (writing less data)."""
    window, log_file_path = log_view_window
    initial_stat = log_file_path.stat()
    truncated_content = "New Line 1"
    assert len(truncated_content) < initial_stat.st_size

    log_file_path.write_text(truncated_content)

    qtbot.wait(LOG_WAIT_MS)

    final_stat = log_file_path.stat()
    assert window.text_edit.toPlainText() == truncated_content
    assert window._last_size == final_stat.st_size
    assert window._last_mtime == final_stat.st_mtime


def test_logview_reappear_file(log_view_window, qtbot):
    """Test the file reappearing after deletion."""
    window, log_file_path = log_view_window

    log_file_path.unlink()
    qtbot.waitUntil(lambda: window._last_size is None, timeout=LOG_WAIT_MS)
    assert f"[Log file {log_file_path} not found or deleted]" in window.text_edit.toPlainText()
    assert window._last_mtime is None

    reappeared_content = "It's back!"
    log_file_path.write_text(reappeared_content)
    new_stat = log_file_path.stat()

    qtbot.wait(LOG_WAIT_MS)

    assert window.text_edit.toPlainText() == reappeared_content
    assert window._last_size == new_stat.st_size
    assert window._last_mtime == new_stat.st_mtime


def test_header_groups(qtbot):
    _, header = make_table_with_headers(
        qtbot, ["Trigger/Open", "Trigger/Rep", "Trigger/Rotor", "A/B/C", "D"]
    )

    widths = [70, 80, 90, 100, 110]
    for idx, width in enumerate(widths):
        header.resizeSection(idx, width)
    qtbot.waitUntil(lambda: header.sectionSize(4) == widths[4])

    header._rebuild_hierarchy()

    header.update_filtered_columns({1})
    QApplication.processEvents()

    # test max levels
    assert header._max_levels == 3
    assert header._levels_by_section[0] == ["Trigger", "Open"]
    assert header._levels_by_section[1] == ["Trigger", "Rep"]
    assert header._levels_by_section[2] == ["Trigger", "Rotor"]
    assert header._levels_by_section[3] == ["A", "B", "C"]
    assert header._levels_by_section[4] == ["D"]

    # test geometry
    assert header._group_span_columns(0, 0) == 3  # Trigger

    total = sum(widths[0:3])
    start0, width0 = header._group_geometry(0, 0)
    start2, width2 = header._group_geometry(2, 0)

    assert width0 == total
    assert width2 == total
    assert start0 == start2 == header.sectionViewportPosition(0)


def test_header_groups_with_offscreen_lead(qtbot):
    view, header = make_table_with_headers(
        qtbot, ["Trigger/Open", "Trigger/Rep", "Trigger/Rotor", "A/B", "C"]
    )

    widths = [220, 220, 220, 220, 220]
    for idx, width in enumerate(widths):
        header.resizeSection(idx, width)
    qtbot.waitUntil(lambda: header.sectionSize(4) == widths[4])

    # Force horizontal scrolling so the first "Trigger" section starts off-screen.
    view.horizontalScrollBar().setValue(widths[0] + 10)
    QApplication.processEvents()
    qtbot.waitUntil(lambda: header.sectionViewportPosition(0) == 0)

    total = sum(widths[0:3])
    start, width = header._group_geometry(2, 0)

    assert start == 0
    assert width == total
    assert header._group_span_columns(2, 0) == 3


def test_header_groups_toggle_resets_state(qtbot):
    _, header = make_table_with_headers(qtbot, ["Trigger/Open", "Trigger/Rep"])

    assert header._hierarchy_enabled is True
    assert header._max_levels == 2
    assert header._levels_by_section[0] == ["Trigger", "Open"]

    header.set_hierarchical_enabled(False)
    assert header._hierarchy_enabled is False
    assert header._max_levels == 1
    assert header._levels_by_section == {}

    header.set_hierarchical_enabled(True)
    assert header._hierarchy_enabled is True
    assert header._levels_by_section[0] == ["Trigger", "Open"]


def test_header_group_toggle_emits_signals(qtbot):
    table_view, header = make_table_with_headers(qtbot, ["Trigger/Open"])

    assert table_view.hierarchical_header_enabled is True
    assert header._hierarchy_enabled is True

    with qtbot.waitSignal(table_view.hierarchical_header_changed):
        table_view.set_hierarchical_header_enabled(False)

    assert table_view.hierarchical_header_enabled is False
    assert header._hierarchy_enabled is False

    with qtbot.waitSignal(table_view.hierarchical_header_changed):
        table_view.set_hierarchical_header_enabled(True)

    assert table_view.hierarchical_header_enabled is True
    assert header._hierarchy_enabled is True


def test_thumbnail_filter_accepts_sparkline():
    model = QtGui.QStandardItemModel(1, 1)
    item = QtGui.QStandardItem()
    line = np.vstack((np.arange(5), np.array([0, 1, 0, 1, 0], dtype=float)))
    item.setData(line, LINE_DATA_ROLE)
    model.setItem(0, 0, item)

    proxy = FilterProxy()
    proxy.setSourceModel(model)
    proxy.set_filter(0, ThumbnailFilter(0, show_with_thumbnail=True, show_without_thumbnail=False))
    assert proxy.rowCount() == 1
    proxy.set_filter(0, ThumbnailFilter(0, show_with_thumbnail=False, show_without_thumbnail=True))
    assert proxy.rowCount() == 0


def test_sparkline_hover_state(qtbot):
    view = TableView()
    model = QtGui.QStandardItemModel(1, 1)
    item = QtGui.QStandardItem()
    line = np.vstack((np.arange(10), np.arange(10, dtype=float)))
    item.setData(line, LINE_DATA_ROLE)
    model.setItem(0, 0, item)
    view.setModel(model)
    view.resize(200, 60)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    proxy_index = view.model().index(0, 0)
    rect = view.visualRect(proxy_index)
    qtbot.mouseMove(view.viewport(), rect.center())

    qtbot.waitUntil(lambda: view._sparkline_hover_index.isValid(), timeout=1000)
    assert view._sparkline_hover_index == proxy_index
    assert 0.0 <= view._sparkline_hover_t <= 1.0
    # Verify hover_t matches the mouse x-position mapping.
    expected_t = (rect.center().x() - rect.left()) / (rect.width() - 1)
    assert abs(view._sparkline_hover_t - expected_t) < 0.1

    view.leaveEvent(QEvent(QEvent.Type.Leave))
    qtbot.waitUntil(lambda: not view._sparkline_hover_index.isValid(), timeout=1000)
    # Leaving the cell clears the hover state.
    assert view._sparkline_hover_t is None


def test_sparkline_delegate_paint_smoke(qtbot):
    view = TableView()
    model = QtGui.QStandardItemModel(1, 1)
    item = QtGui.QStandardItem()
    line = np.vstack((np.arange(5), np.array([0, 1, 0, 1, 0], dtype=float)))
    item.setData(line, LINE_DATA_ROLE)
    model.setItem(0, 0, item)
    view.setModel(model)
    view.resize(200, 60)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    proxy_index = view.model().index(0, 0)
    option = QtWidgets.QStyleOptionViewItem()
    option.rect = view.visualRect(proxy_index)
    option.widget = view

    def image_bytes(img):
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        return bytes(ptr)

    delegate = view.itemDelegate()

    # Render without hover.
    image = QtGui.QImage(200, 60, QtGui.QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    delegate.paint(painter, option, proxy_index)
    painter.end()
    base_bytes = image_bytes(image)

    # Render with hover; output should differ from the baseline.
    hover_image = QtGui.QImage(200, 60, QtGui.QImage.Format.Format_ARGB32)
    hover_image.fill(Qt.GlobalColor.transparent)
    hover_painter = QtGui.QPainter(hover_image)
    view._sparkline_hover_index = proxy_index
    view._sparkline_hover_t = 0.5
    delegate.paint(hover_painter, option, proxy_index)
    hover_painter.end()
    hover_bytes = image_bytes(hover_image)

    assert base_bytes != hover_bytes


def test_adds_provenance_tooltip_to_items(mock_db, qtbot):
    _, db = mock_db
    model = DamnitTableModel(db, {}, None)

    item = model.new_item(42, "scalar1", 0, {}, provenance="pytest")
    assert item.data(PROVENANCE_ROLE) == "pytest"
    assert item.toolTip() == "Provenance: pytest"

    # html tooltip
    item = model.itemPrototype().clone()
    item.setToolTip('<img src="data:image/png;base64,abc">')
    model._apply_provenance_style(item, "pytest")
    assert item.data(PROVENANCE_ROLE) == "pytest"
    assert item.toolTip() == '<img src="data:image/png;base64,abc"><br/>Provenance: pytest'

    # skip default
    item = model.new_item(42, "scalar1", 0, {}, provenance="context.py")
    assert item.data(PROVENANCE_ROLE) is None
    assert not item.toolTip()


def test_item_delegate_paints_provenance_marker_smoke(qtbot):
    view = TableView()
    model = QtGui.QStandardItemModel(1, 1)
    item = QtGui.QStandardItem("value")
    model.setItem(0, 0, item)
    view.setModel(model)
    view.resize(200, 60)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    index = view.model().index(0, 0)
    option = QtWidgets.QStyleOptionViewItem()
    option.rect = view.visualRect(index)
    option.widget = view

    def image_bytes(img):
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        return bytes(ptr)

    delegate = view.itemDelegate()

    base_image = QtGui.QImage(200, 60, QtGui.QImage.Format.Format_ARGB32)
    base_image.fill(Qt.GlobalColor.transparent)
    base_painter = QtGui.QPainter(base_image)
    delegate.paint(base_painter, option, index)
    base_painter.end()
    base_bytes = image_bytes(base_image)

    item.setData("pytest", PROVENANCE_ROLE)

    marker_image = QtGui.QImage(200, 60, QtGui.QImage.Format.Format_ARGB32)
    marker_image.fill(Qt.GlobalColor.transparent)
    marker_painter = QtGui.QPainter(marker_image)
    delegate.paint(marker_painter, option, index)
    marker_painter.end()
    marker_bytes = image_bytes(marker_image)

    assert base_bytes != marker_bytes


def test_best_text_color_for_custom_background(mock_db):
    _, db = mock_db
    model = DamnitTableModel(db, {}, None)

    dark_item = model.new_item(42, "scalar1", 0, {"background": [12, 12, 12]})
    assert dark_item.background().color() == QColor(12, 12, 12)
    assert dark_item.foreground().color() == QColor(Qt.GlobalColor.white)

    light_item = model.new_item(42, "scalar1", 0, {"background": [240, 240, 240]})
    assert light_item.background().color() == QColor(240, 240, 240)
    assert light_item.foreground().color() == QColor(Qt.GlobalColor.black)


def test_variable_title_change(mock_db, mock_kafka_broker, monkeypatch, qtbot):
    db_dir, _ = mock_db
    ctx_path = db_dir / "context.py"

    old_title = "Scalar1"
    new_title = "Scalar1 updated"

    win = MainWindow(db_dir, background_activity=False)
    qtbot.addWidget(win)
    qtbot.waitUntil(lambda: win.update_agent.running, timeout=1000)

    def combo_items(combo_box):
        return [combo_box.itemText(i) for i in range(combo_box.count())]

    assert old_title in combo_items(win.plot._combo_box_y_axis)
    assert new_title not in combo_items(win.plot._combo_box_y_axis)

    ctx_path.write_text(
        ctx_path.read_text().replace(
            f'@Variable(title="{old_title}",',
            f'@Variable(title="{new_title}",',
        )
    )

    with monkeypatch.context() as m:
        m.chdir(db_dir)
        amore_proto(["read-context"])

    qtbot.waitUntil(lambda: new_title in combo_items(win.plot._combo_box_y_axis), timeout=1000)
    assert new_title in combo_items(win.plot._combo_box_y_axis)
    assert old_title not in combo_items(win.plot._combo_box_y_axis)
    assert win.table.find_column(new_title, by_title=True) == win.table.find_column(
        "scalar1", by_title=False
    )


def test_numeric_items_store_units(mock_db):
    _, db = mock_db
    model = DamnitTableModel(db, {}, None)

    item = model.new_item(42, "scalar1", 0, {"units": "mJ"})
    assert item.data(Qt.ItemDataRole.DisplayRole) == "42"
    assert item.data(Qt.ItemDataRole.UserRole) == 42
    assert item.data(UNITS_ROLE) == "mJ"

    item = model.new_item(3.14, "scalar2", 0, {"units": "uJ"})
    assert item.data(Qt.ItemDataRole.DisplayRole) == "3.1400"
    assert item.data(Qt.ItemDataRole.UserRole) == 3.14
    assert item.data(UNITS_ROLE) == "uJ"

    item = model.new_item("ready", "scalar1", 0, {"units": "mJ"})
    assert item.data(Qt.ItemDataRole.DisplayRole) == "ready"
    assert item.data(Qt.ItemDataRole.UserRole) == "ready"
    assert item.data(UNITS_ROLE) is None


def test_item_delegate_paints_units_smoke(qtbot):
    view = TableView()
    model = QtGui.QStandardItemModel(1, 1)
    item = QtGui.QStandardItem()
    item.setData(3.14, Qt.ItemDataRole.UserRole)
    item.setData("3.1400", Qt.ItemDataRole.DisplayRole)
    model.setItem(0, 0, item)
    view.setModel(model)
    view.resize(200, 60)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    index = view.model().index(0, 0)
    option = QtWidgets.QStyleOptionViewItem()
    option.rect = view.visualRect(index)
    option.widget = view

    def image_bytes(img):
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        return bytes(ptr)

    delegate = view.itemDelegate()

    base_image = QtGui.QImage(200, 60, QtGui.QImage.Format.Format_ARGB32)
    base_image.fill(Qt.GlobalColor.transparent)
    base_painter = QtGui.QPainter(base_image)
    delegate.paint(base_painter, option, index)
    base_painter.end()
    base_bytes = image_bytes(base_image)

    item.setData("uJ", UNITS_ROLE)

    units_image = QtGui.QImage(200, 60, QtGui.QImage.Format.Format_ARGB32)
    units_image.fill(Qt.GlobalColor.transparent)
    units_painter = QtGui.QPainter(units_image)
    delegate.paint(units_painter, option, index)
    units_painter.end()
    units_bytes = image_bytes(units_image)

    assert base_bytes != units_bytes
