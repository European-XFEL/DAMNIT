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
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette, QPixmap
from PyQt5.QtWidgets import (QDialog, QFileDialog, QInputDialog, QLineEdit,
                             QMessageBox, QStyledItemDelegate)

import damnit
from damnit.backend.db import DamnitDB, ReducedData
from damnit.backend.extract_data import add_to_db
from damnit.ctxsupport.ctxrunner import ContextFile
from damnit.gui.editor import ContextTestResult
from damnit.gui.main_window import AddUserVariableDialog, MainWindow
from damnit.gui.open_dialog import OpenDBDialog
from damnit.gui.plot import HistogramPlotWindow, ScatterPlotWindow
from damnit.gui.standalone_comments import TimeComment
from damnit.gui.table_filter import (CategoricalFilter,
                                     CategoricalFilterWidget, ThumbnailFilterWidget, FilterMenu,
                                     NumericFilter, NumericFilterWidget, ThumbnailFilter)
from damnit.gui.theme import Theme
from damnit.gui.web_viewer import PlotlyPlot
from damnit.gui.zulip_messenger import ZulipConfig

from .helpers import extract_mock_run, mkcontext, reduced_data_from_dict


# Check if a PID exists by using `kill -0`
def pid_dead(pid):
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True


def test_connect_to_kafka(mock_db, qtbot):
    db_dir, db = mock_db
    consumer_import = "damnit.gui.main_window.KafkaConsumer"
    producer_import = "damnit.kafka.KafkaProducer"

    with patch(consumer_import) as kafka_cns, \
         patch(producer_import) as kafka_prd:
        win = MainWindow(db_dir, False)
        qtbot.addWidget(win)
        kafka_cns.assert_not_called()
        kafka_prd.assert_not_called()

    with patch(consumer_import) as kafka_cns, \
         patch(producer_import) as kafka_prd:
        win = MainWindow(db_dir, True)
        qtbot.addWidget(win, before_close_func=lambda _: win.stop_update_listener_thread())
        kafka_cns.assert_called_once()
        kafka_prd.assert_called_once()

def test_editor(mock_db, mock_ctx, qtbot):
    db_dir, db = mock_db
    ctx_path = db_dir / "context.py"
    ctx_path.write_text(mock_ctx.code)

    win = MainWindow(db_dir, False)
    win.show()
    qtbot.addWidget(win)
    editor = win._editor
    status_bar = win._status_bar

    # If the context file is not saved, the window will prompt the user about
    # it. This makes the tests hang, so before closing the window we manually
    # mark the context as saved. Useful if a test fails for some reason while
    # the context is changed.
    qtbot.addWidget(win, before_close_func=lambda win: win.mark_context_saved())
    qtbot.waitExposed(win)

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
    with patch.object(QMessageBox, "exec", return_value=QMessageBox.Cancel):
        win.close()
        assert win.isVisible()
        assert ctx_path.read_text() == old_code

    # 'Discard' should close the window but not save
    with patch.object(QMessageBox, "exec", return_value=QMessageBox.Discard):
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
        headers = [header.model().headerData(header.logicalIndex(i), Qt.Horizontal) for i in range(header.count())
                   if not header.isSectionHidden(i)]
        return headers

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
        col_one_item = columns.findItems(col_one, Qt.MatchExactly)[0]
        col_two_item = columns.findItems(col_two, Qt.MatchExactly)[0]
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

    columns_widget = win.table_view._columns_widget
    static_columns_widget = win.table_view._static_columns_widget

    last_col_item = columns_widget.item(columns_widget.count() - 1)
    last_col = last_col_item.text()

    # Hide a column
    assert last_col in visible_headers()
    last_col_item.setCheckState(Qt.Unchecked)
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
        "Proposal": 1234,
        "Run": 1,
        "new_var": 42
    }
    db.set_variable(msg["Proposal"], msg["Run"], "new_var", ReducedData(msg["new_var"]))
    win.handle_update(msg)

    # The new column should be at the end
    headers = visible_headers()
    assert "new_var" == headers[-1]

    # And after reloading the database, the ordering should be the same
    win.autoconfigure(db_dir)
    assert headers == visible_headers()

    # Simulate adding a new column while the GUI is *not* running
    db.set_variable(msg["Proposal"], msg["Run"], "newer_var", ReducedData("foo"))

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
    last_static_col_item.setCheckState(Qt.Unchecked)
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
    get_headers = lambda: [win.table_view.model().headerData(i, Qt.Horizontal)
                           for i in range(win.table_view.horizontalHeader().count())]

    # Sending an update should add a row to the table
    msg = {
        "Proposal": 1234,
        "Run": 1,
        "scalar1": 42,
        "string": "foo"
    }

    assert win.table.rowCount() == 0
    win.handle_update(msg)
    assert win.table.rowCount() == 1

    # Columns should be added for the new variables
    headers = get_headers()
    assert "Scalar1" in headers
    assert "string" in headers

    # Send an update for an existing row
    msg["scalar1"] = 43
    win.handle_update(msg)
    assert model().data(model().index(0, headers.index("Scalar1"))) == str(msg["scalar1"])

    # Add a new column to an existing row
    msg["unexpected_var"] = 7
    win.handle_update(msg)
    assert len(headers) + 1 == len(get_headers())
    assert "unexpected_var" in get_headers()

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
        "Proposal": 1234,
        "Run": 2,
        "scalar1": 42,
        "string": "foo"
    }
    win.handle_update(msg)

def test_autoconfigure(tmp_path, bound_port, request, qtbot):
    db_dir = tmp_path / "usr/Shared/amore"
    win = MainWindow(None, False)
    qtbot.addWidget(win)
    pkg = "damnit.gui.main_window"
    template_path = Path(damnit.__file__).parent / 'ctx-templates' / 'SA1_base.py'

    @contextmanager
    def helper_patch():
        # Patch things such that the GUI thinks we're on GPFS trying to open
        # p1234, and the user always wants to create a database and start the
        # backend.
        with (patch(f"{pkg}.OpenDBDialog.run_get_result", return_value=(db_dir, 1234)),
              patch(f"{pkg}.NewContextFileDialog.run_get_result", return_value=template_path),
              patch.object(QMessageBox, "question", return_value=QMessageBox.Yes),
              patch(f"{pkg}.initialize_and_start_backend") as initialize_and_start_backend,
              patch.object(win, "autoconfigure")):
            yield initialize_and_start_backend

    # Autoconfigure from scratch
    with (helper_patch() as initialize_and_start_backend,
          patch(f"{pkg}.backend_is_running", return_value=True)):
        win._menu_bar_autoconfigure()

        # We expect the database to be initialized and the backend started
        win.autoconfigure.assert_called_once_with(db_dir)
        initialize_and_start_backend.assert_called_once_with(db_dir, 1234, template_path)

    # Create the directory and database file to fake the database already existing
    db_dir.mkdir(parents=True)
    DamnitDB.from_dir(db_dir)

    # Autoconfigure with database present & backend 'running':
    with (helper_patch() as initialize_and_start_backend,
          patch(f"{pkg}.backend_is_running", return_value=True)):
        win._menu_bar_autoconfigure()

        # We expect the database to be initialized and the backend started
        win.autoconfigure.assert_called_once_with(db_dir)
        initialize_and_start_backend.assert_not_called()

    # Autoconfigure again, the GUI should start the backend again
    with (helper_patch() as initialize_and_start_backend,
          patch(f"{pkg}.backend_is_running", return_value=False)):
        win._menu_bar_autoconfigure()

        # This time the database is already initialized
        win.autoconfigure.assert_called_once_with(db_dir)
        initialize_and_start_backend.assert_called_once_with(db_dir, 1234)

def test_user_vars(mock_ctx_user, mock_user_vars, mock_db, qtbot):

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
        add_to_db(reduced_data, db, proposal, run_number)


    win = MainWindow(connect_to_kafka=False)
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
    # The menu should not be enabled if the context file is not loaded
    assert not create_user_menu.isEnabled()

    # Adds the variables to the db
    for vv in mock_user_vars.values():
        db.add_user_variable(vv, exist_ok=True)

    # Loads the context file to do the other tests
    win.autoconfigure(db_dir)

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

    with patch.object(QMessageBox, "exec", return_value=QMessageBox.Cancel):
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
        assert add_var_win.result() == QDialog.Accepted

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
        table_view.closeEditor(delegate.widget, 0)

    def get_value_from_field(field_name, row_number = 0):
        col_num = table_model.find_column(field_name)
        return table_model.get_value_at_rc(row_number, col_num)

    def get_value_from_db(field_name):
        if not re.fullmatch(r"[a-zA-Z_]\w+", field_name, flags=re.A):
            raise ValueError(f"Error in field_name: the variable name '{field_name}' is not of the form '[a-zA-Z_]\\w+'")
        return db.conn.execute(f"SELECT {field_name} FROM runs WHERE run = ?", (run_number,)).fetchone()[0]

    # Check that editing is prevented when trying to modfiy a non-editable column 
    assert open_editor_and_get_delegate("dep_number").widget is None

    # Check that editing is allowed when trying to modify a user editable column
    assert open_editor_and_get_delegate("user_number").widget is not None

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
    change_to_value_and_close("")
    assert pd.isna(get_value_from_field("user_boolean"))

    # Check that the value in the db matches what was typed in the table
    assert get_value_from_db("user_boolean") is None

def test_table_and_plotting(mock_db_with_data, mock_ctx, mock_run, monkeypatch, qtbot):
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
    ctx = ContextFile.from_str(ctx_code)
    extract_mock_run(1)

    # Create window
    win = MainWindow(db_dir, False)
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
    assert win.table.data(array_index, role=Qt.FontRole).bold()

    # But not for the constant array
    const_array_index = get_index("Constant array")
    assert win.table.data(const_array_index, role=Qt.FontRole) is None

    # Edit a comment
    comment_index = get_index("Comment")
    win.table.setData(comment_index, "Foo", Qt.EditRole)

    # Check that 2D arrays are treated as images
    image_index = get_index("Image")
    assert isinstance(win.table.data(image_index, role=Qt.DecorationRole), QPixmap)
    with patch.object(QMessageBox, "warning") as warning:
        win.inspect_data(image_index)
        warning.assert_not_called()

    # And that 3D image arrays are also treated as images
    color_image_index = get_index("Color image")
    assert isinstance(win.table.data(color_image_index, role=Qt.DecorationRole), QPixmap)
    with patch.object(QMessageBox, "warning") as warning:
        win.inspect_data(color_image_index)
        warning.assert_not_called()

    # Check that 2D arrays with summary are inspectable
    mean_2d_index = get_index('2D data with summary')
    assert win.table.data(mean_2d_index, role=Qt.FontRole).bold()
    assert isinstance(win.table.data(mean_2d_index, role=Qt.DisplayRole), str)
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
        win.table.data(get_index('error'), role=Qt.DecorationRole), QColor
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
        plot_window._plot_as_lines_checkbox.setCheckState(Qt.Checked)
        plot_window.figure.canvas.flush_events()

        assert not plot_window._dynamic_aspect_checkbox.isEnabled()
        assert plot_window._axis.get_aspect() == 'auto'

        # Toggle back to image mode
        plot_window._plot_as_lines_checkbox.setCheckState(Qt.Unchecked)
        plot_window.figure.canvas.flush_events()

        assert plot_window._dynamic_aspect_checkbox.isEnabled()

    # Test the line plot with xarray data
    xarray_line_data_index = get_index('2D xarray for line plot')
    with patch.object(QMessageBox, "warning") as warning:
        win.inspect_data(xarray_line_data_index)
        xr_plot_window = win._canvas_inspect[-1]
        warning.assert_not_called()

        xr_plot_window._plot_as_lines_checkbox.setCheckState(Qt.Checked)
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

@pytest.mark.parametrize("extension", [".xlsx", ".csv"])
def test_exporting(mock_db_with_data, qtbot, monkeypatch, extension):
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

    win = MainWindow(db_dir, connect_to_kafka=False)
    qtbot.addWidget(win)

    export_path = db_dir / f"export{extension}"
    filter_str = f"Ext (*{extension})"
    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(export_path.stem), filter_str)):
        win.export_table()

    assert export_path.is_file()

    # Check that images are formatted nicely
    df = pd.read_excel(export_path) if extension == ".xlsx" else pd.read_csv(export_path)
    assert df["Image"][0] == "<image>"

def test_delete_variable(mock_db_with_data, qtbot, monkeypatch):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)

    # We'll delete the 'array' variable
    assert "array" in db.variable_names()
    win = MainWindow(db_dir, connect_to_kafka=False)
    qtbot.addWidget(win)
    tbl = win.table
    column_ids_before = [tbl.column_id(i) for i in range(tbl.columnCount())]
    column_titles_before = tbl.column_titles.copy()
    assert "array" in column_ids_before
    col_visibility_before = win.table_view.get_column_states()
    assert "Array" in col_visibility_before  # Keyed by title, not column ID

    # If the user clicks 'No' then we should do nothing
    with patch.object(QMessageBox, "warning", return_value=QMessageBox.No) as warning:
        win.table_view.confirm_delete_variable("array")
        warning.assert_called_once()
    assert "array" in db.variable_names()
    assert [tbl.column_id(i) for i in range(tbl.columnCount())] == column_ids_before
    assert tbl.column_titles == column_titles_before
    assert win.table_view.get_column_states() == col_visibility_before

    # Otherwise it should be deleted from the database and HDF5 files
    with patch.object(QMessageBox, "warning", return_value=QMessageBox.Yes) as warning:
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

def test_precreate_runs(mock_db_with_data, qtbot, monkeypatch):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)

    win = MainWindow(db_dir, connect_to_kafka=False)
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
    static_item.setCheckState(Qt.Unchecked)
    assert count_visible_static() == initial_static_count - 1

    # Hide a variable column
    table_view._columns_widget.findItems('Scalar1', Qt.MatchExactly)[0].setCheckState(Qt.Unchecked)
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
    assert model.data(index, Qt.DisplayRole) == test_comment

    # Add another comment
    test_timestamp2 = 1641081600  # 2022-01-02 00:00:00
    test_comment2 = "Test comment 2"
    model.addComment(test_timestamp2, test_comment2)

    # Test sorting
    # Sort by timestamp ascending
    model.sort(1, Qt.AscendingOrder)
    index = model.index(0, 2)
    assert model.data(index, Qt.DisplayRole) == test_comment

    # Sort by timestamp descending
    model.sort(1, Qt.DescendingOrder)
    index = model.index(0, 2)
    assert model.data(index, Qt.DisplayRole) == test_comment2

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


def test_processing_status(mock_db_with_data, qtbot):
    db_dir, db = mock_db_with_data
    win = MainWindow(db_dir, connect_to_kafka=False)
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
        assert dark_palette.color(QPalette.Window).name() == "#353535"  # Dark theme color
        assert dark_palette.color(QPalette.WindowText).name() == "#ffffff"  # White text

        # Test theme application to editor
        assert win2._editor._lexer.defaultPaper(0).name() == "#232323"  # Dark theme editor background

        # Test theme toggle back to light
        win2._toggle_theme(False)
        assert win2.current_theme == Theme.LIGHT
        assert win2.palette() != dark_palette  # Light theme should have different colors


def test_filter_header(mock_db_with_data, qtbot):
    """Test that the filter header shows icons when filters are active."""
    window = MainWindow(mock_db_with_data[0], connect_to_kafka=False)
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
