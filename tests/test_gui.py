import re
import os
import shelve
import textwrap
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QInputDialog, QDialog, QStyledItemDelegate, QLineEdit

from amore_mid_prototype.ctxsupport.ctxrunner import ContextFile, Results
from amore_mid_prototype.backend.db import db_path, add_user_variable
from amore_mid_prototype.backend.extract_data import add_to_db
from amore_mid_prototype.gui.editor import ContextTestResult
from amore_mid_prototype.gui.main_window import MainWindow, Settings, AddUserVariableDialog


# Check if a PID exists by using `kill -0`
def pid_dead(pid):
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True


def test_connect_to_kafka(mock_db, qtbot):
    db_dir, db = mock_db
    pkg = "amore_mid_prototype.gui.kafka"

    with patch(f"{pkg}.KafkaConsumer") as kafka_cns:
        MainWindow(db_dir, False).close()
        kafka_cns.assert_not_called()

    with patch(f"{pkg}.KafkaConsumer") as kafka_cns:
        MainWindow(db_dir, True).close()
        kafka_cns.assert_called_once()

def test_editor(mock_db, mock_ctx, qtbot):
    db_dir, db = mock_db
    ctx_path = db_dir / "context.py"
    ctx_path.write_text(mock_ctx.code)

    win = MainWindow(db_dir, False)
    win.show()
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
    win._save_btn.clicked.emit()
    assert editor.test_context()[0] == ContextTestResult.OK
    assert ctx_path.read_text() == old_code
    assert status_bar.currentMessage() == str(ctx_path.resolve())

    # The Validate button should trigger validation. Note that we mock
    # editor.test_context() function instead of MainWindow.test_context()
    # because the win._check_btn.clicked has already been connected to the
    # original function, so mocking it will not make Qt call the mock object.
    with patch.object(editor, "test_context", return_value=(None, None)) as test_context:
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

    # 'Save' should close the window and save
    with patch.object(QMessageBox, "exec", return_value=QMessageBox.Save):
        win.close()
        assert win.isHidden()
        assert ctx_path.read_text() == new_code

    # Attempting to save ERROR'ing code should not save anything
    editor.setText("123 = 456")
    win.save_context()
    assert ctx_path.read_text() == new_code

    # But saving WARNING code should work
    warning_code = textwrap.dedent("""
    import numpy as np
    x = 1
    """)
    editor.setText(warning_code)
    assert editor.test_context()[0] == ContextTestResult.WARNING
    win.save_context()
    assert ctx_path.read_text() == warning_code

def test_settings(mock_db_with_data, mock_ctx, tmp_path, monkeypatch, qtbot):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)

    # Store fake data in the DB
    runs = pd.read_sql_query("SELECT * FROM runs", db.conn)

    # Create the window with a mocked Path so that it saves the settings in the
    # home directory.
    with patch("pathlib.Path.home", return_value=tmp_path):
        win = MainWindow(db_dir, False)

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
    runs.insert(len(runs.columns), "new_var", np.random.rand(runs.shape[0]))
    runs.to_sql("runs", db, index=False, if_exists="replace")
    win.handle_update(msg)

    # The new column should be at the end
    headers = visible_headers()
    assert "new_var" == headers[-1]

    # And after reloading the database, the ordering should be the same
    win.autoconfigure(db_dir)
    assert headers == visible_headers()

    # Simulate adding a new column while the GUI is *not* running
    runs.insert(len(runs.columns), "newer_var", np.random.rand(runs.shape[0]))
    runs.to_sql("runs", db, index=False, if_exists="replace")

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

    assert win.data.shape[0] == 0
    win.handle_update(msg)
    assert win.data.shape[0] == 1
    assert win.table_view.model().rowCount() == 1

    # Columns should be added for the new variables
    headers = get_headers()
    assert "Scalar1" in headers
    assert "string" in headers

    # Send an update for an existing row
    msg["scalar1"] = 43
    win.handle_update(msg)
    assert model().data(model().index(0, headers.index("Scalar1"))) == str(msg["scalar1"])

    # Add a new column to an existing row
    msg["array"] = np.array([1,1,2])
    win.handle_update(msg)
    assert len(headers) + 1 == len(get_headers())
    assert "Array" in get_headers()

def test_autoconfigure(tmp_path, bound_port, request, qtbot):
    db_dir = tmp_path / "usr/Shared/amore"
    win = MainWindow(None, False)
    pkg = "amore_mid_prototype.gui.main_window"

    @contextmanager
    def helper_patch():
        # Patch things such that the GUI thinks we're on GPFS trying to open
        # p1234, and the user always wants to create a database and start the
        # backend.
        with (patch.object(win, "gpfs_accessible", return_value=True),
              patch.object(QInputDialog, "getInt", return_value=(1234, True)),
              patch(f"{pkg}.find_proposal", return_value=tmp_path),
              patch.object(QMessageBox, "question", return_value=QMessageBox.Yes),
              patch(f"{pkg}.initialize_and_start_backend") as initialize_and_start_backend,
              patch.object(win, "autoconfigure")):
            yield initialize_and_start_backend

    # Autoconfigure from scratch
    with (helper_patch() as initialize_and_start_backend,
          patch(f"{pkg}.backend_is_running", return_value=True)):
        win._menu_bar_autoconfigure()

        # We expect the database to be initialized and the backend started
        win.autoconfigure.assert_called_once_with(db_dir, proposal=1234)
        initialize_and_start_backend.assert_called_once_with(db_dir, 1234)

    # Create the directory and database file to fake the database already existing
    db_dir.mkdir(parents=True)
    db_path(db_dir).touch()

    # Autoconfigure again, the GUI should start the backend again
    with (helper_patch() as initialize_and_start_backend,
          patch(f"{pkg}.backend_is_running", return_value=False)):
        win._menu_bar_autoconfigure()

        # This time the database is already initialized
        win.autoconfigure.assert_called_once_with(db_dir, proposal=1234)
        initialize_and_start_backend.assert_called_once_with(db_dir)

def test_user_vars(mock_ctx_user, mock_user_vars, mock_db, qtbot):

    proposal = 1234
    run_number = 1000

    db_dir, db = mock_db
    ctx_path = db_dir / "context.py"
    ctx_path.write_text(mock_ctx_user.code)

    reduced_data = {
        "user_integer": 12,
        "user_number": 10.2,
        "user_boolean": True,
        "user_string": "foo",
        "dep_integer": 13,
        "dep_number": 10.2,
        "dep_boolean": False,
        "dep_string": "foofoo"
    }

    with db:
        add_to_db(reduced_data, db, proposal, run_number)


    win = MainWindow(connect_to_kafka=False)
    win.show()

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
    with db:
        for vv in mock_user_vars.values():
            add_user_variable(db, vv)

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
    assert db.execute("SELECT COUNT(*) FROM variables WHERE title = 'My cool integer'").fetchone()[0] == 0

    # Check that the variable with the already existing title is not created
    assert db.execute("SELECT COUNT(*) FROM variables WHERE name = 'my_cool_integer'").fetchone()[0] == 0

    row = db.execute("SELECT name, type, title, description FROM variables WHERE name = 'my_integer'").fetchone()

    # Check that the variable added via the dialog is actually created
    assert row is not None

    # Check that the fields of the variable added via the dialog was created
    assert tuple(row) == ("my_integer", "integer", "My integer", "My cool description")

    col_to_pos = {win.col_title_to_name(cc) : ii for ii, cc in enumerate(win.data.columns)}

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
    table_model = table_view.model()

    def open_editor_and_get_delegate(field_name, row_number = 0):
        table_view.edit(table_model.index(row_number, col_to_pos[field_name]))
        return table_view.itemDelegate()

    def change_to_value_and_close(value):
        delegate = table_view.itemDelegate()
        delegate.widget.insert(value)
        table_view.commitData(delegate.widget)
        table_view.closeEditor(delegate.widget, 0)

    def get_value_from_field(field_name, row_number = 0):
        return win.data.iloc[row_number, col_to_pos[field_name]]

    def get_value_from_db(field_name):
        if not re.fullmatch(r"[a-zA-Z_]\w+", field_name, flags=re.A):
            raise ValueError(f"Error in field_name: the variable name '{field_name}' is not of the form '[a-zA-Z_]\\w+'")
        return db.execute(f"SELECT {field_name} FROM runs WHERE runnr = ?", (run_number,)).fetchone()[0]

    # Check that editing is prevented when trying to modfiy a non-editable column 
    assert open_editor_and_get_delegate("dep_number").widget is None

    # Check that editing is allowed when trying to modfiy a user editable column 
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
    """
    ctx_code = mock_ctx.code + "\n\n" + textwrap.dedent(const_array_code)
    (db_dir / "context.py").write_text(ctx_code)
    ctx = ContextFile.from_str(ctx_code)

    runs = pd.read_sql_query("SELECT * FROM runs", db)
    runs["constant_array"] = np.ones(runs.shape[0])
    runs.to_sql("runs", db, index=False, if_exists="replace")

    # And to an HDF5 file
    proposal = runs["proposal"][0]
    run_number = runs["runnr"][0]
    results = Results.create(ctx, { "run_data": mock_run }, run_number, proposal)
    extracted_data_dir = db_dir / "extracted_data"
    extracted_data_dir.mkdir()
    results.save_hdf5(extracted_data_dir / f"p{proposal}_r{run_number}.h5")

    # Create window
    win = MainWindow(db_dir, False)

    # Helper function to get a QModelIndex from a variable title
    def get_index(title, row=0):
        col = list(win.data.columns).index(title)
        index = win.table.index(row, col)
        assert index.isValid()
        return index

    # We should be able to plot summaries
    win.plot._combo_box_x_axis.setCurrentText("Array")
    win.plot._combo_box_y_axis.setCurrentText("Constant array")

    with patch.object(QMessageBox, "warning") as warning:
        win.plot._button_plot_clicked(False)
        warning.assert_not_called()

    # And plot an array
    array_index = get_index("Array")
    with patch.object(QMessageBox, "warning") as warning:
        win.inspect_data(array_index)
        warning.assert_not_called()

    # And correlate two array variables
    win.table_view.setCurrentIndex(array_index)
    with patch.object(QMessageBox, "warning") as warning:
        win.plot._button_plot_clicked(True)
        warning.assert_not_called()

    # Check that the text for the array that changes is bold
    assert win.table.data(array_index, role=Qt.FontRole).bold()

    # But not for the constant array
    const_array_index = get_index("Constant array")
    assert win.table.data(const_array_index, role=Qt.FontRole) is None

    # Edit a comment
    comment_index = get_index("Comment")
    win.table.setData(comment_index, "Foo", Qt.EditRole)

    # Add a standalone comment
    row_count = win.table.rowCount()
    win.comment.setText("Bar")
    win._comment_button_clicked()
    assert win.table.rowCount() == row_count + 1

    # Edit a standalone comment
    comment_index = get_index("Comment", row=1)
    win.table.setData(comment_index, "Foo", Qt.EditRole)
