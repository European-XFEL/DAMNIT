import os
import shelve
import textwrap
import logging
import subprocess
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMessageBox, QInputDialog
from threading import Thread

from amore_mid_prototype.backend.db import db_path
from amore_mid_prototype.gui.editor import ContextTestResult
from amore_mid_prototype.gui.main_window import MainWindow, Settings

log = logging.getLogger(__name__)

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

    # It would be nice to use qtbot.keyClick() to test the save shortcut, but I
    # couldn't get that to work. Possibly related:
    # https://github.com/pytest-dev/pytest-qt/issues/254
    win._editor.save_requested.emit()

    # Saving OK code should work
    assert editor.test_context()[0] == ContextTestResult.OK
    assert ctx_path.read_text() == old_code
    assert status_bar.currentMessage() == str(ctx_path.resolve())

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

def test_settings(mock_db, mock_ctx, tmp_path, qtbot):
    db_dir, db = mock_db

    # Store fake data in the DB
    runs_cols = ["proposal", "runnr", "start_time", "added_at", "comment"] + list(mock_ctx.vars.keys())
    runs = pd.DataFrame(np.random.randint(100, size=(10, len(runs_cols))),
                        columns=runs_cols)
    time_comments = pd.DataFrame(columns=["timestamp", "comment"])
    runs.to_sql("runs", db, index=False, if_exists="replace")
    time_comments.to_sql("time_comments", db, index=False, if_exists="replace")

    # Create the window with a mocked Path so that it saves the settings in the
    # home directory.
    with patch("pathlib.Path.home", return_value=tmp_path):
        win = MainWindow(db_dir, False)

    settings_db_path = tmp_path / ".local/state/damnit/settings.db"

    columns_widget = win.table_view._columns_widget
    static_columns_widget = win.table_view._static_columns_widget

    last_col_item = columns_widget.item(columns_widget.count() - 1)
    last_col = last_col_item.text()
    last_static_col = static_columns_widget.item(static_columns_widget.count() - 1).text()

    # Hide a column
    last_col_item.setCheckState(Qt.Unchecked)

    # This should have triggered a save, so the settings directory should now
    # exist.
    assert settings_db_path.parent.is_dir()

    # Open the saved settings
    with shelve.open(str(settings_db_path)) as settings_db:
        db_key = list(settings_db.keys())[0]
        settings = settings_db[db_key]
    col_settings = settings[Settings.COLUMNS.value]

    # Check that the visiblity has been saved
    assert col_settings[last_col] == False
    assert all(visible for col, visible in col_settings.items() if col != last_col)

    # Check the order has been saved by calculating the differences between
    # indices of each column in the saved settings. If they are all one after
    # another then the difference should always be 1.
    df_cols = win.data.columns.tolist()
    np.testing.assert_array_equal(np.diff([df_cols.index(col) for col in col_settings.keys()]),
                                  np.ones(len(col_settings) - 1))

    # Change the settings manually
    col_settings = dict(reversed(col_settings.items()))
    col_settings[last_static_col] = False
    with shelve.open(str(settings_db_path)) as settings_db:
        settings_db[db_key] = {Settings.COLUMNS.value: col_settings}

    # Reconfigure to load the new settings
    win.autoconfigure(db_dir)

    # Check the ordering
    df_cols = win.data.columns.tolist()
    np.testing.assert_array_equal(np.diff([df_cols.index(col) for col in col_settings.keys()]),
                                  np.ones(len(col_settings) - 1))

    # Check that last_static_col is hidden
    last_static_col_idx = win.data.columns.get_loc(last_static_col)
    assert win.table_view.isColumnHidden(last_static_col_idx)

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
    msg["array"] = 3.14
    win.handle_update(msg)
    assert len(headers) + 1 == len(get_headers())
    assert "Array" in get_headers()
    
class Helper():
    # tiny class to retrieve values from a callback used in test_log_widget
    def __init__(self):
        self.value =  ''
    def callback(self, value):
        self.value = value 

def test_log_widget(qtbot, mock_db):
    db_dir, db = mock_db
    win = MainWindow(db_dir, False)
    mock_error = "1993-09-28 13:00:00 ERROR  An error was generated"
    helper = Helper()    
    def assert_mock_error():
        win.log_widget.page().toHtml(helper.callback)
        #print(helper.value)
        assert mock_error in helper.value
        
    qtbot.waitUntil(lambda: win.log_def.is_log_view_finished == True)
    with open(str(win.log_def.be_log_path), 'a') as f:
        f.write(mock_error)

    qtbot.waitUntil(assert_mock_error, timeout = 1000)
        
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



