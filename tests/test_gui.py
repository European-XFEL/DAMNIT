import shelve
import textwrap
from unittest.mock import patch

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from amore_mid_prototype.gui.editor import ContextTestResult
from amore_mid_prototype.gui.main_window import MainWindow, Settings


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
    assert not settings_db_path.parent.is_dir()

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
