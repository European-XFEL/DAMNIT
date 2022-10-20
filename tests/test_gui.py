from unittest.mock import patch

from PyQt5.QtWidgets import QMessageBox

from amore_mid_prototype.gui.main_window import MainWindow
from amore_mid_prototype.gui.editor import ContextTestResult


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
