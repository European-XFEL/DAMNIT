from unittest.mock import patch

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from amore_mid_prototype.gui.main_window import MainWindow


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
    win._save_shortcut.activated.emit()
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

    # 'No' should close the window but not save
    with patch.object(QMessageBox, "exec", return_value=QMessageBox.No):
        win.close()
        assert win.isHidden()
        assert ctx_path.read_text() == old_code

    # Show the window again
    win.show()
    qtbot.waitExposed(win)
    assert win.isVisible()

    # 'Yes' should close the window and save
    with patch.object(QMessageBox, "exec", return_value=QMessageBox.Yes):
        win.close()
        assert win.isHidden()
        assert ctx_path.read_text() == new_code

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
