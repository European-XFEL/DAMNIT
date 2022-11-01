import os
import signal
import stat
import textwrap
import subprocess
import configparser
from contextlib import contextmanager
from unittest.mock import patch, MagicMock, DEFAULT

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QInputDialog

from amore_mid_prototype.backend.db import open_db, get_meta
from amore_mid_prototype.gui.main_window import MainWindow
from amore_mid_prototype.gui.editor import ContextTestResult


# Send SIGTERM to a process. Used for killing supervisord.
def kill_pid(pid):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"PID {pid} doesn't exist")

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

def test_autoconfigure(tmp_path, bound_port, request, qtbot):
    db_dir = tmp_path / "usr/Shared/amore"
    pid_path = db_dir / "supervisord.pid"
    config_path = db_dir / "supervisord.conf"
    win = MainWindow(None, False)

    win_initialize_database = win.initialize_database
    def mock_initialize_database(path, proposal):
        win_initialize_database(path, proposal)

        # Tweak the config to run a dummy command
        config = configparser.ConfigParser()
        config.read(config_path)
        config["program:damnit"]["command"] = "sleep 1m"

        with open(config_path, "w") as f:
            config.write(f)

    @contextmanager
    def helper_patch():
        # Patch things such that the GUI thinks we're on GPFS trying to open
        # p1234, and the user always wants to create a database and start the
        # backend.
        with (patch.object(win, "gpfs_accessible", return_value=True),
              patch.object(QInputDialog, "getInt", return_value=(1234, True)),
              patch("amore_mid_prototype.gui.main_window.find_proposal", return_value=tmp_path),
              patch.object(QMessageBox, "question", return_value=QMessageBox.Yes),
              patch.multiple(win,
                             initialize_database=MagicMock(wraps=mock_initialize_database),
                             start_backend=MagicMock(wraps=win.start_backend),
                             autoconfigure=DEFAULT)):
            yield

    # Autoconfigure from scratch
    with helper_patch():
        win._menu_bar_autoconfigure()

        # Add a finalizer to make sure supervisord is killed at the end
        # The PID file isn't created immediately so we wait a bit for it
        qtbot.waitUntil(lambda: pid_path.is_file())
        pid = int(pid_path.read_text())
        request.addfinalizer(lambda: kill_pid(pid))

        # We expect the database to be initialized and the backend started
        win.initialize_database.assert_called_once_with(db_dir, 1234)
        win.start_backend.assert_called_once_with(db_dir)
        win.autoconfigure.assert_called_once_with(db_dir, proposal=1234)

        assert win.backend_is_running(db_dir)

    # Stop the backend
    supervisorctl = ["supervisorctl", "-c", str(config_path)]
    subprocess.run([*supervisorctl, "stop", "damnit"]).check_returncode()
    assert not win.backend_is_running(db_dir, timeout=0)

    # Autoconfigure again, this time with supervisord running but the backend
    # stopped.
    with helper_patch():
        win._menu_bar_autoconfigure()

        # This time the database is already initialized
        win.initialize_database.assert_not_called()
        # But we still need to start the backend
        win.start_backend.assert_called_once_with(db_dir)

        assert win.backend_is_running(db_dir)

    # Stop supervisord and the backend
    kill_pid(pid)
    qtbot.waitUntil(lambda: pid_dead(pid))
    assert not pid_path.is_file()

    # Change the config so that the supervisord port is unavailable
    config = configparser.ConfigParser()
    config.read(config_path)
    config["inet_http_server"]["port"] = str(bound_port)
    with open(config_path, "w") as f:
        config.write(f)

    # Autoconfigure with supervisord stopped and its port in use
    with helper_patch():
        win._menu_bar_autoconfigure()

        # The port should have been changed
        config.read(config_path)
        assert int(config["inet_http_server"]["port"]) != bound_port

        # supervisord should've been started
        assert pid_path.is_file()
        # Add finalizer to kill supervisord
        pid = int(pid_path.read_text())
        request.addfinalizer(lambda: kill_pid(pid))

        # The backend should be running
        assert win.backend_is_running(db_dir)

    # Stop the backend and supervisord
    kill_pid(pid)
    qtbot.waitUntil(lambda: pid_dead(pid))
    assert not pid_path.is_file()

    # Delete the config file
    config_path.unlink()

    # Autoconfigure with no supervisord configuration, such as for proposals
    # previously using tmux to manage the backend.
    with helper_patch():
        win._menu_bar_autoconfigure()

        # supervisord should have been started
        assert config_path.is_file()
        assert pid_path.is_file()

        # Add finalizer to kill it
        pid = int(pid_path.read_text())
        request.addfinalizer(lambda: kill_pid(pid))

        # The backend should be running
        assert win.backend_is_running(db_dir)

def test_backend_management(tmp_path, request, qtbot):
    db_dir = tmp_path / "foo"
    good_file_mode = "-rw-rw-rw-"
    path_filemode = lambda p: stat.filemode(p.stat().st_mode)

    win = MainWindow(None, False)
    win.initialize_database(db_dir, 1234)

    # The directory should be created if it doesn't exist
    assert db_dir.is_dir()
    # And be writable by everyone
    assert path_filemode(db_dir) == "drwxrwxrwx"

    # Check that the database was initialized correctly
    db_path = db_dir / "runs.sqlite"
    assert db_path.is_file()
    assert path_filemode(db_path) == good_file_mode
    db = open_db(db_path)
    assert get_meta(db, "proposal") == 1234

    # Check the context file
    context_path = db_dir / "context.py"
    assert context_path.is_file()
    assert path_filemode(context_path) == good_file_mode

    # Check the config file
    supervisord_config_path = db_dir / "supervisord.conf"
    assert supervisord_config_path.is_file()
    assert path_filemode(supervisord_config_path) == good_file_mode

    config = configparser.ConfigParser()
    config.read(supervisord_config_path)
    assert config["program:damnit"]["command"] == "amore-proto listen ."

    # Modify the config to run a dummy command for testing. This script will
    # create a subprocess that creates a file named 'started' before it sleeps
    # for 10s. A handler is configured to write a file named 'stopped' upon a
    # SIGTERM before exiting. We need to test that child processes are killed
    # because the listener creates subprocesses to process each run.
    subprocess_code = """
    import sys
    import time
    import signal
    from pathlib import Path
    from multiprocessing import Process

    def handler(signum, frame):
        (Path.cwd() / "stopped").write_text("")
        sys.exit(0)

    def subprocess_runner():
        signal.signal(signal.SIGTERM, handler)
        (Path.cwd() / "started").write_text("")
        time.sleep(10)

    p = Process(target=subprocess_runner)
    p.start()
    p.join()
    """

    dummy_command_path = db_dir / "dummy.py"
    dummy_command_path.write_text(textwrap.dedent(subprocess_code))
    config["program:damnit"]["command"] = "python dummy.py"
    with open(supervisord_config_path, "w") as f:
        config.write(f)

    # Test starting the backend
    win.start_backend(db_dir)

    # We should have a log file and PID file. The PID file isn't created
    # immediately so we wait a bit for it.
    assert (db_dir / "supervisord.log").is_file()
    pid_path = db_dir / "supervisord.pid"
    qtbot.waitUntil(lambda: pid_path.is_file(), timeout=1000)
    pid = int(pid_path.read_text())

    # Set a finalizer to kill supervisord at the end
    request.addfinalizer(lambda: kill_pid(pid))

    # Check that it's running
    supervisorctl = ["supervisorctl", "-c", str(supervisord_config_path)]
    assert win.backend_is_running(db_dir)
    qtbot.waitUntil(lambda: (db_dir / "started").is_file(), timeout=1000)

    # Stop the program
    subprocess.run([*supervisorctl, "stop", "damnit"]).check_returncode()

    # Check that the subprocess was also killed
    qtbot.waitUntil(lambda: (db_dir / "stopped").is_file(), timeout=1000)
