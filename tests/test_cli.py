from pathlib import Path
from unittest.mock import patch, ANY
from contextlib import contextmanager

import pytest

from amore_mid_prototype.cli import main
from amore_mid_prototype.backend.db import get_meta


def test_new_id(mock_db, monkeypatch):
    db_dir, db = mock_db

    old_id = get_meta(db, "db_id")

    # Test setting the ID with an explicit path
    with patch("sys.argv", ["amore-proto", "new-id", str(db_dir)]):
        main()
    assert old_id != get_meta(db, "db_id")

    # Test with the default path (PWD)
    monkeypatch.chdir(db_dir)
    old_id = get_meta(db, "db_id")
    with patch("sys.argv", ["amore-proto", "new-id"]):
        main()
    assert old_id != get_meta(db, "db_id")

def test_debug_repl(mock_db, monkeypatch):
    # Change directory so we can call new-id safely
    db_dir, db = mock_db
    monkeypatch.chdir(db_dir)

    # Helper context manager that mocks sys.argv, run_app() to raise an
    # exception, and returns a mocked InteractiveShellEmbed from IPython.
    @contextmanager
    def helper_patch(args):
        with (patch("sys.argv", ["amore-proto", *args]),
              patch("amore_mid_prototype.gui.main_window.run_app", side_effect=RuntimeError),
              patch("amore_mid_prototype.cli.InteractiveShellEmbed") as repl):
            yield repl

    # Without --debug-repl, we should just get an exception
    with helper_patch(["gui"]) as repl:
        with pytest.raises(RuntimeError):
            main()

        repl.assert_not_called()

    # With --debug-repl, we should get a REPL
    with helper_patch(["--debug-repl", "gui"]) as repl:
        assert main() == 1

        repl.assert_called_once()

    # With --debug-repl and no exception, the REPL shouldn't be launched
    with helper_patch(["--debug-repl", "new-id"]) as repl:
        assert main() == 0
        repl.assert_not_called()

def test_gui():
    @contextmanager
    def helper_patch(args=[]):
        with (patch("sys.argv", ["amore-proto", "gui", *args]),
              patch("amore_mid_prototype.cli.find_proposal", return_value="/tmp"),
              patch("amore_mid_prototype.gui.main_window.run_app") as run_app):
            yield run_app

    # Check passing neither a proposal number or directory
    with helper_patch() as run_app:
        main()
        run_app.assert_called_with(None, connect_to_kafka=ANY)

    # Check passing a proposal number
    with helper_patch(["1234"]) as run_app:
        main()
        run_app.assert_called_with(Path("/tmp/usr/Shared/amore"), connect_to_kafka=ANY)

    # Check passing a directory
    with helper_patch(["/tmp"]) as run_app:
        main()
        run_app.assert_called_with(Path("/tmp"), connect_to_kafka=ANY)

    # Check invalid argument
    with helper_patch(["/nope"]) as run_app:
        with pytest.raises(SystemExit):
            main()
        run_app.assert_not_called()
