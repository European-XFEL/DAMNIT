import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, ANY
from contextlib import contextmanager

import pytest
from testpath import MockCommand

from damnit.cli import main, excepthook as ipython_excepthook


def test_new_id(mock_db, monkeypatch):
    db_dir, db = mock_db

    old_id = db.metameta["db_id"]

    # Test setting the ID with an explicit path
    with patch("sys.argv", ["amore-proto", "new-id", str(db_dir)]):
        main()
    assert old_id != db.metameta["db_id"]

    # Test with the default path (PWD)
    monkeypatch.chdir(db_dir)
    old_id = db.metameta["db_id"]
    with patch("sys.argv", ["amore-proto", "new-id"]):
        main()
    assert old_id != db.metameta["db_id"]

def test_debug_repl(mock_db, monkeypatch):
    import IPython

    # Helper context manager that mocks sys.argv, run_app(), and the sys module
    @contextmanager
    def amore_proto(args):
        pkg = "damnit"
        with (patch("sys.argv", ["amore-proto", *args]),
              patch(f"{pkg}.gui.main_window.run_app"),
              patch(f"{pkg}.cli.sys") as mock_sys):
            yield mock_sys

    # We use sys.excepthook, but this function is only used for unhandled
    # exceptions, and pytest will always catch unhandled exceptions from our
    # code, which means that our hook will never be called during tests. So
    # instead, we check that the hook is not set when not asked for:
    with amore_proto(["gui"]) as mock_sys:
        old_excepthook = mock_sys.excepthook
        main()
        assert mock_sys.excepthook == old_excepthook

    # And that it is set when asked for:
    with amore_proto(["--debug-repl", "gui"]) as mock_sys:
        assert mock_sys.excepthook != ipython_excepthook
        main()
        assert mock_sys.excepthook == ipython_excepthook

    # And then test the hook separately
    try:
        raise RuntimeError("Foo")
    except:
        exc_type, value, tb = sys.exc_info()

    with patch.object(IPython, "start_ipython") as repl:
        ipython_excepthook(exc_type, value, tb)
        repl.assert_called_once()

def test_gui():
    @contextmanager
    def helper_patch(args=[]):
        with (patch("sys.argv", ["amore-proto", "gui", *args]),
              patch("damnit.cli.find_proposal", return_value="/tmp"),
              patch("damnit.gui.main_window.run_app") as run_app):
            yield run_app

    # Check passing neither a proposal number or directory
    with helper_patch() as run_app:
        main()
        run_app.assert_called_with(None, software_opengl=False, connect_to_kafka=ANY)

    # Check passing a proposal number
    with helper_patch(["1234"]) as run_app:
        main()
        run_app.assert_called_with(Path("/tmp/usr/Shared/amore"), software_opengl=False, connect_to_kafka=ANY)

    # Check passing a directory
    with helper_patch(["/tmp"]) as run_app:
        main()
        run_app.assert_called_with(Path("/tmp"), software_opengl=False, connect_to_kafka=ANY)

    # Check invalid argument
    with helper_patch(["/nope"]) as run_app:
        with pytest.raises(SystemExit):
            main()
        run_app.assert_not_called()

def test_listen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pkg = "damnit.backend"

    # Helper context manager that mocks sys.argv
    @contextmanager
    def amore_proto(args):
        with (patch("sys.argv", ["amore-proto", *args]),
              patch(f"{pkg}.initialize_and_start_backend") as initialize_and_start_backend):
            yield initialize_and_start_backend

    with (amore_proto(["listen"]),
          patch(f"{pkg}.listener.listen") as listen):
        main()
        listen.assert_called_once()

    with (amore_proto(["listen", "--test"]),
          patch(f"{pkg}.test_listener.listen") as listen):
        main()
        listen.assert_called_once()

    # Should fail without an existing database
    with (amore_proto(["listen", "--daemonize"]) as initialize_and_start_backend,
          pytest.raises(SystemExit)):
        main()
        initialize_and_start_backend.assert_not_called()

    # Should work with an existing database
    (tmp_path / "runs.sqlite").touch()
    with amore_proto(["listen", "--daemonize"]) as initialize_and_start_backend:
        main()
        initialize_and_start_backend.assert_called_once()

    # Can't pass both --test and --daemonize
    with (amore_proto(["listen", "--daemonize", "--test"]),
          pytest.raises(SystemExit)):
        main()

def test_reprocess(mock_db_with_data, monkeypatch):
    db_dir, db = mock_db_with_data
    db.metameta["proposal"] = 1234
    monkeypatch.chdir(db_dir)

    # Create a proposal directory with raw/
    raw_dir = db_dir / "mock_proposal" / "raw"
    raw_dir.mkdir(parents=True)

    # Helper context manager to patch KafkaProducer to do nothing and
    # find_proposal() to return the mock proposal directory we created.
    @contextmanager
    def amore_proto(args):
        with (patch("sys.argv", ["amore-proto", *args]),
              patch("damnit.backend.extraction_control.find_proposal", return_value=raw_dir.parent)):
            yield

    def mock_sbatch():
        return MockCommand.fixed_output("sbatch", "9876; maxwell")

    # Since none of the runs in the database exist on disk, we should skip all
    # of them (i.e. not call sbatch).
    with amore_proto(["reprocess", "all"]):
        with mock_sbatch() as sbatch:
            main()

        assert sbatch.get_calls() == []

    # Create raw/ directories for 10 runs
    for i in range(10):
        (raw_dir / f"r{i:04}").mkdir()

    # Reprocessing run 1 should call sbatch, because the run directory exists
    with amore_proto(["reprocess", "1"]):
        with mock_sbatch() as sbatch:
            main()

        sbatch.assert_called()

    # No directory for run 10, so this shouldn't try to submit a job
    with amore_proto(["reprocess", "10"]):
        with mock_sbatch() as sbatch:
            main()

        assert sbatch.get_calls() == []
