import sys
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
    main(["new-id", str(db_dir)])
    assert old_id != db.metameta["db_id"]

    # Test with the default path (PWD)
    monkeypatch.chdir(db_dir)
    old_id = db.metameta["db_id"]
    main(["new-id"])
    assert old_id != db.metameta["db_id"]

def test_debug_repl(mock_db, monkeypatch):
    import IPython

    pkg = "damnit"

    # We use sys.excepthook, but this function is only used for unhandled
    # exceptions, and pytest will always catch unhandled exceptions from our
    # code, which means that our hook will never be called during tests. So
    # instead, we check that the hook is not set when not asked for:
    with (patch(f"{pkg}.gui.main_window.run_app"),
          patch(f"{pkg}.cli.sys") as mock_sys):
        old_excepthook = mock_sys.excepthook
        main(["gui"])
        assert mock_sys.excepthook == old_excepthook

    # And that it is set when asked for:
    with (patch(f"{pkg}.gui.main_window.run_app"),
          patch(f"{pkg}.cli.sys") as mock_sys):
        assert mock_sys.excepthook != ipython_excepthook
        main(["--debug-repl", "gui"])
        assert mock_sys.excepthook == ipython_excepthook

    # And then test the hook separately
    try:
        raise RuntimeError("Foo")
    except:
        exc_type, value, tb = sys.exc_info()

    with patch.object(IPython, "start_ipython") as repl:
        ipython_excepthook(exc_type, value, tb)
        repl.assert_called_once()

def test_gui(monkeypatch):
    monkeypatch.setattr("damnit.cli.find_proposal", lambda p: "/tmp")

    # Check passing neither a proposal number or directory
    with patch("damnit.gui.main_window.run_app") as run_app:
        main(["gui"])
        run_app.assert_called_with(None, software_opengl=False, connect_to_kafka=ANY)

    # Check passing a proposal number
    with patch("damnit.gui.main_window.run_app") as run_app:
        main(["gui", "1234"])
        run_app.assert_called_with(Path("/tmp/usr/Shared/amore"), software_opengl=False, connect_to_kafka=ANY)

    # Check passing a directory
    with patch("damnit.gui.main_window.run_app") as run_app:
        main(["gui", "/tmp"])
        run_app.assert_called_with(Path("/tmp"), software_opengl=False, connect_to_kafka=ANY)

    # Check invalid argument
    with patch("damnit.gui.main_window.run_app") as run_app:
        with pytest.raises(SystemExit):
            main(["gui", "/nope"])
        run_app.assert_not_called()

def test_listen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pkg = "damnit.backend"

    with patch(f"{pkg}.listener.listen") as listen:
        main(["listen"])
        listen.assert_called_once()

    with patch(f"{pkg}.test_listener.listen") as listen:
        main(["listen", "--test"])
        listen.assert_called_once()

    # Should fail without an existing database
    with (patch(f"{pkg}.initialize_and_start_backend") as initialize_and_start_backend,
          pytest.raises(SystemExit)):
        main(["listen", "--daemonize"])
        initialize_and_start_backend.assert_not_called()

    # Should work with an existing database
    (tmp_path / "runs.sqlite").touch()
    with patch(f"{pkg}.initialize_and_start_backend") as initialize_and_start_backend:
        main(["listen", "--daemonize"])
        initialize_and_start_backend.assert_called_once()

    # Can't pass both --test and --daemonize
    with pytest.raises(SystemExit):
        main(["listen", "--daemonize", "--test"])

def test_reprocess(mock_db_with_data, monkeypatch):
    db_dir, db = mock_db_with_data
    db.metameta["proposal"] = 1234
    monkeypatch.chdir(db_dir)

    # Create a proposal directory with raw/
    raw_dir = db_dir / "mock_proposal" / "raw"
    raw_dir.mkdir(parents=True)

    # patch find_proposal() to return the mock proposal directory we created.
    monkeypatch.setattr(
        "damnit.backend.extraction_control.find_proposal", lambda p: raw_dir.parent
    )

    def mock_sbatch():
        return MockCommand.fixed_output("sbatch", "9876; maxwell")

    # Since none of the runs in the database exist on disk, we should skip all
    # of them (i.e. not call sbatch).
    with mock_sbatch() as sbatch:
        main(["reprocess", "all"])

    assert sbatch.get_calls() == []

    # Create raw/ directories for 10 runs
    for i in range(10):
        (raw_dir / f"r{i:04}").mkdir()

    # Reprocessing run 1 should call sbatch, because the run directory exists
    with mock_sbatch() as sbatch:
        main(["reprocess", "1"])

    sbatch.assert_called()

    # No directory for run 10, so this shouldn't try to submit a job
    with mock_sbatch() as sbatch:
        main(["reprocess", "10"])

    assert sbatch.get_calls() == []


def test_cli_command_name(capsys, monkeypatch):
    """Test that the CLI works with both 'damnit' and 'amore-proto' names,
    and shows deprecation warning for 'amore-proto'."""
    with patch('sys.argv', ['damnit', '--help']):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert 'Warning:' not in captured.err  # No warning for 'damnit'
        assert 'usage:' in captured.out  # Help text is shown

    with patch('sys.argv', ['amore-proto', '--help']):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "Warning: 'amore-proto' has been renamed to 'damnit'" in captured.err  # Shows deprecation
        assert 'usage:' in captured.out  # Help text is shown
