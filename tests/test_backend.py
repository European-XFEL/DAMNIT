import os
import stat
import time
import pickle
import signal
import logging
import graphlib
import textwrap
import tempfile
import subprocess
import configparser
from pathlib import Path
from numbers import Number
from unittest.mock import patch

import pytest
import numpy as np
import xarray as xr

from amore_mid_prototype.util import wait_until
from amore_mid_prototype.context import ContextFile, Results, RunData, get_proposal_path
from amore_mid_prototype.backend.db import open_db, get_meta
from amore_mid_prototype.backend import initialize_and_start_backend, backend_is_running
from amore_mid_prototype.backend.extract_data import add_to_db, Extractor
from amore_mid_prototype.backend.supervisord import write_supervisord_conf


def kill_pid(pid):
    """
    Send SIGTERM to a process.
    """
    print(f"Killing {pid}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"PID {pid} doesn't exist")


def test_context_file(mock_ctx):
    code = """
    from amore_mid_prototype.context import Variable

    @Variable(title="Foo")
    def foo(run):
        return 42
    """
    code = textwrap.dedent(code)

    # Test creating from a string
    ctx = ContextFile.from_str(code)
    assert len(ctx.vars) == 1

    # Test creating from a file
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(code.encode())
        tmp.flush()

        ctx = ContextFile.from_py_file(Path(tmp.name))

    assert len(ctx.vars) == 1

    duplicate_titles_code = """
    from amore_mid_prototype.context import Variable

    @Variable(title="Foo")
    def foo(run): return 42

    @Variable(title="Foo")
    def bar(run): return 43
    """

    with pytest.raises(RuntimeError):
        ContextFile.from_str(textwrap.dedent(duplicate_titles_code))

    # Helper lambda to get the names of the direct dependencies of a variable
    var_deps = lambda name: set(mock_ctx.vars[name].arg_dependencies().values())
    # Helper lambda to get the names of all dependencies of a variable
    all_var_deps = lambda name: mock_ctx.all_dependencies(mock_ctx.vars[name])

    # Check that each variable has the right dependencies
    assert { "scalar1" } == var_deps("scalar2")
    assert { "scalar1", "scalar2" } == var_deps("array")
    assert { "array", "timestamp" } == var_deps("meta_array")

    # Check that the ordering is correct for execution
    assert mock_ctx.ordered_vars() == ("scalar1", "empty_string", "timestamp", "string", "scalar2", "array", "meta_array")

    # Check that we can retrieve direct and indirect dependencies
    assert set() == all_var_deps("scalar1")
    assert { "scalar1" } == all_var_deps("scalar2")
    assert { "scalar1", "scalar2" } == all_var_deps("array")
    assert { "array", "timestamp", "scalar1", "scalar2" } == all_var_deps("meta_array")

    # Create a context file with a cycle
    cycle_code = """
    from amore_mid_prototype.context import Variable

    @Variable(title="foo")
    def foo(run, bar: "var#bar"):
        return bar

    @Variable(title="bar")
    def bar(run, foo: "var#foo"):
        return foo
    """

    # Creating a context from this should fail
    with pytest.raises(graphlib.CycleError):
        ContextFile.from_str(textwrap.dedent(cycle_code))

    # Context file with raw variable's depending on proc variable's
    bad_dep_code = """
    from amore_mid_prototype.context import Variable

    @Variable(title="foo", data="proc")
    def foo(run):
        return 42

    @Variable(title="bar", data="raw")
    def bar(run, foo: "var#foo"):
        return foo
    """

    with pytest.raises(RuntimeError):
        ContextFile.from_str(textwrap.dedent(bad_dep_code))

    var_promotion_code = """
    from amore_mid_prototype.context import Variable

    @Variable(title="foo", data="proc")
    def foo(run):
        return 42

    @Variable(title="bar")
    def bar(run, foo: "var#foo"):
        return foo
    """
    # This should not raise an exception
    var_promotion_ctx = ContextFile.from_str(textwrap.dedent(var_promotion_code))

    # `bar` should automatically be promoted to use proc data because it depends
    # on a proc variable.
    assert var_promotion_ctx.vars["bar"].data == RunData.PROC

def run_ctx_helper(context, run, run_number, proposal, caplog):
    # Track all error messages during creation. This is necessary because a
    # variable that throws an error will be logged by Results, the exception
    # will not bubble up.
    with caplog.at_level(logging.ERROR):
        results = Results.create(context, run, run_number, proposal)

    # Check that there were no errors
    assert caplog.records == []
    return results


def test_results(mock_ctx, mock_run, caplog):
    run_number = 1000
    proposal = 1234

    # Simple test
    results = run_ctx_helper(mock_ctx, mock_run, run_number, proposal, caplog)
    assert set(mock_ctx.ordered_vars()) <= results.data.keys()

    # Check that the summary of a DataArray is a single number
    assert isinstance(results.data["meta_array"], xr.DataArray)
    assert isinstance(results.reduced["meta_array"], Number)

    # Check the result values
    assert results.data["scalar1"] == 42
    assert results.data["scalar2"] == 3.14
    assert results.data["empty_string"] == ""
    np.testing.assert_equal(results.data["array"], [42, 3.14])
    np.testing.assert_equal(results.data["meta_array"].data, [run_number, proposal])
    assert results.data["string"] == str(get_proposal_path(mock_run))

    # Test behaviour with dependencies throwing exceptions
    raising_code = """
    from amore_mid_prototype.context import Variable

    @Variable(title="Foo")
    def foo(run):
        raise RuntimeError()

    @Variable(title="bar")
    def bar(run, foo: "var#foo"):
        return foo
    """
    raising_ctx = ContextFile.from_str(textwrap.dedent(raising_code))

    with caplog.at_level(logging.WARNING):
        results = Results.create(raising_ctx, mock_run, run_number, proposal)

        # An error about foo and warning about bar should have been logged
        assert len(caplog.records) == 2
        assert "in foo" in caplog.text
        assert "Skipping bar" in caplog.text

        # No variables should have been computed, except for the default 'start_time'
        assert tuple(results.data.keys()) == ("start_time",)

    caplog.clear()

    # Same thing, but with variables returning None
    return_none_code = """
    from amore_mid_prototype.context import Variable

    @Variable(title="Foo")
    def foo(run):
        return None

    @Variable(title="bar")
    def bar(run, foo: "var#foo"):
        return foo
    """
    return_none_ctx = ContextFile.from_str(textwrap.dedent(return_none_code))

    with caplog.at_level(logging.WARNING):
        results = Results.create(return_none_ctx, mock_run, run_number, proposal)

        # One warning about foo should have been logged
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"

        # There should be no computed variables since we treat None as a missing dependency
        assert tuple(results.data.keys()) == ("start_time",)

def test_filtering(mock_ctx, mock_run, caplog):
    run_number = 1000
    proposal = 1234

    # First run with raw data and a filter so only one Variable is executed
    ctx = mock_ctx.filter(run_data=RunData.RAW, name_matches=["string"])
    assert set(ctx.vars) == { "string" }
    results = run_ctx_helper(ctx, mock_run, run_number, proposal, caplog)
    assert set(results.data) == { "string", "start_time" }

    # Now select a Variable with dependencies
    ctx = mock_ctx.filter(run_data=RunData.RAW, name_matches=["scalar2", "timestamp"])
    assert set(ctx.vars) == { "scalar1", "scalar2", "timestamp" }
    results = run_ctx_helper(ctx, mock_run, run_number, proposal, caplog)
    assert set(results.data) == { "scalar1", "scalar2", "timestamp", "start_time" }
    ts = results.data["timestamp"]

    # Requesting a Variable that requires proc data with only raw data
    # should not execute anything.
    ctx = mock_ctx.filter(run_data=RunData.RAW, name_matches=["meta_array"])
    assert set(ctx.vars) == set()
    results = run_ctx_helper(ctx, mock_run, run_number, proposal, caplog)
    assert set(results.data) == {"start_time"}

    # But with proc data all dependencies should be executed
    ctx = mock_ctx.filter(run_data=RunData.PROC, name_matches=["meta_array"])
    assert set(ctx.vars) == { "scalar1", "scalar2", "timestamp", "array", "meta_array" }
    results = run_ctx_helper(ctx, mock_run, run_number, proposal, caplog)
    assert set(results.data) == { "scalar1", "scalar2", "timestamp", "array", "meta_array", "start_time" }
    assert results.data["timestamp"] > ts

def test_add_to_db(mock_db):
    db_dir, db = mock_db

    reduced_data = {
        "none": None,
        "string": "foo",
        "scalar": 42,
        "np_scalar": np.float32(10),
        "zero_dim_array": np.asarray(42),
        "image": np.random.rand(10, 10)
    }

    add_to_db(reduced_data, db, 1234, 42)

    cursor = db.execute("SELECT * FROM runs")
    row = cursor.fetchone()

    assert row["string"] == reduced_data["string"]
    assert row["scalar"] == reduced_data["scalar"]
    assert row["np_scalar"] == reduced_data["np_scalar"].item()
    assert row["zero_dim_array"] == reduced_data["zero_dim_array"].item()
    np.testing.assert_array_equal(pickle.loads(row["image"]), reduced_data["image"])
    assert row["none"] == reduced_data["none"]

def test_extractor(mock_ctx, mock_db, mock_run, monkeypatch):
    # Change to the DB directory
    db_dir, db = mock_db
    monkeypatch.chdir(db_dir)
    pkg = "ctxrunner"

    # Write context file
    ctx_path = db_dir / "context.py"
    ctx_path.write_text(mock_ctx.code)

    out_path = db_dir / "extracted_data" / "p1234_r42.h5"
    out_path.parent.mkdir(exist_ok=True)

    # This works because we loaded amore_mid_prototype.context above
    from ctxrunner import main

    # Process run
    with patch(f"{pkg}.extra_data.open_run", return_value=mock_run):
        main(['1234', '42', 'raw', '--save', str(out_path)])

    # Check that a file was created
    assert out_path.is_file()

def test_initialize_and_start_backend(tmp_path, bound_port, request):
    db_dir = tmp_path / "foo"
    supervisord_config_path = db_dir / "supervisord.conf"
    good_file_mode = "-rw-rw-rw-"
    path_filemode = lambda p: stat.filemode(p.stat().st_mode)

    def mock_write_supervisord_conf(root_path, port=None):
        write_supervisord_conf(root_path)

        # This is a dummy command for testing. The script will create a
        # subprocess that creates a file named 'started' before it sleeps for
        # 10s. A handler is configured to write a file named 'stopped' upon a
        # SIGTERM before exiting. We need to test that child processes are
        # killed because the listener creates subprocesses to process each run.
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

        dummy_command_path = root_path / "dummy.py"
        dummy_command_path.write_text(textwrap.dedent(subprocess_code))
        config = configparser.ConfigParser()
        config.read(supervisord_config_path)

        # Chage the command, and optionally the port
        config["program:damnit"]["command"] = "python dummy.py"
        if port is not None:
            config["inet_http_server"]["port"] = str(bound_port)

        with open(supervisord_config_path, "w") as f:
            config.write(f)

    pkg = "amore_mid_prototype.backend.supervisord"
    with patch(f"{pkg}.write_supervisord_conf",
               side_effect=mock_write_supervisord_conf):
        assert initialize_and_start_backend(db_dir, 1234)

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
    assert supervisord_config_path.is_file()
    assert path_filemode(supervisord_config_path) == good_file_mode

    # We should have a log file and PID file. They aren't created immediately so
    # we wait a bit for it.
    pid_path = db_dir / "supervisord.pid"
    log_path = db_dir / "supervisord.log"
    wait_until(lambda: pid_path.is_file() and log_path.is_file())
    pid = int(pid_path.read_text())

    # Set a finalizer to kill supervisord at the end
    request.addfinalizer(lambda: kill_pid(pid))

    assert path_filemode(pid_path) == good_file_mode
    assert path_filemode(log_path) == good_file_mode

    # Check that it's running
    assert backend_is_running(db_dir)

    wait_until(lambda: (db_dir / "started").is_file(), timeout=1)

    # Stop the program
    supervisorctl = ["supervisorctl", "-c", str(supervisord_config_path)]
    subprocess.run([*supervisorctl, "stop", "damnit"]).check_returncode()

    # Check that the subprocess was also killed
    wait_until(lambda: (db_dir / "stopped").is_file())
    assert not backend_is_running(db_dir)

    # Try starting it again. This time we don't pass the proposal number, it
    # should be picked up from the existing database.
    with patch(f"{pkg}.write_supervisord_conf",
               side_effect=mock_write_supervisord_conf):
        assert initialize_and_start_backend(db_dir)

    assert backend_is_running(db_dir)

    # Now kill supervisord
    kill_pid(pid)
    assert not backend_is_running(db_dir)

    # Change the config to use the bound port
    mock_write_supervisord_conf(db_dir, port=bound_port)

    # And try starting it again
    with patch(f"{pkg}.write_supervisord_conf",
               side_effect=mock_write_supervisord_conf):
        assert initialize_and_start_backend(db_dir)

    wait_until(lambda: pid_path.is_file() and log_path.is_file())
    pid = int(pid_path.read_text())
    request.addfinalizer(lambda: kill_pid(pid))

    # Check that the backend is running
    assert backend_is_running(db_dir)

    # Trying to start it again should do nothing
    with patch(f"{pkg}.write_supervisord_conf",
               side_effect=mock_write_supervisord_conf):
        assert initialize_and_start_backend(db_dir)

    assert backend_is_running(db_dir)
