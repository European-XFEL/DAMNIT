import configparser
import graphlib
import io
import logging
import os
import signal
import stat
import subprocess
import textwrap
from unittest.mock import MagicMock, patch

import extra_data as ed
import h5py
import numpy as np
import plotly.express as px
import pytest
import requests
import xarray as xr
import yaml
from PIL import Image

from damnit.backend import backend_is_running, initialize_and_start_backend
from damnit.backend.db import DamnitDB
from damnit.backend.extract_data import Extractor, add_to_db
from damnit.backend.supervisord import wait_until, write_supervisord_conf
from damnit.context import (ContextFile, ContextFileErrors, PNGData, Results,
                            RunData, get_proposal_path)
from damnit.ctxsupport.ctxrunner import THUMBNAIL_SIZE
from damnit.gui.main_window import MainWindow

from .helpers import mkcontext, reduced_data_from_dict


def kill_pid(pid):
    """
    Send SIGTERM to a process.
    """
    print(f"Killing {pid}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"PID {pid} doesn't exist")


def test_context_file(mock_ctx, tmp_path):
    code = """
    from damnit.context import Variable

    @Variable(title="Foo")
    def foo(run):
        return 42
    """
    # Test creating from a string
    ctx = mkcontext(code)
    assert len(ctx.vars) == 1

    # Test creating from a file
    ctx_path = tmp_path / 'context.py'
    ctx_path.write_text(textwrap.dedent(code))
    ctx = ContextFile.from_py_file(ctx_path)

    assert len(ctx.vars) == 1

    duplicate_titles_code = """
    from damnit.context import Variable

    @Variable(title="Foo")
    def foo(run): return 42

    @Variable(title="Foo")
    def bar(run): return 43
    """

    ctx = mkcontext(duplicate_titles_code)
    with pytest.raises(ContextFileErrors):
        ctx.check()

    # Helper lambda to get the names of the direct dependencies of a variable
    var_deps = lambda name: set(mock_ctx.vars[name].arg_dependencies().values())
    # Helper lambda to get the names of all dependencies of a variable
    all_var_deps = lambda name: mock_ctx.all_dependencies(mock_ctx.vars[name])

    # Check that each variable has the right dependencies
    assert { "scalar1" } == var_deps("scalar2")
    assert { "scalar1", "scalar2" } == var_deps("array")
    assert { "array", "timestamp" } == var_deps("meta_array")

    # Check that the ordering is correct for execution
    assert mock_ctx.ordered_vars() == ("scalar1", "empty_string", "timestamp", "string", "plotly_mc_plotface", "scalar2", "array", "meta_array")

    # Check that we can retrieve direct and indirect dependencies
    assert set() == all_var_deps("scalar1")
    assert { "scalar1" } == all_var_deps("scalar2")
    assert { "scalar1", "scalar2" } == all_var_deps("array")
    assert { "array", "timestamp", "scalar1", "scalar2" } == all_var_deps("meta_array")

    # Create a context file with a cycle
    cycle_code = """
    from damnit.context import Variable

    @Variable(title="foo")
    def foo(run, bar: "var#bar"):
        return bar

    @Variable(title="bar")
    def bar(run, foo: "var#foo"):
        return foo
    """

    # Creating a context from this should fail
    with pytest.raises(graphlib.CycleError):
        mkcontext(cycle_code)

    # Context file with raw variable's depending on proc variable's
    bad_dep_code = """
    from damnit.context import Variable

    @Variable(title="foo", data="proc")
    def foo(run):
        return 42

    @Variable(title="bar", data="raw")
    def bar(run, foo: "var#foo"):
        return foo
    """

    ctx = mkcontext(bad_dep_code)
    with pytest.raises(ContextFileErrors):
        ctx.check()

    var_promotion_code = """
    from damnit.context import Variable

    @Variable(title="foo", data="proc")
    def foo(run):
        return 42

    @Variable(title="bar")
    def bar(run, foo: "var#foo"):
        return foo
    """
    # This should not raise an exception
    var_promotion_ctx = mkcontext(var_promotion_code)

    # `bar` should automatically be promoted to use proc data because it depends
    # on a proc variable.
    assert var_promotion_ctx.vars["bar"].data == RunData.PROC

    # Test depending on mymdc fields
    good_mymdc_code = """
    from damnit.context import Variable

    @Variable(title="foo")
    def foo(run, sample: "mymdc#sample_name", run_type: "mymdc#run_type"):
        return 42
    """
    # This should not raise an exception
    mkcontext(good_mymdc_code).check()

    bad_mymdc_code = """
    from damnit.context import Variable

    @Variable(title="foo")
    def foo(run, quux: "mymdc#quux"):
        return 42
    """
    ctx = mkcontext(bad_mymdc_code)

    # This should raise an exception because it's using an unsupported mymdc
    # field.
    with pytest.raises(ContextFileErrors):
        ctx.check()

def run_ctx_helper(context, run, run_number, proposal, caplog, input_vars=None):
    # Track all error messages during creation. This is necessary because a
    # variable that throws an error will be logged by Results, the exception
    # will not bubble up.
    with caplog.at_level(logging.ERROR):
        results = context.execute(run, run_number, proposal, input_vars or {})

    # Check that there were no errors
    assert caplog.records == []
    return results

def test_results(mock_ctx, mock_run, caplog, tmp_path):
    run_number = 1000
    proposal = 1234
    results_create = lambda ctx: ctx.execute(mock_run, run_number, proposal, {})

    # Simple test
    results = run_ctx_helper(mock_ctx, mock_run, run_number, proposal, caplog)
    assert set(mock_ctx.ordered_vars()) <= results.data.keys()

    # Check that the summary of a DataArray is a single number
    assert isinstance(results.data["meta_array"], xr.DataArray)
    assert results.reduced["meta_array"].ndim == 0

    # Check the result values
    assert results.data["scalar1"] == 42
    assert results.data["scalar2"] == 3.14
    assert results.data["empty_string"] == ""
    np.testing.assert_equal(results.data["array"], [42, 3.14])
    np.testing.assert_equal(results.data["meta_array"].data, [run_number, proposal])
    assert results.data["string"] == str(get_proposal_path(mock_run))
    assert results.data['plotly_mc_plotface'] == px.bar(x=['a', 'b', 'c'], y=[1, 3, 2])

    # Test behaviour with dependencies throwing exceptions
    raising_code = """
    from damnit.context import Variable

    @Variable(title="Foo")
    def foo(run):
        raise RuntimeError()

    @Variable(title="bar")
    def bar(run, foo: "var#foo"):
        return foo
    """
    raising_ctx = mkcontext(raising_code)

    with caplog.at_level(logging.WARNING):
        results = results_create(raising_ctx)

        # An error about foo and warning about bar should have been logged
        assert len(caplog.records) == 2
        assert "in foo" in caplog.text
        assert "Skipping bar" in caplog.text

        # No variables should have been computed, except for the default 'start_time'
        assert tuple(results.data.keys()) == ("start_time",)

    caplog.clear()

    # Same thing, but with variables returning None
    return_none_code = """
    from damnit.context import Variable

    @Variable(title="Foo")
    def foo(run):
        return None

    @Variable(title="bar")
    def bar(run, foo: "var#foo"):
        return foo
    """
    return_none_ctx = mkcontext(return_none_code)

    with caplog.at_level(logging.WARNING):
        results = results_create(return_none_ctx)

        # One warning about foo should have been logged
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"

        # There should be no computed variables since we treat None as a missing dependency
        assert tuple(results.data.keys()) == ("start_time",)

    default_value_code = """
    from damnit_ctx import Variable

    @Variable(title="foo")
    def foo(run): return None

    @Variable(title="bar")
    def bar(run, foo: "var#foo"=1): return 41 + foo
    """
    default_value_ctx = mkcontext(default_value_code)
    results = results_create(default_value_ctx)
    assert results.reduced["bar"].item() == 42

    # Test that the backend completely updates all datasets belonging to a
    # variable during reprocessing. e.g. if it had a trainId dataset but now
    # doesn't, the trainId dataset should be deleted from the HDF5 file.
    results_hdf5_path = tmp_path / "results.hdf5"
    with_coords_code = """
    import xarray as xr
    from damnit.context import Variable

    @Variable(title="Foo")
    def foo(run):
        return xr.DataArray(
            [1, 2, 3],
            coords={"trainId": [100, 101, 102]},
            name="foo/manchu"
        )
    """
    with_coords_ctx = mkcontext(with_coords_code)
    results = results_create(with_coords_ctx)
    results.save_hdf5(results_hdf5_path)

    # This time there should be a trainId dataset saved
    with h5py.File(results_hdf5_path) as f:
        assert "foo/trainId" in f

    data_array = xr.load_dataarray(results_hdf5_path, group="foo", engine="h5netcdf")
    assert data_array.name == 'foo_manchu'

    without_coords_code = """
    import xarray as xr
    from damnit.context import Variable

    @Variable(title="Foo")
    def foo(run): return xr.DataArray([1, 2, 3])
    """
    without_coords_ctx = mkcontext(without_coords_code)
    results = results_create(without_coords_ctx)
    results.save_hdf5(results_hdf5_path)

    # But now it should be deleted from the file
    with h5py.File(results_hdf5_path) as f:
        assert "foo/trainId" not in f

    figure_code = """
    import numpy as np
    from damnit_ctx import Variable
    from matplotlib import pyplot as plt

    @Variable("2D array")
    def twodarray(run):
        return np.random.rand(1000, 1000)

    @Variable(title="Axes")
    def axes(run):
        _, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [4, 3, 2, 1])

        return ax

    @Variable(title="Figure")
    def figure(run):
        fig = plt.figure()
        plt.plot([1, 2, 3, 4], [4, 3, 2, 1])

        return fig
    """
    figure_ctx = mkcontext(figure_code)
    results = results_create(figure_ctx)
    assert isinstance(results.reduced["figure"], PNGData)
    assert isinstance(results.reduced["axes"], PNGData)

    results_hdf5_path.unlink()
    results.save_hdf5(results_hdf5_path)
    with h5py.File(results_hdf5_path) as f:
        # The plots should be saved as 3D RGBA arrays
        assert f["figure/data"].ndim == 3
        assert f["axes/data"].ndim == 3

        # Test that the summaries are the right size
        twodarray_png = Image.open(io.BytesIO(f[".reduced/twodarray"][()]))
        assert np.asarray(twodarray_png).shape == (THUMBNAIL_SIZE, THUMBNAIL_SIZE, 4)

        figure_png = Image.open(io.BytesIO(f[".reduced/figure"][()]))
        assert max(np.asarray(figure_png).shape) == THUMBNAIL_SIZE

    # Test returning xarray.Datasets
    dataset_code = """
    from damnit_ctx import Variable
    import xarray as xr

    @Variable(title="Dataset")
    def dataset(run):
        return xr.Dataset(data_vars={
            "foo": xr.DataArray([1, 2, 3]) ,
            "bar/baz": xr.DataArray([4, 5, 6]),
        })
    """
    dataset_ctx = mkcontext(dataset_code)
    results = results_create(dataset_ctx)
    results.save_hdf5(results_hdf5_path)

    dataset = xr.load_dataset(results_hdf5_path, group="dataset", engine="h5netcdf")
    assert "foo" in dataset
    assert "bar_baz" in dataset
    with h5py.File(results_hdf5_path) as f:
        assert f[".reduced/dataset"].asstr()[()].startswith("Dataset")

    # Test getting mymdc fields
    mymdc_code = """
    from damnit_ctx import Variable

    @Variable(title="Sample")
    def sample(run, x: "mymdc#sample_name"):
        return x

    @Variable(title="Run type")
    def run_type(run, x: "mymdc#run_type"):
        return x
    """
    mymdc_ctx = mkcontext(mymdc_code)

    # Create some mock credentials and set the mock_run files to appear to be
    # under `tmp_path`.
    (tmp_path / "usr").mkdir()
    with open(tmp_path / "usr/mymdc-credentials.yml", "w") as f:
        yaml.dump({
            "token": "foo",
            "server": "https://out.xfel.eu/metadata"
        }, f)
    mock_run.files = [MagicMock(filename=tmp_path / "raw/r0001/RAW-R0004-DA03-S00000.h5")]

    # Helper function to mock requests.get() for different endpoints
    def mock_get(url, headers, timeout):
        assert headers["X-API-key"] == "foo"

        if "proposals/by_number" in url:
            result = dict(runs=[dict(sample_id=1, experiment_id=1)])
        elif "samples" in url:
            result = dict(name="mithril")
        elif "experiments" in url:
            result = dict(name="alchemy")

        response = MagicMock()
        response.json.return_value = result
        return response

    # Execute the context file and check the results
    with patch.object(requests, "get", side_effect=mock_get), \
         patch.object(ed.read_machinery, "find_proposal", return_value=tmp_path):
        results = results_create(mymdc_ctx)
    assert results.data["sample"] == "mithril"
    assert results.data["run_type"] == "alchemy"

def test_results_bad_obj(mock_run, tmp_path):
    # Test returning an object we can't save in HDF5
    bad_obj_code = """
    from damnit_ctx import Variable

    @Variable()
    def good(run):
        return 7

    @Variable()
    def bad(run):
        return object()
    """
    bad_obj_ctx = mkcontext(bad_obj_code)
    results = bad_obj_ctx.execute(mock_run, 1000, 123, {})
    results_hdf5_path = tmp_path / 'results.h5'
    results.save_hdf5(results_hdf5_path)
    with h5py.File(results_hdf5_path) as f:
        assert set(f) == {".reduced", "good", "start_time"}
        assert set(f[".reduced"]) == {"good", "start_time"}

@pytest.mark.skip(reason="Depending on user variables is currently disabled")
def test_results_with_user_vars(mock_ctx_user, mock_user_vars, mock_run, caplog):

    proposal = 1234
    run_number = 1000

    user_var_values = {
        "user_integer": 12,
        "user_number": 10.2,
        "user_boolean": True,
        "user_string": "foo"
    }

    results = run_ctx_helper(mock_ctx_user, mock_run, run_number, proposal, caplog, input_vars=user_var_values)

    # Tests if computations that depends on user variable return the correct results
    assert results.data["dep_integer"] == user_var_values["user_integer"] + 1
    assert results.data["dep_number"] == user_var_values["user_number"]
    assert results.data["dep_boolean"] == False
    assert results.data["dep_string"] == user_var_values["user_string"] * 2

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
        "string": "foo",
        "scalar": 42,
        "float": 31.7,
        "image": b'\x89PNG\r\n\x1a\n...'  # Not a valid PNG, but good enough for this
    }

    add_to_db(reduced_data_from_dict(reduced_data), db, 1234, 42)

    cursor = db.conn.execute("SELECT * FROM runs")
    row = cursor.fetchone()

    assert row["string"] == reduced_data["string"]
    assert row["scalar"] == reduced_data["scalar"]
    assert row["float"] == reduced_data["float"]
    assert row["image"] == reduced_data["image"]

def test_extractor(mock_ctx, mock_db, mock_run, monkeypatch):
    # Change to the DB directory
    db_dir, db = mock_db
    db.metameta["proposal"] = 1234
    monkeypatch.chdir(db_dir)
    pkg = "damnit.backend.extract_data"

    # Write context file
    no_summary_var = """
    @Variable(title="Array")
    def array(run):
        return np.random.rand(2, 2, 2, 2)

    @Variable(title="Scalar", cluster=True)
    def slurm_scalar(run):
        return 42
    """
    mock_code = mock_ctx.code + "\n\n" + textwrap.dedent(no_summary_var)
    mock_ctx = mkcontext(mock_code)

    ctx_path = db_dir / "context.py"
    ctx_path.write_text(mock_ctx.code)

    out_path = db_dir / "extracted_data" / "p1234_r42.h5"
    out_path.parent.mkdir(exist_ok=True)

    # Create Extractor with a mocked KafkaProducer
    with patch(f"{pkg}.KafkaProducer") as _:
        extractor = Extractor()

    # Test regular variables and slurm variables are executed
    reduced_data = reduced_data_from_dict({ "n": 53 })
    with patch(f"{pkg}.extract_in_subprocess", return_value=reduced_data) as extract_in_subprocess, \
         patch(f"{pkg}.subprocess.run") as subprocess_run:
        extractor.extract_and_ingest(1234, 42, cluster=False,
                                     run_data=RunData.ALL)
        extract_in_subprocess.assert_called_once()
        extractor.kafka_prd.send.assert_called()
        subprocess_run.assert_called_once()

    # This works because we loaded damnit.context above
    from ctxrunner import main

    # Process run
    with patch("ctxrunner.extra_data.open_run", return_value=mock_run):
        main(['exec', '1234', '42', 'raw', '--save', str(out_path)])

    # Check that a file was created
    assert out_path.is_file()

    with h5py.File(out_path) as f:
        assert f[".reduced"]["array"].asstr()[()] == "float64: (2, 2, 2, 2)"
        assert f["array"]["data"].shape == (2, 2, 2, 2)

    # Helper function to raise an exception when proc data isn't available, like
    # open_run(data="proc") would.
    def mock_open_run(*_, data=None):
        if data == "proc":
            raise FileNotFoundError("foo.h5")
        else:
            return mock_run

    # Reprocess with `data='all'`, but as if there is no proc data
    with patch("ctxrunner.extra_data.open_run", side_effect=mock_open_run):
        main(['exec', '1234', '42', 'all', '--save', str(out_path)])

    # Check that `meta_array` wasn't processed, since it requires proc data
    with h5py.File(out_path) as f:
        assert "meta_array" not in f

    # Reprocess with proc data
    with patch("ctxrunner.extra_data.open_run", return_value=mock_run):
        main(['exec', '1234', '42', 'all', '--save', str(out_path)])

    # Now `meta_array` should have been processed
    with h5py.File(out_path) as f:
        assert "meta_array" in f

    # Runs shouldn't be opened with open_run() when --mock is passed
    with patch("ctxrunner.extra_data.open_run") as open_run:
        main(["exec", "1234", "42", "all", "--mock"])
        open_run.assert_not_called()

    # When only proc variables are reprocessed, the run should still be opened
    # with `data='all'` so that raw data is available. Note that we patch
    # Results too so that the test doesn't throw exceptions.
    with patch("ctxrunner.extra_data.open_run", return_value=mock_run) as open_run, \
         patch("ctxrunner.Results"):
        main(["exec", "1234", "42", "proc"])

        open_run.assert_called_with(1234, 42, data="all")

def test_custom_environment(mock_db, virtualenv, monkeypatch, qtbot):
    db_dir, db = mock_db
    monkeypatch.chdir(db_dir)

    ctxrunner_deps = ["extra_data", "matplotlib", "plotly", "pyyaml", "requests"]

    # Install dependencies for ctxrunner and a light-weight package (sfollow)
    # that isn't in our current environment.
    virtualenv.install_package(" ".join([*ctxrunner_deps, "sfollow"]),
                               installer="pip install")

    # Write a context file that requires the new package
    new_env_code = f"""
    from pathlib import Path

    import sfollow
    from damnit_ctx import Variable

    # Ensure that the context file is *always* evaluated in the database
    # directory. This is necessary to ensure that things like relative imports
    # and janky sys.path shenanigans work.
    assert str(Path.cwd()) == {str(db_dir)!r}

    @Variable(title="Foo")
    def foo(run):
        return 42
    """
    (db_dir / "context.py").write_text(textwrap.dedent(new_env_code))

    pkg = "damnit.backend.extract_data"

    with patch(f"{pkg}.KafkaProducer"), pytest.raises(ImportError):
        Extractor()

    # Set the context_python field in the database
    db.metameta["context_python"] = str(virtualenv.python)

    with patch(f"{pkg}.KafkaProducer"):
        Extractor().extract_and_ingest(1234, 42, mock=True)

    with h5py.File(db_dir / "extracted_data" / "p1234_r42.h5") as f:
        assert f["foo/data"][()] == 42

    # Make sure that the GUI evaluates the context file correctly (which it does
    # upon opening a database directory).
    win = MainWindow(db_dir, False)

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

    pkg = "damnit.backend.supervisord"
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
    db = DamnitDB(db_path)
    assert db.metameta["proposal"] == 1234

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
