import configparser
import graphlib
import io
import json
import logging
import os
import signal
import stat
import subprocess
import textwrap
from time import sleep, time
from uuid import uuid4
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
from testpath import MockCommand

from damnit.backend import backend_is_running, initialize_and_start_backend
from damnit.backend.db import DamnitDB
from damnit.backend.extract_data import Extractor, RunExtractor, add_to_db, load_reduced_data
from damnit.backend.extraction_control import ExtractionJobTracker
from damnit.backend.listener import (MAX_CONCURRENT_THREADS, EventProcessor,
                                     local_extraction_threads)
from damnit.backend.supervisord import wait_until, write_supervisord_conf
from damnit.context import (ContextFile, ContextFileErrors, PNGData, RunData,
                            get_proposal_path)
from damnit.ctxsupport.ctxrunner import THUMBNAIL_SIZE, add_to_h5_file
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


def check_rgba(dset):
    assert dset.ndim == 3
    assert dset.shape[-1] == 4

def check_png(dset):
    assert dset.ndim == 1
    assert dset.dtype == np.dtype(np.uint8)
    assert dset[:8].tobytes() == b'\x89PNG\r\n\x1a\n'

def test_add_to_h5_file(tmp_path):
    path = tmp_path / "foo.h5"
    good_file_mode = "-rw-rw-rw-"
    path_filemode = lambda p: stat.filemode(p.stat().st_mode)

    with pytest.raises(RuntimeError):
        with add_to_h5_file(path) as f:
            f["foo"] = 1
            raise RuntimeError("foo")

    # When an exception is raised the file permissions should still be set
    assert path.is_file()
    assert path_filemode(path) == good_file_mode

    # Test that the file is opened in append mode, such that previously saved
    # data is still present.
    with add_to_h5_file(path) as f:
        assert f["foo"][()] == 1

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
    assert mock_ctx.ordered_vars() == (
        # First everything without dependencies, in definition order
        "scalar1", "empty_string", "timestamp", "string", "plotly_mc_plotface",
        "results", "image", "array_preview", "plotly_preview",
        # Then the dependencies
        "scalar2", "array", "meta_array"
    )

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

    @Variable()
    def baz(run, bar: "var#bar"):
        return bar * 2

    @Variable(title="foo", data="proc", cluster=True)
    def foo(run):
        return 42

    @Variable(title="bar")
    def bar(run, foo: "var#foo"):
        return foo
    """
    # This should not raise an exception
    var_promotion_ctx = mkcontext(var_promotion_code)

    # `bar` & `baz` should be promoted to use proc data & run on a dedicated
    # node because they depend on foo.
    assert var_promotion_ctx.vars["bar"].data == RunData.PROC
    assert var_promotion_ctx.vars["baz"].data == RunData.PROC

    assert var_promotion_ctx.vars["bar"].cluster is True
    assert var_promotion_ctx.vars["baz"].cluster is True

    # Test depending on mymdc fields
    good_mymdc_code = """
    from damnit.context import Variable

    @Variable(title="foo")
    def foo(run, sample: "mymdc#sample_name", run_type: "mymdc#run_type"):
        return 42

    @Variable(title='X')
    def X(run, foo: "var#foo"):
        return foo
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

    # test tag validity
    empty_string_tag = """
    from damnit.context import Variable

    @Variable(title="Foo", tags=['bar', ''])
    def foo(run):
        return 42
    """
    ctx = mkcontext(empty_string_tag)
    with pytest.raises(ContextFileErrors, match='must be a non-empty string'):
        ctx.check()

def run_ctx_helper(context, run, run_number, proposal, caplog, input_vars=None):
    caplog.clear()
    # Track all error messages during creation. This is necessary because a
    # variable that throws an error will be logged by Results, the exception
    # will not bubble up.
    with caplog.at_level(logging.ERROR):
        results = context.execute(run, run_number, proposal, input_vars or {})

    # Check that there were no errors
    print(caplog.text)
    assert caplog.records == []
    return results

def test_results(mock_ctx, mock_run, caplog, tmp_path):
    run_number = 1000
    proposal = 1234
    results_create = lambda ctx: ctx.execute(mock_run, run_number, proposal, {})

    # Simple test
    results = run_ctx_helper(mock_ctx, mock_run, run_number, proposal, caplog)
    assert set(mock_ctx.ordered_vars()) <= results.cells.keys()

    # Check that the summary of a DataArray is a single number
    assert isinstance(results.cells["meta_array"].data, xr.DataArray)
    assert results.reduced["meta_array"].ndim == 0

    # Check the result values
    assert results.cells["scalar1"].data == 42
    assert results.cells["scalar2"].data == 3.14
    assert results.cells["empty_string"].data == ""
    np.testing.assert_equal(results.cells["array"].data, [42, 3.14])
    np.testing.assert_equal(results.cells["meta_array"].data.data, [run_number, proposal])
    assert results.cells["string"].data == str(get_proposal_path(mock_run))
    assert results.cells['plotly_mc_plotface'].data == px.bar(x=['a', 'b', 'c'], y=[1, 3, 2])

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
        assert tuple(results.cells.keys()) == ("start_time",)

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
        assert set(results.cells.keys()) == {"start_time", "foo"}
        assert results.cells['foo'].data is None

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
    import xarray as xr
    from damnit_ctx import Variable
    from matplotlib import pyplot as plt

    @Variable("2D ndarray")
    def twodarray(run):
        return np.random.rand(1000, 1000)

    @Variable("2D xarray")
    def twodxarray(run):
        return xr.DataArray(np.random.rand(100, 100))

    @Variable("2D-ish xarray")
    def twod_ish_xarray(run):
        return xr.DataArray(np.random.rand(1, 100))

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
        check_rgba(f["figure/data"])
        check_rgba(f["axes/data"])

        # Test that the summaries are the right size
        for var in ["twodarray", "twodxarray", "twod_ish_xarray"]:
            png = Image.open(io.BytesIO(f[f".reduced/{var}"][()]))
            assert np.asarray(png).shape == (THUMBNAIL_SIZE, THUMBNAIL_SIZE, 4)

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

    # Test returning complex results
    complex_code = """
    from damnit_ctx import Variable
    import numpy as np
    import xarray as xr

    data = np.array([1+1j, 2+2j, 3+3j])

    @Variable(title="Complex Dataset")
    def complex_dataset(run):
        return xr.Dataset(data_vars={"foo": xr.DataArray(data),})

    @Variable(title='Complex Array')
    def complex_array(run):
        return data
    """
    complex_ctx = mkcontext(complex_code)
    results = results_create(complex_ctx)
    results.save_hdf5(results_hdf5_path)

    dataset = xr.load_dataset(results_hdf5_path, group="complex_dataset", engine="h5netcdf")
    assert "foo" in dataset
    assert dataset['foo'].dtype == np.complex128
    with h5py.File(results_hdf5_path) as f:
        assert f[".reduced/complex_dataset"].asstr()[()].startswith("Dataset")
        assert np.allclose(f['complex_array/data'][()], np.array([1+1j, 2+2j, 3+3j]))

    # Test getting mymdc fields
    mymdc_code = """
    from damnit_ctx import Variable

    @Variable(title="Sample")
    def sample(run, x: "mymdc#sample_name"):
        return x

    @Variable(title="Run type")
    def run_type(run, x: "mymdc#run_type"):
        return x

    @Variable(title="Run Techniques")
    def techniques(run, x: "mymdc#techniques"):
        return ', '.join(t['name'] for t in x)
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
            result = dict(runs=[dict(id=1, sample_id=1, experiment_id=1)])
        elif "samples" in url:
            result = dict(name="mithril")
        elif "experiments" in url:
            result = dict(name="alchemy")
        elif "/runs/" in url:
            result = {'techniques': [
                {'identifier': 'PaNET01168', 'name': 'SFX'},
                {'identifier': 'PaNET01188', 'name': 'SAXS'},
            ]}

        response = MagicMock()
        response.json.return_value = result
        return response

    # Execute the context file and check the results
    with patch.object(requests, "get", side_effect=mock_get), \
         patch.object(ed.read_machinery, "find_proposal", return_value=tmp_path):
        results = results_create(mymdc_ctx)

    assert results.cells["sample"].data == "mithril"
    assert results.cells["run_type"].data == "alchemy"
    assert results.cells["techniques"].data == "SFX, SAXS"


def test_return_bool(mock_run, tmp_path):
    code = """
    from damnit_ctx import Variable

    @Variable()
    def bool(run):
        return True
    """
    ctx = mkcontext(code)
    results = ctx.execute(mock_run, 1000, 123, {})

    results_hdf5_path = tmp_path / 'results.h5'
    results.save_hdf5(results_hdf5_path)

    with h5py.File(results_hdf5_path) as f:
        assert f['bool/data'][()] == True
        assert f['.reduced/bool'][()] == True


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

    @Variable()
    def also_bad(run):
        return np.datetime64('1987-03-20')
    """
    bad_obj_ctx = mkcontext(bad_obj_code)
    results = bad_obj_ctx.execute(mock_run, 1000, 123, {})
    results_hdf5_path = tmp_path / 'results.h5'
    results.save_hdf5(results_hdf5_path)
    with h5py.File(results_hdf5_path) as f:
        assert set(f) == {".errors", ".reduced", "good", "start_time"}
        assert set(f[".reduced"]) == {"good", "start_time"}

def test_results_cell(mock_run, tmp_path):
    ctx_code = """
    from damnit_ctx import Variable, Cell
    import numpy as np

    @Variable()
    def var1(run):
        return 7

    @Variable(summary='sum')
    def var2(run):
        return Cell(np.arange(10), bold=False, background='#bbdddd')

    @Variable()
    def var3(run):
        return Cell(np.arange(7), summary_value=4)
    """
    bad_obj_ctx = mkcontext(ctx_code)
    results = bad_obj_ctx.execute(mock_run, 1000, 123, {})
    results_hdf5_path = tmp_path / 'results.h5'
    results.save_hdf5(results_hdf5_path)
    with h5py.File(results_hdf5_path) as f:
        assert f['.reduced/var2'][()] == 45
        assert f['.reduced/var2'].attrs['bold'] == False
        np.testing.assert_array_equal(
            f['.reduced/var2'].attrs['background'], [0xbb, 0xdd, 0xdd]
        )

        assert f['.reduced/var3'][()] == 4


def test_results_empty_array(mock_run, tmp_path, caplog):
    # test failing summary
    empty_array = """
    from damnit_ctx import Variable

    @Variable(title='Foo', summary='max')
    def foo(run):
        import numpy as np
        return np.array([])
    """
    ctx = mkcontext(empty_array)
    with caplog.at_level(logging.ERROR):
        results = ctx.execute(mock_run, 1000, 123, {})
        results.save_hdf5(tmp_path / 'results.h5', reduced_only=True)

        # One warning about foo should have been logged
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "ERROR"
        assert caplog.records[0].msg == "Failed to produce summary data"


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
    assert results.cells["dep_integer"].data == user_var_values["user_integer"] + 1
    assert results.cells["dep_number"].data == user_var_values["user_number"]
    assert results.cells["dep_boolean"].data == False
    assert results.cells["dep_string"].data == user_var_values["user_string"] * 2

def test_results_preview(mock_run, tmp_path):
    ctx_code = """
    from damnit_ctx import Variable, Cell
    import numpy as np
    import matplotlib.pyplot as plt

    @Variable()
    def var1(run):
        return Cell(np.zeros((5, 5, 5)), preview=np.ones(5))

    @Variable()
    def var2(run):
        arr = np.zeros((10, 10))
        fig, ax = plt.subplots()
        ax.plot(arr.mean())
        return Cell(arr, preview=fig)
    """
    ctx = mkcontext(ctx_code)
    results = ctx.execute(mock_run, 1000, 123, {})
    results_hdf5_path = tmp_path / 'results.h5'
    results.save_hdf5(results_hdf5_path)
    with h5py.File(results_hdf5_path) as f:
        np.testing.assert_array_equal(f['.preview/var1'][()], np.ones(5))
        check_png(f['.reduced/var1'])  # Summary thumbnail made from preview
        check_rgba(f['.preview/var2'])
        check_png(f['.reduced/var2'])

def test_filtering(mock_ctx, mock_run, caplog):
    run_number = 1000
    proposal = 1234

    # First run with raw data and a filter so only one Variable is executed
    ctx = mock_ctx.filter(run_data=RunData.RAW, name_matches=["string"])
    assert set(ctx.vars) == { "string" }
    results = run_ctx_helper(ctx, mock_run, run_number, proposal, caplog)
    assert set(results.cells) == { "string", "start_time" }

    # Now select a Variable with dependencies
    ctx = mock_ctx.filter(run_data=RunData.RAW, name_matches=["scalar2", "timestamp"])
    assert set(ctx.vars) == { "scalar1", "scalar2", "timestamp" }
    results = run_ctx_helper(ctx, mock_run, run_number, proposal, caplog)
    assert set(results.cells) == { "scalar1", "scalar2", "timestamp", "start_time" }
    ts = results.cells["timestamp"].data

    # Requesting a Variable that requires proc data with only raw data
    # should not execute anything.
    ctx = mock_ctx.filter(run_data=RunData.RAW, name_matches=["meta_array"])
    assert set(ctx.vars) == set()
    results = run_ctx_helper(ctx, mock_run, run_number, proposal, caplog)
    assert set(results.cells) == {"start_time"}

    # But with proc data all dependencies should be executed
    ctx = mock_ctx.filter(run_data=RunData.PROC, name_matches=["meta_array"])
    assert set(ctx.vars) == { "scalar1", "scalar2", "timestamp", "array", "meta_array" }
    results = run_ctx_helper(ctx, mock_run, run_number, proposal, caplog)
    assert set(results.cells) == { "scalar1", "scalar2", "timestamp", "array", "meta_array", "start_time" }
    assert results.cells["timestamp"].data > ts

    # Test cluster filtering
    ctx_code = """
    from damnit_ctx import Variable

    @Variable("foo", cluster=True)
    def foo(run):
        return 1

    @Variable("bar")
    def bar(run):
        return 1
    """
    ctx = mkcontext(ctx_code)
    assert set(ctx.filter().vars) == { "foo", "bar" }
    assert set(ctx.filter(cluster=True).vars) == { "foo" }
    assert set(ctx.filter(cluster=False).vars) == { "bar" }


def test_add_to_db(mock_db):
    db_dir, db = mock_db

    reduced_data = {
        "string": "foo",
        "scalar": 42,
        "float": 31.7,
        "image": b'\x89PNG\r\n\x1a\n...'  # Not a valid PNG, but good enough for this
    }

    reduced_objs = reduced_data_from_dict(reduced_data)
    reduced_objs["float"].attributes = {"background": [255, 0, 0]}

    add_to_db(reduced_objs, db, 1234, 42)

    cursor = db.conn.execute("SELECT * FROM runs")
    row = cursor.fetchone()

    assert row["string"] == reduced_data["string"]
    assert row["scalar"] == reduced_data["scalar"]
    assert row["float"] == reduced_data["float"]
    assert row["image"] == reduced_data["image"]

    row = db.conn.execute(
        "SELECT * FROM run_variables WHERE proposal=1234 AND run=42 AND name='float'"
    ).fetchone()
    assert json.loads(row["attributes"]) == {"background": [255, 0, 0]}

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
        extractor = RunExtractor(1234, 42, cluster=False, run_data=RunData.ALL)

    # Test regular variables and slurm variables are executed
    reduced_data = reduced_data_from_dict({ "n": 53 })
    with patch(f"{pkg}.RunExtractor.extract_in_subprocess", return_value=reduced_data) as extract_in_subprocess, \
         MockCommand.fixed_output("sbatch", "9876; maxwell") as sbatch:
        extractor.extract_and_ingest()
        extract_in_subprocess.assert_called_once()
        extractor.kafka_prd.send.assert_called()
        sbatch.assert_called()

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

def test_custom_environment(mock_db, venv, monkeypatch, qtbot):
    db_dir, db = mock_db
    monkeypatch.chdir(db_dir)

    ctxrunner_deps = ["extra_data", "matplotlib", "plotly", "pyyaml", "requests"]

    # Install dependencies for ctxrunner and a light-weight package (sfollow)
    # that isn't in our current environment.
    subprocess.run([venv.python, "-m", "pip", "install", *ctxrunner_deps, "sfollow"],
                   check=True)

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
    db.metameta["context_python"] = str(venv.python)

    with patch(f"{pkg}.KafkaProducer"):
        RunExtractor(1234, 42, mock=True).extract_and_ingest()

    with h5py.File(db_dir / "extracted_data" / "p1234_r42.h5") as f:
        assert f["foo/data"][()] == 42

    # Make sure that the GUI evaluates the context file correctly (which it does
    # upon opening a database directory).
    win = MainWindow(db_dir, False)
    qtbot.addWidget(win)

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


def test_copy_ctx_and_user_vars(tmp_path, mock_db, mock_user_vars):
    prev_db_dir, prev_db = mock_db

    for var in mock_user_vars.values():
        prev_db.add_user_variable(var)

    db_dir = tmp_path / "new"
    db_dir.mkdir()
    with patch("damnit.backend.supervisord.start_backend"):
        initialize_and_start_backend(
            db_dir,
            1234,
            context_file_src=(prev_db_dir / "context.py"),
            user_vars_src=(prev_db_dir / "runs.sqlite")
        )

    ctx_file = db_dir / "context.py"
    assert ctx_file.is_file()
    assert ctx_file.read_text() == (prev_db_dir / "context.py").read_text()

    db_file = db_dir / "runs.sqlite"
    assert db_file.is_file()
    assert set(DamnitDB(db_file).get_user_variables()) == set(mock_user_vars)


def test_event_processor(mock_db, caplog):
    db_dir, db = mock_db
    db.metameta["proposal"] = 1234

    with patch('damnit.backend.listener.KafkaConsumer') as kcon:
        processor = EventProcessor(db_dir)

    kcon.assert_called_once()
    assert len(local_extraction_threads) == 0

    # slurm not available
    with (
        patch('subprocess.run', side_effect=FileNotFoundError),
        patch('damnit.backend.extraction_control.ExtractionSubmitter.execute_direct', lambda *_: sleep(1))
    ):
        with caplog.at_level(logging.WARNING):
            event = MagicMock(timestamp=time())
            processor.handle_event(event, {'proposal': 1234, 'run': 1}, RunData.RAW)

        assert 'Slurm not available' in caplog.text
        assert len(local_extraction_threads) == 1
        local_extraction_threads[0].join()

        with caplog.at_level(logging.WARNING):
            for idx in range(MAX_CONCURRENT_THREADS + 1):
                event = MagicMock(timestamp=time())
                processor.handle_event(event, {'proposal': 1234, 'run': idx + 1}, RunData.RAW)

        assert len(local_extraction_threads) == MAX_CONCURRENT_THREADS
        assert 'Too many events processing' in caplog.text


def test_job_tracker():
    tracker = ExtractionJobTracker()

    d = {'proposal': 1234, 'data': 'all', 'hostname': '', 'username': '',
         'slurm_cluster': '', 'slurm_job_id': '', 'status': 'RUNNING'}

    prid1, prid2 = str(uuid4()), str(uuid4())

    # Add two running jobs
    tracker.on_processing_state_set(d | {'run': 1, 'processing_id': prid1})
    tracker.on_processing_state_set(d | {
        'run': 2, 'processing_id': prid2,
        'slurm_cluster': 'maxwell', 'slurm_job_id': '321'
    })
    assert set(tracker.jobs) == {prid1, prid2}

    # One job finishes normally
    tracker.on_processing_finished({'processing_id': prid1})
    assert set(tracker.jobs) == {prid2}

    # The other one fails to send a finished message, so checking Slurm reveals
    # that it failed.
    with MockCommand.fixed_output('squeue', '') as fake_squeue:
        tracker.check_slurm_jobs()

    fake_squeue.assert_called()
    assert set(tracker.jobs) == set()

def test_transient_variables(mock_run, mock_db, tmp_path):
    db_dir, db = mock_db

    ctx_code = """
    from damnit_ctx import Variable, Cell
    import numpy as np

    @Variable()
    def var1(run):
        return 7

    @Variable(transient=True)
    def var2(run, data: 'var#var1'):
        # transient vars can return any data type
        return [np.arange(data), run]

    @Variable(summary='max')
    def var3(run, data: 'var#var2'):
        data, run = data
        return data.size * data
    """
    ctx = mkcontext(ctx_code)
    results = ctx.execute(mock_run, 1000, 123, {})
    results_hdf5_path = tmp_path / 'results.h5'
    results.save_hdf5(results_hdf5_path)

    with h5py.File(results_hdf5_path) as f:
        assert '.reduced/var1' in f
        assert 'var1' in f
        # transient variables are not saved
        assert '.reduced/var2' not in f
        assert 'var2' not in f
        assert '.reduced/var3' in f
        assert 'var3' in f

        assert f['.reduced/var3'][()] == 42
        assert np.allclose(f['var3/data'][()], np.arange(7) * 7)

    reduced_data = load_reduced_data(results_hdf5_path)
    add_to_db(reduced_data, db, 1000, 123)
    vars = db.conn.execute('SELECT value FROM run_variables WHERE name="var3"').fetchall()
    assert vars[0]['value'] == 42
    # also not saved in the db
    vars = db.conn.execute('SELECT * FROM run_variables WHERE name="var2"').fetchall()
    assert vars == []


def test_capture_errors(mock_run, mock_db, tmp_path):
    db_dir, db = mock_db

    ctx_code = """
    from damnit_ctx import Variable, Skip

    @Variable()
    def var1(run):
        1/0

    @Variable()
    def var2(run):
        raise Skip("Testing Skip")

    @Variable(transient=True)
    def var3(run):
        return 1/0

    @Variable()
    def var4(run, var3: 'var#var3'):
        return 1
    """
    ctx = mkcontext(ctx_code)
    results = ctx.execute(mock_run, 1000, 123, {})
    results_hdf5_path = tmp_path / 'results.h5'
    results.save_hdf5(results_hdf5_path)

    with h5py.File(results_hdf5_path) as f:
        for i in [1, 2, 4]:
            assert f'.errors/var{i}' in f
            assert f'.reduced/var{i}' not in f
            assert f'var{i}' not in f

    reduced_data = load_reduced_data(results_hdf5_path)
    add_to_db(reduced_data, db, 1000, 123)
    attrs = db.conn.execute(
        "SELECT attributes FROM run_variables WHERE name='var1'"
    ).fetchone()[0]
    assert json.loads(attrs) == {
        'error': 'division by zero', 'error_cls': 'ZeroDivisionError'
    }

    attrs = db.conn.execute(
        "SELECT attributes FROM run_variables WHERE name='var2'"
    ).fetchone()[0]
    assert json.loads(attrs) == {'error': 'Testing Skip', 'error_cls': 'Skip'}

    attrs = db.conn.execute(
        "SELECT attributes FROM run_variables WHERE name='var4'"
    ).fetchone()[0]
    assert json.loads(attrs) == {"error": "\ndependency (var3) failed: ZeroDivisionError('division by zero')", "error_cls": "Exception"}


def test_variable_group(mock_run, tmp_path, caplog):
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class TestGroup:
        calibration_factor: float = 1.0
        test_value: int = 0

        def _some_value(self):
            return 42

        @Variable(title="Raw Value", tags="raw")
        def raw_value(self, run):
            return 10

        @Variable()
        def calibrated_value(self, run, raw: "self#raw_value"):
            return raw * self.calibration_factor

        @Variable(title="Offset Value", summary="max")
        def offset_value(self, run, offset: 'input#offset'=5):
            return self.test_value + offset + self._some_value()

    # Instantiate the group
    my_group = TestGroup("My Test Group", calibration_factor=1.5, test_value=5, tags="test_group")
    your_group = TestGroup(calibration_factor=2.0, test_value=10)
    """
    ctx = mkcontext(code)

    # Test variable naming and structure
    assert len(ctx.vars) == 6
    assert "my_group__raw_value" in ctx.vars
    assert "my_group__calibrated_value" in ctx.vars
    assert "my_group__offset_value" in ctx.vars
    assert "your_group__raw_value" in ctx.vars
    assert "your_group__calibrated_value" in ctx.vars
    assert "your_group__offset_value" in ctx.vars

    # Test title generation
    assert ctx.vars["my_group__raw_value"].title == "My Test Group/Raw Value"
    assert ctx.vars["my_group__calibrated_value"].title == "My Test Group/calibrated_value"
    assert ctx.vars["your_group__offset_value"].title == "your_group/Offset Value"

    # Test tag merging
    assert "test_group" in ctx.vars["my_group__raw_value"].tags
    assert "raw" in ctx.vars["my_group__raw_value"].tags

    # Check topological ordering for dependency resolution
    ordered = ctx.ordered_vars()
    assert ordered.index("my_group__raw_value") < ordered.index("my_group__calibrated_value")

    # Execute and check results
    results = run_ctx_helper(ctx, mock_run, 1000, 1234, caplog, input_vars={"offset": 10})
    assert results.cells["my_group__raw_value"].data == 10
    # Check that calibration factor from group __init__ was used
    assert results.cells["my_group__calibrated_value"].data == 15.0
    # Check that input# variables are passed through correctly
    assert results.cells["my_group__offset_value"].data == 57.0

    # Test that an error in a dependency correctly stops downstream variables
    error_code = """
    from damnit_ctx import Variable, Group

    @Group
    class ErrorGroup:
        @Variable()
        def bad_var(self, run):
            return 1 / 0

        @Variable()
        def good_var(self, run, bad_data: "self#bad_var"):
            return 42

    error_group = ErrorGroup("Error Group")
    """
    ctx_err = mkcontext(error_code)

    with caplog.at_level(logging.ERROR):
        results_err = ctx_err.execute(mock_run, 1000, 1234, {})

    # Check that an error was logged for bad_var
    assert caplog.records[0].levelname == "ERROR"

    assert "error_group__good_var" not in results_err.cells
    assert "error_group__good_var" not in results_err.errors
    # The error itself should be recorded
    assert "error_group__bad_var" in results_err.errors

    code = """
    from damnit_ctx import Variable, Group

    # top-level variable to act as a global dependency
    @Variable(title="Global Base")
    def base(run):
        return 100

    @Group
    class ProcessingGroup:
        offset: int = 0

        @Variable(title="Local base", transient=True)
        def base(self, run):
            return 50

        @Variable(title="Step 1")
        def step1(self, run, base: "var#base"):
            # This variable depends on a top-level ('root') variable
            return base + 1

        @Variable(title="Step 2")
        def step2(self, run, s1_data: "self#step1", base: "self#base"):
            # This variable has an intra-group dependency
            return s1_data * 2 + base + self.offset

    proc_b = ProcessingGroup(title="Processor B", offset=-3)
    proc_a = ProcessingGroup(title="Processor A")

    # A variable that depends on outputs from both group instances
    @Variable(title="A vs B")
    def a_vs_b_diff(run, a_result: "var#proc_a.step2", b_result: "var#proc_b.step2"):
        return a_result - b_result
    """
    ctx = mkcontext(code)

    # Check that all variable names are generated correctly
    expected_vars = {
        "base",
        "proc_a__base", "proc_a__step1", "proc_a__step2",
        "proc_b__base", "proc_b__step1", "proc_b__step2",
        "a_vs_b_diff"
    }
    assert set(ctx.vars.keys()) == expected_vars
    
    for var in ctx.vars.values():
        print(var.name, var.title, var.tags, var.annotations())

    # Check the dependency graph and execution order
    ordered_vars = ctx.ordered_vars()

    assert ordered_vars.index("base") < ordered_vars.index("proc_a__step1")
    assert ordered_vars.index("base") < ordered_vars.index("proc_b__step1")

    assert ordered_vars.index("proc_a__step1") < ordered_vars.index("proc_a__step2")
    assert ordered_vars.index("proc_b__step1") < ordered_vars.index("proc_b__step2")

    assert ordered_vars.index("proc_a__step2") < ordered_vars.index("a_vs_b_diff")
    assert ordered_vars.index("proc_b__step2") < ordered_vars.index("a_vs_b_diff")

    # Execute the context and check the results
    results = run_ctx_helper(ctx, mock_run, 1000, 1234, caplog)

    # Check results for proc_a
    assert results.cells["proc_a__step1"].data == 101
    assert results.cells["proc_a__step2"].data == 252

    # Check results for proc_b
    assert results.cells["proc_b__step1"].data == 101
    assert results.cells["proc_b__step2"].data == 249

    # Check the final composer variable
    assert results.cells["a_vs_b_diff"].data == 3

    results_hdf5_path = tmp_path / "results.hdf5"
    results.save_hdf5(results_hdf5_path)
    with h5py.File(results_hdf5_path) as f:
        assert f["proc_a__step1/data"][()] == 101
        assert f["proc_a__step2/data"][()] == 252
        assert f["proc_b__step1/data"][()] == 101
        assert f["proc_b__step2/data"][()] == 249
        assert f["a_vs_b_diff/data"][()] == 3
        assert "base" in f
        assert "proc_a__base" not in f

    # Test that a missing root dependency is handled correctly
    bad_root_code = """
    from damnit_ctx import Variable, Group

    @Group
    class BadRootGroup:
        @Variable()
        def step1(self, run, base: "var#missing_global"):
            return 1

    bad_group = BadRootGroup("Bad Group")
    """
    with pytest.raises(KeyError, match="missing_global"):
        mkcontext(bad_root_code)
    
    # Test that a missing dependency is handled correctly
    bad_root_code = """
    from damnit_ctx import Variable, Group

    @Group
    class BadRootGroup:
        @Variable()
        def step1(self, run, base: "self#missing_local"):
            return 1

    bad_group = BadRootGroup("Bad Group")
    """
    with pytest.raises(KeyError, match="missing_local"):
        mkcontext(bad_root_code)

    # test inheritance of Group
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class BaseGroup:
        value: int = 0.

        @Variable()
        def step1(self, run):
            return self.value + 1

    class ChildGroup(BaseGroup):
        @Variable()
        def step2(self, run, s1_data: "self#step1"):
            return s1_data * 2

    class ChildGroup2(BaseGroup):
        @Variable()
        def step1(run):
            return 'overridden step1'

    base_group = BaseGroup("Base Group")
    child_group = ChildGroup("Child Group", value=1, tags=['Child', 'other tag'])
    child_group2 = ChildGroup2("Child Group 2")
    """
    ctx = mkcontext(code)
    assert set(ctx.vars) == {"base_group__step1", "child_group2__step1", "child_group__step1", "child_group__step2"}

    assert ctx.vars["base_group__step1"].tags is None
    assert set(ctx.vars["child_group__step1"].tags) == {"Child", "other tag"}
    assert set(ctx.vars["child_group__step2"].tags) == {"Child", "other tag"}

    results = run_ctx_helper(ctx, mock_run, 1000, 1234, caplog)
    assert results.cells["child_group__step2"].data == 4
    assert results.cells["child_group2__step1"].data == 'overridden step1'

    # Test Group composition
    code = """
    from damnit_ctx import Variable, Group

    # Define some reusable, low-level groups
    @Group
    class XGMGroup:
        factor: float = 1.0

        @Variable(title="Intensity")
        def intensity(self, run):
            return 10 * self.factor

    @Group
    class DetectorGroup:
        @Variable(title="Image", transient=True)
        def image(self, run):
            import numpy as np
            return np.ones((100, 100), dtype=np.uint8)

        @Variable(summary="sum")
        def photon_count(self, run, image_data: "self#image"):
            # A mock calculation
            return image_data.sum()

    # A mid-level group that composes the low-level ones
    @Group
    class InstrumentDiagnostics:
        # Nested group instances
        xgm = XGMGroup("XGM", factor=1.5)
        detector = DetectorGroup("Detector", tags="detector")

        @Variable(title="Intensity per Photon")
        def intensity_per_photon(self, run,
                                 intensity: "self#xgm.intensity",
                                 photons: "self#detector.photon_count"):
            # This depends on variables from two different nested subgroups
            return intensity / photons

    # A top-level group that nests the mid-level group (2 levels of nesting)
    @Group
    class Experiment:
        instrument = InstrumentDiagnostics(title="SCS Instrument", tags=["scs"])

        @Variable(title="Experiment Quality")
        def quality(self, run, ipp: "self#instrument.intensity_per_photon"):
            # Depends on a variable from a nested group
            return "good" if ipp > 1.0 else "bad"
        
        @Variable()
        def deep_nested_access(run, data: "self#instrument.detector.image"):
            return data.mean()

    # Instantiate the top-level group
    exp1 = Experiment(title="My Experiment")

    @Variable()
    def nested_var_access(run, data: "var#exp1.instrument.detector.image"):
        return data.mean()
    """
    ctx = mkcontext(code)

    # Check that all variables from the nested structure are present and named correctly
    expected_vars = {
        "exp1__instrument__xgm__intensity",
        "exp1__instrument__detector__image",
        "exp1__instrument__detector__photon_count",
        "exp1__instrument__intensity_per_photon",
        "exp1__quality",
        "exp1__deep_nested_access",
        "nested_var_access",
    }
    assert set(ctx.vars.keys()) == expected_vars

    # Check title generation with multiple levels of nesting
    assert ctx.vars["exp1__instrument__xgm__intensity"].title == "My Experiment/SCS Instrument/XGM/Intensity"
    assert ctx.vars["exp1__instrument__detector__photon_count"].title == "My Experiment/SCS Instrument/Detector/photon_count"
    assert ctx.vars["exp1__instrument__intensity_per_photon"].title == "My Experiment/SCS Instrument/Intensity per Photon"
    assert ctx.vars["exp1__quality"].title == "My Experiment/Experiment Quality"

    # Check that tags are correctly inherited down the hierarchy
    # 'detector' tag from DetectorGroup instance should be present
    assert "detector" in ctx.vars["exp1__instrument__detector__photon_count"].tags
    # 'scs' tag from InstrumentDiagnostics instance should be present on all its children
    assert "scs" in ctx.vars["exp1__instrument__detector__photon_count"].tags
    assert "scs" in ctx.vars["exp1__instrument__xgm__intensity"].tags

    for var in ctx.vars.values():
        print(var.name, var.title, var.tags)

    # Check dependency resolution and execution order
    ordered_vars = ctx.ordered_vars()
    # Check dependencies within the deepest group
    assert ordered_vars.index("exp1__instrument__detector__image") < ordered_vars.index("exp1__instrument__detector__photon_count")
    # Check dependencies for the mid-level composer variable
    assert ordered_vars.index("exp1__instrument__xgm__intensity") < ordered_vars.index("exp1__instrument__intensity_per_photon")
    assert ordered_vars.index("exp1__instrument__detector__photon_count") < ordered_vars.index("exp1__instrument__intensity_per_photon")
    # Check dependencies for the top-level composer variable
    assert ordered_vars.index("exp1__instrument__intensity_per_photon") < ordered_vars.index("exp1__quality")

    # Execute the context and verify the results
    results = run_ctx_helper(ctx, mock_run, 1000, 1234, caplog)

    # Verify intermediate and final calculations
    assert results.cells["exp1__instrument__xgm__intensity"].data == 15.0  # 10 * 1.5
    assert results.cells["exp1__instrument__detector__photon_count"].data == 10_000
    assert results.cells["exp1__instrument__intensity_per_photon"].data == 15.0 / 10_000
    assert results.cells["exp1__quality"].data == "bad"
    assert results.cells["exp1__deep_nested_access"].data == 1.0

    # Check that transient variables are not saved
    results_hdf5_path = tmp_path / "results.h5"
    results.save_hdf5(results_hdf5_path)
    with h5py.File(results_hdf5_path) as f:
        # The transient 'image' variable should not exist in the file
        assert "exp1__instrument__detector__image" not in f
        # A non-transient variable should exist
        assert "exp1__quality" in f
        assert f["exp1__quality/data"][()] == b"bad"

    code_group_settings = """
    from damnit_ctx import Variable, Group

    @Group(title='A', tags='A')
    class A:
        @Variable(cluster=False, transient=True)
        def var1(self, run):
            return 42

    class B(A):
        @Variable(title='b', tags='B', cluster=True)
        def var2(self, run, data: 'self#var1'):
            return data + 1

        @Variable()
        def var3(self, run, data: 'self#var2'):
            return data + 2

    @Group
    class C(A):
        @Variable()
        def var1(self, run):
            return 43

    @Group(title='D', tags='D')
    class D:
        @Variable(title='d_var', data='raw', tags='D_var', cluster=True)
        def d_var(self, run):
            return 44

    @Group(tags='DD')
    class DD(D):
        @Variable(title='DD_var', data='proc', tags='DD_var')
        def dd_var(self, run, data: 'self#d_var'):
            return data + 1

        @Variable()
        def dd_var2(self, run, data: 'self#dd_var'):
            return data + 2

    a = A()
    aa = A(title='AA', tags='AA')
    b = B()
    c = C()
    d = D()
    dd = DD()
    dd2 = DD(title='DD2', tags='DD2')
    """
    ctx = mkcontext(code_group_settings)

    # Check that the group settings are applied correctly
    assert ctx.vars["a__var1"].title == "A/var1"
    assert set(ctx.vars["a__var1"].tags) == {'A'}
    assert ctx.vars["a__var1"].cluster is False
    assert ctx.vars["a__var1"].transient is True
    assert ctx.vars["a__var1"].data == RunData.RAW

    assert ctx.vars["aa__var1"].title == "AA/var1"
    assert set(ctx.vars["aa__var1"].tags) == {"AA"}
    assert ctx.vars["aa__var1"].cluster is False
    assert ctx.vars["aa__var1"].transient is True
    assert ctx.vars["aa__var1"].data == RunData.RAW

    assert ctx.vars["b__var1"].title == "A/var1"
    assert set(ctx.vars["b__var1"].tags) == {'A'}
    assert ctx.vars["b__var1"].cluster is False
    assert ctx.vars["b__var1"].transient is True
    assert ctx.vars["b__var1"].data == RunData.RAW

    assert ctx.vars["b__var2"].title == "A/b"
    assert set(ctx.vars["b__var2"].tags) == {"A", "B",}
    assert ctx.vars["b__var2"].cluster is True
    assert ctx.vars["b__var2"].transient is False
    assert ctx.vars["b__var2"].data == RunData.RAW

    assert ctx.vars["b__var3"].title == "A/var3"
    assert set(ctx.vars["b__var3"].tags) == {'A'}
    assert ctx.vars["b__var3"].cluster is True  # promoted to cluster as it dependes on a cluster variable
    assert ctx.vars["b__var3"].transient is False
    assert ctx.vars["b__var3"].data == RunData.RAW

    assert ctx.vars["c__var1"].title == "c/var1"
    assert ctx.vars["c__var1"].tags is None
    assert ctx.vars["c__var1"].cluster is False

    assert ctx.vars["d__d_var"].title == "D/d_var"
    assert set(ctx.vars["d__d_var"].tags) == {"D", "D_var"}
    assert ctx.vars["d__d_var"].cluster is True
    assert ctx.vars["d__d_var"].data == RunData.RAW

    assert ctx.vars["dd__d_var"].title == "dd/d_var"
    assert set(ctx.vars["dd__d_var"].tags) == {"DD", "D_var"}
    assert ctx.vars["dd__d_var"].cluster is True
    assert ctx.vars["dd__dd_var"].title == "dd/DD_var"
    assert set(ctx.vars["dd__dd_var"].tags) == {"DD", "DD_var"}
    # assert ctx.vars["dd__dd_var"].cluster is False  # <<< This is a bug, it should be False
    assert ctx.vars["dd__dd_var"].data == RunData.PROC
    assert ctx.vars["dd__dd_var2"].title == "dd/dd_var2"
    assert set(ctx.vars["dd__dd_var2"].tags) == {"DD"}
    # assert ctx.vars["dd__dd_var2"].cluster is False  # <<< This is a bug, it should be False
    assert ctx.vars["dd__dd_var2"].data == RunData.PROC

    assert ctx.vars["dd2__dd_var"].title == "DD2/DD_var"
    assert set(ctx.vars["dd2__dd_var"].tags) == {"DD2", "DD_var"}
    # assert ctx.vars["dd2__dd_var"].cluster is False  # This is a bug, it should be False
    assert ctx.vars["dd2__dd_var2"].title == "DD2/dd_var2"
    assert set(ctx.vars["dd2__dd_var2"].tags) == {"DD2"}
    # assert ctx.vars["dd2__dd_var2"].cluster is False  # This is a bug, it should be False


    code_fields = """"""

def test_asdf():
    code = """
    from damnit_ctx import Variable, Group

    @Group(tags=['ClassA'])
    class A:
        @Variable(title='A Variable', tags=['varA'])
        def a_var(self, run):
            return 42

    class B(A):
        @Variable(title='B Variable', tags=['varB'])
        def b_var(self, run, a_data: "self#a_var"):
            return a_data + 1

    class C(A):
        @Variable(title='C Variable', tags=['varC'])
        def a_var(self, run):
            return 43

    class D(A):
        pass

    @Group
    class E(A):
        pass

    @Group(title='F Group', tags=['ClassF'])
    class F(A):
        pass

    a = A()
    b = B()
    c = C()
    d = D()
    e = E()
    f = F()
    aa = A(title='A Group', tags=['group_a'])
    bb = B(title='B Group', tags=['group_b'])
    cc = C(title='C Group', tags=['group_c'])
    dd = D(title='D Group', tags=['group_d'])
    ee = E(title='E Group', tags=['group_e'])
    ff = F(title='F', tags=['group_f'])
    """
    ctx = mkcontext(code)

    assert ctx.vars["a__a_var"].title == "a/A Variable"
    assert ctx.vars["b__a_var"].title == "b/A Variable"
    assert ctx.vars["c__a_var"].title == "c/C Variable"
    assert ctx.vars["d__a_var"].title == "d/A Variable"
    assert ctx.vars["e__a_var"].title == "e/A Variable"
    assert ctx.vars["f__a_var"].title == "F Group/A Variable"

    assert ctx.vars["aa__a_var"].title == "A Group/A Variable"
    assert ctx.vars["bb__a_var"].title == "B Group/A Variable"
    assert ctx.vars["cc__a_var"].title == "C Group/C Variable"
    assert ctx.vars["dd__a_var"].title == "D Group/A Variable"
    assert ctx.vars["ee__a_var"].title == "E Group/A Variable"
    assert ctx.vars["ff__a_var"].title == "F/A Variable"

    assert set(ctx.vars["a__a_var"].tags) == {"varA", "ClassA"}
    assert set(ctx.vars["b__a_var"].tags) == {"varA", "ClassA"}
    assert set(ctx.vars["b__b_var"].tags) == {"varB", "ClassA"}
    assert set(ctx.vars["c__a_var"].tags) == {"varC", "ClassA"}
    assert set(ctx.vars["d__a_var"].tags) == {"varA", "ClassA"}
    assert set(ctx.vars["e__a_var"].tags) == {"varA"}
    assert set(ctx.vars["f__a_var"].tags) == {"varA", "ClassF"}

    assert set(ctx.vars["aa__a_var"].tags) == {"varA", "group_a"}
    assert set(ctx.vars["bb__a_var"].tags) == {"varA", "group_b"}
    assert set(ctx.vars["cc__a_var"].tags) == {"varC", "group_c"}
    assert set(ctx.vars["dd__a_var"].tags) == {"varA", "group_d"}
    assert set(ctx.vars["ee__a_var"].tags) == {"varA", "group_e"}
    assert set(ctx.vars["ff__a_var"].tags) == {"varA", "group_f"}