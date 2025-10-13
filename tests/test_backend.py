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
from pathlib import Path
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

from damnit.backend import listener_is_running, initialize_proposal, start_listener
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

    no_parenthesis = """
    from damnit.context import Variable
    
    @Variable
    def foo(run):
        return 42
    """
    ctx = mkcontext(no_parenthesis)
    assert ctx.vars["foo"].title == "foo"
    assert ctx.vars["foo"].name == "foo"


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

def test_initialize_proposal(tmp_path):
    db_dir = tmp_path / "foo"
    good_file_mode = "-rw-rw-rw-"
    path_filemode = lambda p: stat.filemode(p.stat().st_mode)

    initialize_proposal(db_dir, 1234)

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


def test_start_listener(tmp_path, bound_port, request):
    supervisord_config_path = tmp_path / "supervisord.conf"
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
        assert start_listener(tmp_path)

    # Check the config file
    assert supervisord_config_path.is_file()
    assert path_filemode(supervisord_config_path) == good_file_mode

    # We should have a log file and PID file. They aren't created immediately so
    # we wait a bit for it.
    pid_path = tmp_path / "supervisord.pid"
    log_path = tmp_path / "supervisord.log"
    wait_until(lambda: pid_path.is_file() and log_path.is_file())
    pid = int(pid_path.read_text())

    # Set a finalizer to kill supervisord at the end
    request.addfinalizer(lambda: kill_pid(pid))

    assert path_filemode(pid_path) == good_file_mode
    assert path_filemode(log_path) == good_file_mode

    # Check that it's running
    assert listener_is_running(tmp_path)

    wait_until(lambda: (tmp_path / "started").is_file(), timeout=1)

    # Stop the program
    supervisorctl = ["supervisorctl", "-c", str(supervisord_config_path)]
    subprocess.run([*supervisorctl, "stop", "damnit"]).check_returncode()

    # Check that the subprocess was also killed
    wait_until(lambda: (tmp_path / "stopped").is_file())
    assert not listener_is_running(tmp_path)

    # Try starting it again. This time we don't pass the proposal number, it
    # should be picked up from the existing database.
    with patch(f"{pkg}.write_supervisord_conf",
               side_effect=mock_write_supervisord_conf):
        assert start_listener(tmp_path)

    assert listener_is_running(tmp_path)

    # Now kill supervisord
    kill_pid(pid)
    assert not listener_is_running(tmp_path)

    # Change the config to use the bound port
    mock_write_supervisord_conf(tmp_path, port=bound_port)

    # And try starting it again
    with patch(f"{pkg}.write_supervisord_conf",
               side_effect=mock_write_supervisord_conf):
        assert start_listener(tmp_path)

    wait_until(lambda: pid_path.is_file() and log_path.is_file())
    pid = int(pid_path.read_text())
    request.addfinalizer(lambda: kill_pid(pid))

    # Check that the backend is running
    assert listener_is_running(tmp_path)

    # Trying to start it again should do nothing
    with patch(f"{pkg}.write_supervisord_conf",
               side_effect=mock_write_supervisord_conf):
        assert start_listener(tmp_path)

    assert listener_is_running(tmp_path)


def test_copy_ctx_and_user_vars(tmp_path, mock_db, mock_user_vars):
    prev_db_dir, prev_db = mock_db

    for var in mock_user_vars.values():
        prev_db.add_user_variable(var)

    db_dir = tmp_path / "new"
    db_dir.mkdir()
    initialize_proposal(
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

def test_listener(mock_sandbox_out_file, tmp_path, caplog, monkeypatch):
    monkeypatch.setenv("XFEL_DATA_ROOT", str(tmp_path))

    # Create the processor and get it to run our mock sandbox script
    with patch('damnit.backend.listener.KafkaConsumer') as kcon:
        processor = EventProcessor(tmp_path)
        processor.db.settings["sandbox_args"] = str(Path(__file__).parent / "mock_sandbox.sh")

        # We need to allow local processing or it will give up after running
        # through slurm fails.
        processor.db.settings["allow_local_processing"] = True

    # Helper function to wait for all active jobs to finish
    def wait_for_jobs():
        for thread in local_extraction_threads:
            thread.join()

    # Helper function to count how many jobs were executed by counting the
    # number of lines in the file our mock sandbox script writes.
    def wait_and_count_jobs():
        wait_for_jobs()
        with mock_sandbox_out_file.open() as f:
            return sum(1 for _ in f)

    kcon.assert_called_once()
    assert len(local_extraction_threads) == 0

    # Create an 'official' database
    proposal_dir = tmp_path / "MID" / "202501" / "p001234"
    proposal_dir.mkdir(parents=True)
    db_dir = proposal_dir / "usr/Shared/amore"
    initialize_proposal(db_dir, 1234)

    # Reprocess a run. Because the listener is in static mode by default it
    # should not do anything.
    event = MagicMock(timestamp=time())
    processor.handle_event(event, {"proposal": 1234, "run": 1}, RunData.RAW)
    assert len(processor.db.proposal_db_dirs(1234)) == 0
    assert len(local_extraction_threads) == 0
    wait_for_jobs()
    assert not mock_sandbox_out_file.exists()

    # With static mode disabled it should add the database to the proposal and
    # process the run.
    processor.db.settings["static_mode"] = False
    with caplog.at_level(logging.WARNING):
        processor.handle_event(event, {"proposal": 1234, "run": 1}, RunData.RAW)
    assert wait_and_count_jobs() == 1

    # We should get a warning about slurm not being available
    assert 'Slurm job submission failed' in caplog.text

    # Add an unofficial database so that the listener launches two jobs
    mock_sandbox_out_file.unlink()
    fake_db_dir = tmp_path / "fakedb"
    initialize_proposal(fake_db_dir, 1234)
    processor.db.add_proposal_db(1234, fake_db_dir, False)
    processor.handle_event(event, {"proposal": 1234, "run": 1}, RunData.RAW)
    assert wait_and_count_jobs() == 2

    # Test processing too many runs concurrently
    mock_sandbox_out_file.unlink()
    with caplog.at_level(logging.WARNING):
        for idx in range(MAX_CONCURRENT_THREADS + 1):
            event = MagicMock(timestamp=time())
            processor.handle_event(event, {'proposal': 1234, 'run': idx + 1}, RunData.RAW)

    assert len(local_extraction_threads) == MAX_CONCURRENT_THREADS
    assert 'Too many events processing' in caplog.text
    assert wait_and_count_jobs() == MAX_CONCURRENT_THREADS

    # With a non-existent proposal specified handle_event() should do nothing
    mock_sandbox_out_file.unlink()
    with caplog.at_level(logging.WARNING):
        processor.handle_event(event, {'proposal': 4321, 'run': 1}, RunData.RAW)
    assert "Could not find proposal directory" in caplog.text
    assert not mock_sandbox_out_file.exists()

    # Test removing a database
    processor.db.remove_proposal_db(fake_db_dir)
    assert processor.db.proposal_db_dirs(1234) == [db_dir]


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


def test_pattern_matching_dependency(mock_run):
    ctx_code = """
    from damnit_ctx import Variable, Skip
    import numpy as np

    @Variable()
    def res1(run, data: 'var#var?'):
        result = data['var1'] + data['var2'].sum() + data['var3'].sum()
        return [result, len(data)]

    @Variable()
    def res2(run, data: 'var#var*'):
        return ','.join(sorted(data.keys()))

    @Variable()
    def res3(run, data: 'var#bar?'):
        # no match on data, this variable won't execute
        return 1

    @Variable()
    def var1(run):
        return 7

    @Variable()
    def var2(run, data: 'var#var1'):
        # transient vars can return any data type
        return np.arange(data)

    @Variable()
    def var3(run, data: 'var#var2'):
        return data.size * data

    @Variable()
    def var10(run):
        return 10

    @Variable()
    def barb(run):
        raise Skip()
    """
    ctx = mkcontext(ctx_code)
    results = ctx.execute(mock_run, 1000, 123, {})

    assert results.cells['res1'].data.tolist() == [175, 3]
    assert results.cells['res2'].data == 'var1,var10,var2,var3'
    assert 'res3' not in results.cells

    missing_dep = """
    from damnit_ctx import Variable

    @Variable()
    def foo(run, bar: 'var#bar'):
        return bar
    """
    with pytest.raises(KeyError, match='bar'):
        mkcontext(missing_dep)

    missing_dep_pattern = """
    from damnit_ctx import Variable

    @Variable()
    def foo(run, bar: 'var#bar*'):
        return bar
    """
    with pytest.raises(KeyError, match='bar*'):
        ctx = mkcontext(missing_dep_pattern)

    single_match_pattern = """
    from damnit_ctx import Variable

    @Variable()
    def foo(run, data: 'var#b?r'):
        return data['bar']

    @Variable()
    def bar(run):
        return 42
    """
    ctx = mkcontext(single_match_pattern)
    results = ctx.execute(mock_run, 1000, 123, {})

    assert results.cells['foo'].data == 42
