import configparser
import sys
import threading
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

import h5py
import numpy as np
import plotly.express as px
import pytest
import xarray as xr
from PIL import Image
from testpath import MockCommand

from damnit.backend import listener_is_running, initialize_proposal, start_listener
from damnit.backend.db import DamnitDB
from damnit.backend.extract_data import Extractor, RunExtractor, add_to_db, load_reduced_data, main as extract_data_main
from damnit.backend.extraction_control import ExtractionJobTracker
from damnit.backend.listener import (MAX_CONCURRENT_THREADS, EventProcessor,
                                     local_extraction_threads)
from damnit.backend.supervisord import wait_until, write_supervisord_conf
from damnit.context import (ContextFile, ContextFileErrors, PNGData, RunData,
                            get_proposal_path)
from damnit.ctxsupport.ctxrunner import THUMBNAIL_SIZE, add_to_h5_file
from damnit.ctxsupport.damnit_ctx import Group, is_group, is_group_instance
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

    with (
        pytest.MonkeyPatch.context() as mp,
        patch("extra_proposal.proposal.find_proposal", return_value=tmp_path),
    ):
        from extra_proposal.proposal import RunReference
        mp.setattr(RunReference, 'sample_name', lambda self: "mithril")
        mp.setattr(RunReference, 'run_type', lambda self: "alchemy")
        mp.setattr(RunReference, 'techniques', lambda self: [
            {'identifier': 'PaNET01168', 'name': 'SFX'},
            {'identifier': 'PaNET01188', 'name': 'SAXS'},
        ])

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

    @Variable
    def var3(run):
        return Cell('asdf', preview=np.ones(5, dtype=np.bool_))
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
        assert f['.preview/var3'].dtype == bool

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
        "image": b'\x89PNG\r\n\x1a\n...',  # Not a valid PNG, but good enough for this
        "nested.var": "Asdf",
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

    ctxrunner_deps = ["extra_data", "extra_proposal", "matplotlib", "plotly"]

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

    with patch(f"{pkg}.KafkaProducer"), pytest.raises(RuntimeError, match="sfollow"):
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


def test_listener(tmp_path, caplog, monkeypatch):
    monkeypatch.setenv("XFEL_DATA_ROOT", str(tmp_path))

    # Create the processor
    with patch('damnit.backend.listener.KafkaConsumer') as kcon:
        processor = EventProcessor(tmp_path)
        # Default: do not allow local processing
        processor.db.settings["allow_local_processing"] = False

    kcon.assert_called_once()
    assert len(local_extraction_threads) == 0

    # Create an 'official' database where find_proposal() will find it
    proposal_dir = tmp_path / "MID" / "202501" / "p001234"
    proposal_dir.mkdir(parents=True)
    db_dir = proposal_dir / "usr/Shared/amore"
    initialize_proposal(db_dir, 1234)
    db = DamnitDB.from_dir(db_dir)
    db.metameta["context_python"] = sys.executable

    # First event: static_mode == True -> ignore event
    event = MagicMock(timestamp=time())
    processor.handle_event(event, {"proposal": 1234, "run": 1}, RunData.RAW)
    assert len(processor.db.proposal_db_dirs(1234)) == 0
    assert len(local_extraction_threads) == 0

    # Disable static mode: expect 1 Slurm submission (official DB)
    processor.db.settings["static_mode"] = False
    with patch("damnit.backend.extraction_control.ExtractionSubmitter.submit",
               return_value=("9876", "solaris")) as submit:
        processor.handle_event(event, {"proposal": 1234, "run": 1}, RunData.RAW)
        assert submit.call_count == 1

    # Add an unofficial DB so that the listener launches two Slurm jobs
    fake_db_dir = tmp_path / "fakedb"
    initialize_proposal(fake_db_dir, 1234)
    fake_db = DamnitDB.from_dir(fake_db_dir)
    fake_db.metameta["context_python"] = sys.executable
    processor.db.add_proposal_db(1234, fake_db_dir, False)

    with patch("damnit.backend.extraction_control.ExtractionSubmitter.submit",
               return_value=("9999", "solaris")) as submit2:
        processor.handle_event(event, {"proposal": 1234, "run": 1}, RunData.RAW)
        # Two DBs -> two submissions
        assert submit2.call_count == 2

    # With a non-existent proposal specified handle_event() should do nothing
    with (
        caplog.at_level(logging.WARNING),
        patch("damnit.backend.extraction_control.ExtractionSubmitter.submit") as submit3
    ):
        processor.handle_event(event, {'proposal': 4321, 'run': 1}, RunData.RAW)
    assert "Could not find proposal directory" in caplog.text
    assert len(local_extraction_threads) == 0
    assert submit3.call_count == 0

    # Test removing a database
    processor.db.remove_proposal_db(fake_db_dir)
    assert processor.db.proposal_db_dirs(1234) == [db_dir]


def test_listener_local(tmp_path, caplog, monkeypatch):
    """When Slurm fails and local fallback is enabled, listener runs jobs locally
    """
    monkeypatch.setenv("XFEL_DATA_ROOT", str(tmp_path))

    # Create the processor
    with patch('damnit.backend.listener.KafkaConsumer'):
        processor = EventProcessor(tmp_path)
        processor.db.settings["allow_local_processing"] = True

    # Create an 'official' database where find_proposal() will find it
    proposal_dir = tmp_path / "MID" / "202501" / "p001234"
    proposal_dir.mkdir(parents=True)
    db_dir = proposal_dir / "usr/Shared/amore"
    initialize_proposal(db_dir, 1234)
    db = DamnitDB.from_dir(db_dir)
    db.metameta["context_python"] = sys.executable

    # Disable static mode so the proposal DB is tracked
    processor.db.settings["static_mode"] = False

    # Simulate Slurm failure and local execution that takes a moment
    jobs = []
    state = {'active': 0, 'peak': 0}
    lock = threading.Lock()

    def fake_exec_direct(self, req):
        with lock:
            state['active'] += 1
            state['peak'] = max(state['peak'], state['active'])
        try:
            jobs.append((req.proposal, req.run, req.run_data.value))
            sleep(1)
        finally:
            with lock:
                state['active'] -= 1

    event = MagicMock(timestamp=time())

    with (
        patch("damnit.backend.extraction_control.ExtractionSubmitter.submit", side_effect=Exception("sbatch failed")),
        patch("damnit.backend.extraction_control.ExtractionSubmitter.execute_direct", new=fake_exec_direct),
    ):
        with caplog.at_level(logging.ERROR):
            processor.handle_event(event, {"proposal": 1234, "run": 1}, RunData.RAW)

        # Wait for any local jobs to finish
        for th in local_extraction_threads:
            th.join()
        assert len(jobs) == 1  # Only the official DB so far
        assert 'Slurm job submission failed' in caplog.text

        # Add an unofficial DB so that the listener launches two local jobs
        fake_db_dir = tmp_path / "fakedb"
        initialize_proposal(fake_db_dir, 1234)
        fake_db = DamnitDB.from_dir(fake_db_dir)
        fake_db.metameta["context_python"] = sys.executable
        processor.db.add_proposal_db(1234, fake_db_dir, False)

        caplog.clear()
        processor.handle_event(event, {"proposal": 1234, "run": 1}, RunData.RAW)
        for th in local_extraction_threads:
            th.join()
        assert len(jobs) == 3  # +2 jobs for official + unofficial

        # Test processing too many runs concurrently
        jobs.clear()
        state['active'] = 0
        state['peak'] = 0
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            for idx in range(MAX_CONCURRENT_THREADS + 1):
                event = MagicMock(timestamp=time())
                processor.handle_event(event, {'proposal': 1234, 'run': idx + 1}, RunData.RAW)

        # Wait for all threads to finish, then assert peak concurrency and total jobs
        for th in local_extraction_threads:
            th.join()
        assert state['peak'] == MAX_CONCURRENT_THREADS
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


def test_extract_data_sandbox(mock_db, tmp_path, monkeypatch):
    """extract_data should invoke the sandbox for whoami and for exec payload.

    We verify both invocations and that the wrapper forwards to the payload (so
    processing completes) by creating a tiny sandbox script that logs and execs.
    """
    db_dir, db = mock_db
    # Ensure context loads using an available interpreter
    db.metameta["context_python"] = sys.executable
    db.metameta["proposal"] = 1234
    monkeypatch.chdir(db_dir)

    # Create a logging + forwarding sandbox script
    log_path = tmp_path / "sandbox_calls.log"
    script_path = tmp_path / "sandbox.sh"
    script_path.write_text("""\
#! /usr/bin/env bash
log="$1"; shift
(
  flock -x 200
  echo "$@" >> "$log"
) 200>"$log.lock"
# Skip arguments until the separator then exec the payload
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done
if [ "$1" = "--" ]; then shift; fi
exec "$@"
"""
    )
    script_path.chmod(0o755)

    # Run extract_data.main with sandbox args
    pkg = "damnit.backend.extract_data"
    with patch(f"{pkg}.KafkaProducer"):
        extract_data_main([
            "1234", "1", "all",
            "--mock",
            "--sandbox-args", f"{script_path} {log_path}",
        ])

    # Check that sandbox was called for whoami, then ctx (context inspection), and exec
    lines = log_path.read_text().splitlines()
    assert len(lines) >= 3
    # First invocation should be the whoami probe
    assert lines[0].endswith("whoami")
    # Second invocation should run ctxrunner ctx for the requested proposal/run
    assert " ctxrunner " in lines[1]
    assert " ctx " in lines[1]
    assert "1234" in lines[1]
    # Third invocation should run ctxrunner exec for the requested proposal/run
    assert " ctxrunner " in lines[2]
    assert " exec " in lines[2]
    assert " 1234 " in lines[2]
    assert " 1 " in lines[2]


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

    @Variable(transient=True)
    def var5(run):
        raise NotImplementedError

    @Variable()
    def summary(run, data: 'var#var?'):
        return len(data)
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

    attrs = db.conn.execute(
        "SELECT attributes FROM run_variables WHERE name='summary'"
    ).fetchone()[0]
    data = json.loads(attrs)
    assert '(var5)' in data['error']
    assert '(var3)' in data['error']


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
    my_group = TestGroup(name='my_group', title="My Test Group", calibration_factor=1.5, test_value=5, tags="test_group")
    your_group = TestGroup(2.0, 10, name='your_group')
    """
    ctx = mkcontext(code)

    # Test variable naming and structure
    assert len(ctx.vars) == 6
    assert "my_group.raw_value" in ctx.vars
    assert "my_group.calibrated_value" in ctx.vars
    assert "my_group.offset_value" in ctx.vars
    assert "your_group.raw_value" in ctx.vars
    assert "your_group.calibrated_value" in ctx.vars
    assert "your_group.offset_value" in ctx.vars

    # Test title generation
    assert ctx.vars["my_group.raw_value"].title == "My Test Group/Raw Value"
    assert ctx.vars["my_group.calibrated_value"].title == "My Test Group/calibrated_value"
    assert ctx.vars["your_group.offset_value"].title == "your_group/Offset Value"

    # Test tag merging
    assert "test_group" in ctx.vars["my_group.raw_value"].tags
    assert "raw" in ctx.vars["my_group.raw_value"].tags

    # Check topological ordering for dependency resolution
    ordered = ctx.ordered_vars()
    assert ordered.index("my_group.raw_value") < ordered.index("my_group.calibrated_value")

    # Execute and check results
    results = run_ctx_helper(ctx, mock_run, 1000, 1234, caplog, input_vars={"offset": 10})
    assert results.cells["my_group.raw_value"].data == 10
    # Check that calibration factor from group __init__ was used
    assert results.cells["my_group.calibrated_value"].data == 15.0
    # Check that input# variables are passed through correctly
    assert results.cells["my_group.offset_value"].data == 57.0

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

    error_group = ErrorGroup(title="Error Group")
    """
    ctx_err = mkcontext(error_code)

    with caplog.at_level(logging.ERROR):
        results_err = ctx_err.execute(mock_run, 1000, 1234, {})

    # Check that an error was logged for bad_var
    assert caplog.records[0].levelname == "ERROR"

    assert "ErrorGroup.good_var" not in results_err.cells
    assert "ErrorGroup.good_var" not in results_err.errors
    # The error itself should be recorded
    assert "ErrorGroup.bad_var" in results_err.errors

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

    proc_b = ProcessingGroup(name='proc_b', title="Processor B", offset=-3)
    proc_a = ProcessingGroup(name='proc_a', title="Processor A")

    # A variable that depends on outputs from both group instances
    @Variable(title="A vs B")
    def a_vs_b_diff(run, a_result: "var#proc_a.step2", b_result: "var#proc_b.step2"):
        return a_result - b_result
    """
    ctx = mkcontext(code)

    # Check that all variable names are generated correctly
    expected_vars = {
        "base",
        "proc_a.base", "proc_a.step1", "proc_a.step2",
        "proc_b.base", "proc_b.step1", "proc_b.step2",
        "a_vs_b_diff"
    }
    assert set(ctx.vars.keys()) == expected_vars

    # Check the dependency graph and execution order
    ordered_vars = ctx.ordered_vars()

    assert ordered_vars.index("base") < ordered_vars.index("proc_a.step1")
    assert ordered_vars.index("base") < ordered_vars.index("proc_b.step1")

    assert ordered_vars.index("proc_a.step1") < ordered_vars.index("proc_a.step2")
    assert ordered_vars.index("proc_b.step1") < ordered_vars.index("proc_b.step2")

    assert ordered_vars.index("proc_a.step2") < ordered_vars.index("a_vs_b_diff")
    assert ordered_vars.index("proc_b.step2") < ordered_vars.index("a_vs_b_diff")

    # Execute the context and check the results
    results = run_ctx_helper(ctx, mock_run, 1000, 1234, caplog)

    # Check results for proc_a
    assert results.cells["proc_a.step1"].data == 101
    assert results.cells["proc_a.step2"].data == 252

    # Check results for proc_b
    assert results.cells["proc_b.step1"].data == 101
    assert results.cells["proc_b.step2"].data == 249

    # Check the final composer variable
    assert results.cells["a_vs_b_diff"].data == 3

    results_hdf5_path = tmp_path / "results.hdf5"
    results.save_hdf5(results_hdf5_path)
    with h5py.File(results_hdf5_path) as f:
        assert f["proc_a.step1/data"][()] == 101
        assert f["proc_a.step2/data"][()] == 252
        assert f["proc_b.step1/data"][()] == 101
        assert f["proc_b.step2/data"][()] == 249
        assert f["a_vs_b_diff/data"][()] == 3
        assert "base" in f
        assert "proc_a.base" not in f

    # Test that a missing root dependency is handled correctly
    bad_root_code = """
    from damnit_ctx import Variable, Group

    @Group
    class BadRootGroup:
        @Variable()
        def step1(self, run, base: "var#missing_global"):
            return 1

    bad_group = BadRootGroup(title="Bad Group")
    """
    with pytest.raises(KeyError, match="missing_global"):
        mkcontext(bad_root_code)

    # Test that a missing dependency is handled correctly
    bad_root_code = """
    from damnit_ctx import Variable, Group

    @Group
    class BadRootGroupLocal:
        @Variable()
        def step1(self, run, base: "self#missing_local"):
            return 1

    bad_group = BadRootGroupLocal(title="Bad Group")
    """
    with pytest.raises(AttributeError, match="missing_local"):
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

    base_group = BaseGroup(title="Base Group")
    child_group = ChildGroup(title="Child Group", value=1, tags=['Child', 'other tag'])
    child_group2 = ChildGroup2(title="Child Group 2")
    """
    ctx = mkcontext(code)
    assert set(ctx.vars) == {"BaseGroup.step1", "ChildGroup2.step1", "ChildGroup.step1", "ChildGroup.step2"}

    assert ctx.vars["BaseGroup.step1"].tags is None
    assert set(ctx.vars["ChildGroup.step1"].tags) == {"Child", "other tag"}
    assert set(ctx.vars["ChildGroup.step2"].tags) == {"Child", "other tag"}

    results = run_ctx_helper(ctx, mock_run, 1000, 1234, caplog)
    assert results.cells["ChildGroup.step2"].data == 4
    assert results.cells["ChildGroup2.step1"].data == 'overridden step1'

    # Test Group references in dependencies
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class XGMGroup:
        factor: float = 1.0

        @Variable(title="Intensity")
        def intensity(self, run):
            return 10 * self.factor

    @Group
    class AGIPD4M:
        @Variable(title="Image", transient=True)
        def image(self, run):
            import numpy as np
            return np.ones((100, 100), dtype=np.uint8)

        @Variable(summary="sum", title='Photon Count', tags=['photons'])
        def photon_count(self, run, image_data: "self#image"):
            # A mock calculation
            return image_data.sum()

    @Group(title="SCS Instrument", tags=["scs"])
    class InstrumentDiagnostics:
        # group instances reference
        xgm: XGMGroup
        det: AGIPD4M

        @Variable(title="Intensity per Photon")
        def intensity_per_photon(self, run,
                                 intensity: "self#xgm.intensity",
                                 photons: "self#det.photon_count"):
            # This depends on variables from two different nested subgroups
            return intensity / photons

    @Group
    class Experiment:
        diagnostics: InstrumentDiagnostics

        @Variable(title="Experiment Quality")
        def quality(self, run, ipp: "self#diagnostics.intensity_per_photon"):
            # Depends on a variable from a nested group
            return "good" if ipp > 1.0 else "bad"

        @Variable()
        def deep_nested_access(run, data: "self#diagnostics.det.image"):
            return data.mean()

    xgm_upstream = XGMGroup(name="xgm_upstream", title="XGM", factor=1.5)
    agipd4m = AGIPD4M(name="agipd4m", title="Detector", tags="detector")

    diag = InstrumentDiagnostics(name="diag", xgm=xgm_upstream, det=agipd4m)

    # Instantiate the top-level group
    exp1 = Experiment(name="exp1", title="My Experiment", diagnostics=diag)

    @Variable()
    def nested_var_access(run, data: "var#agipd4m.image"):
        return data.mean()
    """
    ctx = mkcontext(code)

    # Check that all variables from the nested structure are present and named correctly
    expected_vars = {
        "xgm_upstream.intensity",
        "agipd4m.image",
        "agipd4m.photon_count",
        "diag.intensity_per_photon",
        "exp1.quality",
        "exp1.deep_nested_access",
        "nested_var_access",
    }
    assert set(ctx.vars.keys()) == expected_vars

    # Check title generation with multiple levels of nesting
    assert ctx.vars["xgm_upstream.intensity"].title == "XGM/Intensity"
    assert ctx.vars["agipd4m.image"].title == "Detector/Image"
    assert ctx.vars["agipd4m.photon_count"].title == "Detector/Photon Count"
    assert ctx.vars["diag.intensity_per_photon"].title == "SCS Instrument/Intensity per Photon"
    assert ctx.vars["exp1.quality"].title == "My Experiment/Experiment Quality"

    # Check that tags are correctly inherited
    assert "detector" in ctx.vars["agipd4m.photon_count"].tags
    assert "photons" in ctx.vars["agipd4m.photon_count"].tags

    # Check dependency resolution and execution order
    ordered_vars = ctx.ordered_vars()
    # Check dependencies within the deepest group
    assert ordered_vars.index("agipd4m.image") < ordered_vars.index("agipd4m.photon_count")
    # Check dependencies for the mid-level composer variable
    assert ordered_vars.index("xgm_upstream.intensity") < ordered_vars.index("diag.intensity_per_photon")
    assert ordered_vars.index("agipd4m.photon_count") < ordered_vars.index("diag.intensity_per_photon")
    # Check dependencies for the top-level composer variable
    assert ordered_vars.index("diag.intensity_per_photon") < ordered_vars.index("exp1.quality")

    # Execute the context and verify the results
    results = run_ctx_helper(ctx, mock_run, 1000, 1234, caplog)

    # Verify intermediate and final calculations
    assert results.cells["xgm_upstream.intensity"].data == 15.0  # 10 * 1.5
    assert results.cells["agipd4m.photon_count"].data == 10_000
    assert results.cells["diag.intensity_per_photon"].data == 15.0 / 10_000
    assert results.cells["exp1.quality"].data == "bad"
    assert results.cells["exp1.deep_nested_access"].data == 1.0

    # Check that transient variables are not saved
    results_hdf5_path = tmp_path / "results.h5"
    results.save_hdf5(results_hdf5_path)
    with h5py.File(results_hdf5_path) as f:
        # The transient 'image' variable should not exist in the file
        assert "agipd4m.image" not in f
        # A non-transient variable should exist
        assert "exp1.quality" in f
        assert f["exp1.quality/data"][()] == b"bad"

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

    a = A(name='a')
    aa = A(name='aa', title='AA', tags='AA')
    b = B(name='b')
    c = C(name='c')
    d = D(name='d')
    dd = DD(name='dd')
    dd2 = DD(name='dd2', title='DD2', tags='DD2')
    """
    ctx = mkcontext(code_group_settings)

    # Check that the group settings are applied correctly
    assert ctx.vars["a.var1"].title == "A/var1"
    assert set(ctx.vars["a.var1"].tags) == {'A'}
    assert ctx.vars["a.var1"].cluster is False
    assert ctx.vars["a.var1"].transient is True
    assert ctx.vars["a.var1"].data == RunData.RAW

    assert ctx.vars["aa.var1"].title == "AA/var1"
    assert set(ctx.vars["aa.var1"].tags) == {"AA"}
    assert ctx.vars["aa.var1"].cluster is False
    assert ctx.vars["aa.var1"].transient is True
    assert ctx.vars["aa.var1"].data == RunData.RAW

    assert ctx.vars["b.var1"].title == "A/var1"
    assert set(ctx.vars["b.var1"].tags) == {'A'}
    assert ctx.vars["b.var1"].cluster is False
    assert ctx.vars["b.var1"].transient is True
    assert ctx.vars["b.var1"].data == RunData.RAW

    assert ctx.vars["b.var2"].title == "A/b"
    assert set(ctx.vars["b.var2"].tags) == {"A", "B",}
    assert ctx.vars["b.var2"].cluster is True
    assert ctx.vars["b.var2"].transient is False
    assert ctx.vars["b.var2"].data == RunData.RAW

    assert ctx.vars["b.var3"].title == "A/var3"
    assert set(ctx.vars["b.var3"].tags) == {'A'}
    assert ctx.vars["b.var3"].cluster is True  # promoted to cluster as it dependes on a cluster variable
    assert ctx.vars["b.var3"].transient is False
    assert ctx.vars["b.var3"].data == RunData.RAW

    assert ctx.vars["c.var1"].title == "c/var1"
    assert ctx.vars["c.var1"].tags is None
    assert ctx.vars["c.var1"].cluster is False

    assert ctx.vars["d.d_var"].title == "D/d_var"
    assert set(ctx.vars["d.d_var"].tags) == {"D", "D_var"}
    assert ctx.vars["d.d_var"].cluster is True
    assert ctx.vars["d.d_var"].data == RunData.RAW

    assert ctx.vars["dd.d_var"].title == "dd/d_var"
    assert set(ctx.vars["dd.d_var"].tags) == {"DD", "D_var"}
    assert ctx.vars["dd.d_var"].cluster is True
    assert ctx.vars["dd.dd_var"].title == "dd/DD_var"
    assert set(ctx.vars["dd.dd_var"].tags) == {"DD", "DD_var"}
    assert ctx.vars["dd.dd_var"].cluster is True  # promoted to cluster as it dependes on a cluster variable
    assert ctx.vars["dd.dd_var"].data == RunData.PROC
    assert ctx.vars["dd.dd_var2"].title == "dd/dd_var2"
    assert set(ctx.vars["dd.dd_var2"].tags) == {"DD"}
    assert ctx.vars["dd.dd_var2"].cluster is True  # promoted to cluster as it dependes on a cluster variable
    assert ctx.vars["dd.dd_var2"].data == RunData.PROC

    assert ctx.vars["dd2.dd_var"].title == "DD2/DD_var"
    assert set(ctx.vars["dd2.dd_var"].tags) == {"DD2", "DD_var"}
    assert ctx.vars["dd2.dd_var"].cluster is True  # promoted to cluster as it dependes on a cluster variable
    assert ctx.vars["dd2.dd_var2"].title == "DD2/dd_var2"
    assert set(ctx.vars["dd2.dd_var2"].tags) == {"DD2"}
    assert ctx.vars["dd2.dd_var2"].cluster is True  # promoted to cluster as it dependes on a cluster variable

    code_fields = """
    from dataclasses import field

    import numpy as np
    from damnit_ctx import Variable, Group

    @Group
    class FieldsGroup:
        field1: int = 2
        field2: str = "default"
        field3: float = 1.0
        field4: bool = True
        field5: list = field(default_factory=list)
        field6: dict = field(default_factory=dict)
        field7: tuple = (1, 2, 3)
        field8: set = field(default_factory=lambda: {1, 2, 3})

        @Variable(title="Field 1 Variable")
        def field1_var(self, run):
            return self.field1 + 1

        @Variable(title="Field 2 Variable")
        def field2_var(self, run):
            return self.field2.upper()

        @Variable(title="Field 3 Variable")
        def field3_var(self, run):
            return self.field3 * 2.0

        @Variable(title="Field 4 Variable")
        def field4_var(self, run):
            return not self.field4

        @Variable(title="Field 5 Variable")
        def field5_var(self, run):
            return [x * 2 for x in self.field5]

        @Variable(title="Field 6 Variable")
        def field6_var(self, run):
            return np.asarray([v * 2 for v in self.field6.values()])

        @Variable(title="Field 7 Variable")
        def field7_var(self, run):
            return tuple(x * 2 for x in self.field7)

        @Variable(title="Field 8 Variable")
        def field8_var(self, run):
            return [x * 2 for x in self.field8]

    class ChildGroup(FieldsGroup):
        field9: int = 40

        @Variable()
        def field9_var(self, run):
            return self.field9 + self.field1

    @Group
    class ChildGroup2(ChildGroup):
        field10: int = 2

        @Variable()
        def field10_var(self, run):
            return self.field10 + self.field9

    fields_group = FieldsGroup(
        name='fields_group',
        title="Fields Group",
        field1=10,
        field2="test",
        field3=3.5,
        field4=False,
        field5=[1, 2, 3],
        field6={'a': 1, 'b': 2},
        field7=(4, 5),
        field8={6, 7}
    )
    child_group = ChildGroup(name='child_group')
    child_group2 = ChildGroup2(name='child_group2', title="Child Group 2", tags=['c2'], field1=0)
    """
    ctx = mkcontext(code_fields)
    # Check that all fields are correctly initialized and variables are created
    for i in range(1, 9):
        assert ctx.vars[f"fields_group.field{i}_var"].title == f"Fields Group/Field {i} Variable"

    assert ctx.vars["fields_group.field1_var"].tags is None
    assert ctx.vars["child_group.field9_var"].tags is None
    assert set(ctx.vars["child_group2.field1_var"].tags) == {'c2'}
    assert set(ctx.vars["child_group2.field10_var"].tags) == {'c2'}

    # Check that field values are correctly set
    results = run_ctx_helper(ctx, mock_run, 1000, 1234, caplog)
    assert results.cells["fields_group.field1_var"].data == 11
    assert results.cells["fields_group.field2_var"].data == "TEST"
    assert results.cells["fields_group.field3_var"].data == 7.0
    assert results.cells["fields_group.field4_var"].data.tolist() is True
    assert results.cells["fields_group.field5_var"].data.tolist() == [2, 4, 6]
    assert results.cells["fields_group.field6_var"].data.tolist() == [2, 4]
    assert results.cells["fields_group.field7_var"].data.tolist() == [8, 10]
    assert results.cells["fields_group.field8_var"].data.tolist() == [12, 14]

    assert results.cells["child_group.field9_var"].data == 42

    assert results.cells["child_group2.field10_var"].data == 42

    redefined_default_field_code = """
    from damnit_ctx import Variable, Group

    @Group
    class G:
        tags: str = 'G'
    """
    with pytest.raises(TypeError, match="tags"):
        ctx = mkcontext(redefined_default_field_code)

    redefined_field_code = """
    from damnit_ctx import Variable, Group

    @Group
    class BaseGroup:
        field1: int = 10

    @Group
    class DerivedGroup(BaseGroup):
        field1: int = 20  # Redefine field1

    base = BaseGroup(name='base')
    derived = DerivedGroup(name='derived')

    assert base.field1 == 10
    assert derived.field1 == 20
    """
    ctx = mkcontext(redefined_field_code)

    code_tags = """
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

    a = A(name='a')
    b = B(name='b')
    c = C(name='c')
    d = D(name='d')
    e = E(name='e')
    f = F(name='f')
    aa = A(name='aa', title='A Group', tags=['group_a'])
    bb = B(name='bb', title='B Group', tags=['group_b'])
    cc = C(name='cc', title='C Group', tags=['group_c'])
    dd = D(name='dd', title='D Group', tags=['group_d'])
    ee = E(name='ee', title='E Group', tags=['group_e'])
    ff = F(name='ff', title='F', tags=['group_f'])
    """
    ctx = mkcontext(code_tags)

    assert ctx.vars["a.a_var"].title == "a/A Variable"
    assert ctx.vars["b.a_var"].title == "b/A Variable"
    assert ctx.vars["c.a_var"].title == "c/C Variable"
    assert ctx.vars["d.a_var"].title == "d/A Variable"
    assert ctx.vars["e.a_var"].title == "e/A Variable"
    assert ctx.vars["f.a_var"].title == "F Group/A Variable"

    assert ctx.vars["aa.a_var"].title == "A Group/A Variable"
    assert ctx.vars["bb.a_var"].title == "B Group/A Variable"
    assert ctx.vars["cc.a_var"].title == "C Group/C Variable"
    assert ctx.vars["dd.a_var"].title == "D Group/A Variable"
    assert ctx.vars["ee.a_var"].title == "E Group/A Variable"
    assert ctx.vars["ff.a_var"].title == "F/A Variable"

    assert set(ctx.vars["a.a_var"].tags) == {"varA", "ClassA"}
    assert set(ctx.vars["b.a_var"].tags) == {"varA", "ClassA"}
    assert set(ctx.vars["b.b_var"].tags) == {"varB", "ClassA"}
    assert set(ctx.vars["c.a_var"].tags) == {"varC", "ClassA"}
    assert set(ctx.vars["d.a_var"].tags) == {"varA", "ClassA"}
    assert set(ctx.vars["e.a_var"].tags) == {"varA"}
    assert set(ctx.vars["f.a_var"].tags) == {"varA", "ClassF"}

    assert set(ctx.vars["aa.a_var"].tags) == {"varA", "group_a"}
    assert set(ctx.vars["bb.a_var"].tags) == {"varA", "group_b"}
    assert set(ctx.vars["cc.a_var"].tags) == {"varC", "group_c"}
    assert set(ctx.vars["dd.a_var"].tags) == {"varA", "group_d"}
    assert set(ctx.vars["ee.a_var"].tags) == {"varA", "group_e"}
    assert set(ctx.vars["ff.a_var"].tags) == {"varA", "group_f"}

    code_custom_sep = """
    from damnit_ctx import Variable, Group

    @Group(title="My Analysis", sep=" -> ")
    class CustomSepGroup:
        @Variable(title="Step 1")
        def step1(self, run):
            return 1

    my_group = CustomSepGroup(name='my_group')
    """
    ctx = mkcontext(code_custom_sep)
    assert ctx.vars["my_group.step1"].title == "My Analysis -> Step 1"

    code_empty_group = """
    from damnit_ctx import Group

    @Group(title="Empty")
    class EmptyGroup:
        pass

    empty = EmptyGroup()
    """
    ctx = mkcontext(code_empty_group)
    assert len(ctx.vars) == 0

    code_invalid_group_argument = """
    from damnit_ctx import Group

    @Group(unknown_property="value")
    class UnknownGroup:
        pass
    """
    with pytest.raises(TypeError, match="unknown_property"):
        ctx = mkcontext(code_invalid_group_argument)

    # test helper functions
    @Group
    class A:
        pass

    assert is_group(A)
    assert is_group(A())
    assert not is_group_instance(A)
    assert is_group_instance(A())

    # test group linking with string attributes
    code_linking = """
    from damnit_ctx import Variable, Group

    @Group(title="Shared Component")
    class SharedComponent:
        base_value: int = 0

        @Variable(title="Shared Value")
        def shared_value(self, run):
            return self.base_value

    @Group(title="Linking Group")
    class LinkingGroup:
        # holds the name of a SharedComponent instance.
        source: SharedComponent

        @Variable(title="Processed Value")
        def processed_value(self, run, data: "self#source.shared_value"):
            return data + 1

    # Define two separate, shared instances
    shared1 = SharedComponent(name='shared1', base_value=100)
    shared2 = SharedComponent(name='shared2', base_value=200)

    # Link two different groups to the two different shared instances
    linker_a = LinkingGroup(name='linker_a', source=shared1)
    linker_b = LinkingGroup(name='linker_b', source=shared2)
    """
    ctx = mkcontext(code_linking)

    expected_vars = {
        "shared1.shared_value",
        "shared2.shared_value",
        "linker_a.processed_value",
        "linker_b.processed_value",
    }
    assert set(ctx.vars.keys()) == expected_vars

    ordered_vars = ctx.ordered_vars()
    assert ordered_vars.index("shared1.shared_value") < ordered_vars.index("linker_a.processed_value")
    assert ordered_vars.index("shared2.shared_value") < ordered_vars.index("linker_b.processed_value")

    results = run_ctx_helper(ctx, mock_run, 1000, 1234, caplog)

    # linker_a should be linked to shared1
    assert results.cells["shared1.shared_value"].data == 100
    assert results.cells["linker_a.processed_value"].data == 101

    # linker_b should be linked to shared2
    assert results.cells["shared2.shared_value"].data == 200
    assert results.cells["linker_b.processed_value"].data == 201


    # test Group variables are dropped if optional dependency is missing
    code_linking_missing = """
    from damnit_ctx import Variable, Group

    @Group(title="Linking Group")
    class LinkingGroup:
        source: Group = None

        @Variable(title="Processed Value")
        def processed_value(self, run, data: "self#source.some_var"):
            return 1

        @Variable
        def another_processed_value(self, run):
            return 'the answer'

    # Instantiate the group WITHOUT providing the required 'source'
    linker_a = LinkingGroup()
    """
    ctx = mkcontext(code_linking_missing)
    assert set(ctx.vars.keys()) == {'LinkingGroup.another_processed_value'}


    # test fields with no default argument
    code_nodefault = """
    from damnit_ctx import Variable, Group

    @Group(title="No Default Group")
    class NoDefaultGroup:
        arg0: int
        arg1: str

        @Variable(title="No Default Var")
        def no_default_var(self, run):
            return 1 + self.arg0

    no_default = NoDefaultGroup(123, arg1='test')
    """
    ctx = mkcontext(code_nodefault)
    results = run_ctx_helper(ctx, mock_run, 1000, 1234, caplog)
    assert results.cells["NoDefaultGroup.no_default_var"].data == 124

    # test that a Group argument can be None
    code_default_group_field = """
    from damnit_ctx import Variable, Group

    @Group
    class A:
        @Variable
        def var(self, run):
            return 41

    @Group
    class B:
        a: A = None

        @Variable
        def v1(self, run, a: 'self#a.var'):
            return a + 1

        @Variable
        def v2(self, run, a: 'self#a.var' = 42):
            return a + 1

    a = A(name='a')
    b1 = B(name='b1', a=a)
    b2 = B(name='b2')  # a is None
    """
    ctx = mkcontext(code_default_group_field)

    expected_vars = {
        "a.var",
        "b1.v1",
        "b1.v2",
        # b2.v1 is removed as it's group dependency is None
        "b2.v2",  # group deps is None, but has a default argument
    }
    assert set(ctx.vars.keys()) == expected_vars

    results = run_ctx_helper(ctx, mock_run, 1000, 1234, caplog)
    assert results.cells["b1.v1"].data == 42
    assert results.cells["b1.v2"].data == 42
    assert results.cells["b2.v2"].data == 43
