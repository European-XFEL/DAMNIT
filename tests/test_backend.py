import logging
import graphlib
import textwrap
import tempfile
from pathlib import Path
from functools import partial
from unittest.mock import patch

import h5py
import pytest
import numpy as np

from amore_mid_prototype.context import ContextFile
from amore_mid_prototype.backend.extract_data import Results, RunData, run_and_save, get_proposal_path


def test_dag(mock_ctx):
    # Helper lambda to get the names of the direct dependencies of a variable
    var_deps = lambda name: set(mock_ctx.vars[name].arg_dependencies().values())
    # Helper lambda to get the names of all dependencies of a variable
    all_var_deps = lambda name: mock_ctx.all_dependencies(mock_ctx.vars[name])

    # Check that each variable has the right dependencies
    assert { "scalar1" } == var_deps("scalar2")
    assert { "scalar1", "scalar2" } == var_deps("array")
    assert { "array", "timestamp" } == var_deps("meta_array")

    # Check that the ordering is correct for execution
    assert mock_ctx.ordered_vars() == ("scalar1", "timestamp", "string", "scalar2", "array", "meta_array")

    # Check that we can retrieve direct and indirect dependencies
    assert set() == all_var_deps("scalar1")
    assert { "scalar1" } == all_var_deps("scalar2")
    assert { "scalar1", "scalar2" } == all_var_deps("array")
    assert { "array", "timestamp", "scalar1", "scalar2" } == all_var_deps("meta_array")

    # Create a context file with a cycle
    bad_code = """
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
        ContextFile.from_str(textwrap.dedent(bad_code))

def test_create_context_file():
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

def test_results(mock_ctx, mock_run, caplog):
    run_number = 1000
    proposal = 1234

    # Track all error messages during creation. This is necessary because a
    # variable that throws an error will be logged by Results, the exception
    # will not bubble up.
    with caplog.at_level(logging.ERROR):
        results = Results.create(mock_ctx, mock_run, run_number, proposal)

    # Check that there were no errors and all variables were executed
    assert len(caplog.records) == 0
    assert set(mock_ctx.ordered_vars()) <= results.data.keys()

    # Check the result values
    assert results.data["scalar1"] == 42
    assert results.data["scalar2"] == 3.14
    np.testing.assert_equal(results.data["array"], [42, 3.14])
    np.testing.assert_equal(results.data["meta_array"], [run_number, proposal])
    assert results.data["string"] == str(get_proposal_path(mock_run))

def test_run_and_save(mock_ctx, mock_run):
    run_number = 1000
    proposal = 1234

    ed_module_path = "amore_mid_prototype.backend.extract_data"
    with (patch(f"{ed_module_path}.extra_data.open_run", side_effect=lambda *_, **__: mock_run),
          tempfile.NamedTemporaryFile() as hdf5_file,
          tempfile.NamedTemporaryFile() as ctx_file):
        # Save the context code
        ctx_file.write(mock_ctx.code.encode())
        ctx_file.flush()

        def var_datasets():
            with h5py.File(hdf5_file.name) as f:
                datasets = set(f.keys()) - { "start_time", ".reduced" }
            return datasets

        def var_data(ds_name):
            with h5py.File(hdf5_file.name) as f:
                data = f[ds_name]["data"][()]
            return data

        # Helper function that sets a bunch of default arguments
        run_and_save_helper = partial(run_and_save, proposal, run_number,
                                      Path(hdf5_file.name), context_path=Path(ctx_file.name))

        # First run with raw data and a filter so only one Variable is executed
        run_and_save_helper(run_data=RunData.RAW, match=["string"])
        assert var_datasets() == { "string" }

        # Now select a Variable with dependencies
        run_and_save_helper(run_data=RunData.RAW, match=["scalar2", "timestamp"])
        assert var_datasets() == { "string", "scalar1", "scalar2", "timestamp" }
        ts = var_data("timestamp")

        # Requesting a Variable that requires proc data with only raw data
        # should not execute anything.
        run_and_save_helper(run_data=RunData.RAW, match=["meta_array"])
        assert var_datasets() == { "string", "scalar1", "scalar2", "timestamp" }
        assert var_data("timestamp") == ts

        # But with proc data all dependencies should be executed
        run_and_save_helper(run_data=RunData.PROC, match=["meta_array"])
        assert var_datasets() == { "string", "scalar1", "scalar2", "timestamp", "array", "meta_array" }
        assert var_data("timestamp") > ts
