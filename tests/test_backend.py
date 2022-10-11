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
from amore_mid_prototype.backend.extract_data import Results, RunData, get_proposal_path


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

    # Check the result values
    assert results.data["scalar1"] == 42
    assert results.data["scalar2"] == 3.14
    np.testing.assert_equal(results.data["array"], [42, 3.14])
    np.testing.assert_equal(results.data["meta_array"], [run_number, proposal])
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
