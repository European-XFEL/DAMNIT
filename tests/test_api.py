from textwrap import dedent

import pytest
import numpy as np
import xarray as xr

from damnit.context import ContextFile
from damnit import Damnit, RunVariables

from .helpers import amore_proto


def test_damnit(mock_db_with_data):
    db_dir, db = mock_db_with_data

    damnit = Damnit(db_dir)

    # Smoke test
    assert str(damnit.proposal) in repr(damnit)

    assert damnit.proposal == db.metameta["proposal"]
    assert damnit.runs() == [1]

    # Test indexing
    with pytest.raises(KeyError):
        damnit[0]
    with pytest.raises(TypeError):
        damnit["foo"]

    assert isinstance(damnit[1], RunVariables)
    assert damnit[1, "scalar1"].name == "scalar1"

    # Test table()
    df = damnit.table()
    assert len(df) == 1
    assert "scalar1" in df.columns

    df = damnit.table(with_titles=True)
    assert "Scalar1" in df.columns

def test_run_variables(mock_db_with_data, monkeypatch):
    db_dir, db = mock_db_with_data
    damnit = Damnit(db_dir)
    monkeypatch.chdir(db_dir)

    rv = damnit[1]

    # Test properties
    assert rv.proposal == db.metameta["proposal"]
    assert rv.run == 1

    # Test getting keys and titles
    assert "scalar1" in rv.keys()
    assert "Scalar1" in rv.titles()
    ctx = ContextFile.from_py_file(db_dir / "context.py")
    assert set(rv.keys()) == set(ctx.vars.keys()) | set(["start_time"])

    # Reprocess a single variable for another run
    amore_proto(["reprocess", "100", "--match", "scalar1", "--mock"])
    assert damnit.runs() == [1, 100]
    # We should only see variables for which data is actually available
    assert damnit[100].keys() == ["scalar1", "start_time"]

    # Test indexing by name and title
    assert rv["scalar1"].name == "scalar1"
    assert rv["Scalar1"].name == "scalar1"

    with pytest.raises(KeyError):
        rv["foo"]

def test_variable_data(mock_db_with_data, monkeypatch):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)
    damnit = Damnit(db_dir)
    rv = damnit[1]

    # Insert a DataSet variable
    dataset_code = """
    from damnit_ctx import Variable
    import xarray as xr

    @Variable(title="Dataset")
    def dataset(run):
        return xr.Dataset(data_vars={ "foo": xr.DataArray([1, 2, 3]) })
    """
    (db_dir / "context.py").write_text(dedent(dataset_code))
    amore_proto(["reprocess", "1", "--mock"])

    # Test properties
    assert rv["scalar1"].name == "scalar1"
    assert rv["scalar1"].title == "Scalar1"

    # Test reading different data types
    assert rv["scalar1"].read() == 42
    assert rv["scalar1"].summary() == 42
    assert rv["empty_string"].read() == ""

    array = rv["array"].read()
    assert isinstance(array, np.ndarray)
    assert np.allclose(array, [rv["scalar1"].read(), rv["scalar2"].read()])

    meta_array = rv["meta_array"].read()
    assert isinstance(meta_array, xr.DataArray)
    assert np.allclose(meta_array, [1, damnit.proposal])

    dataset = rv["dataset"].read()
    assert isinstance(dataset, xr.Dataset)
    assert isinstance(dataset.foo, xr.DataArray)

    # Datasets have a internal _damnit attribute that should be removed
    assert len(dataset.attrs) == 0
