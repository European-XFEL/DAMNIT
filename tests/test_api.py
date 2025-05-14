import subprocess
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import numpy as np
import plotly.express as px
import pytest
import xarray as xr
from matplotlib.image import AxesImage
from plotly.graph_objects import Figure as PlotlyFigure

from damnit import Damnit, RunVariables
from damnit.context import ContextFile
from damnit.backend.user_variables import UserEditableVariable
from .helpers import extract_mock_run


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
    extract_mock_run(100, match=['scalar1'])
    assert damnit.runs() == [1, 100]
    # We should only see variables for which data is actually available
    assert damnit[100].keys() == ["scalar1", "start_time"]

    # Test indexing by name and title
    assert rv["scalar1"].name == "scalar1"
    assert rv["Scalar1"].name == "scalar1"

    # Test getting the start_time. This one's a bit special because we create it
    # automatically, it's a proper variable except for the fact that we don't
    # insert it into the `variables` table.
    assert isinstance(rv["start_time"].read(), float)
    assert "Timestamp" in rv.titles()

    # Test getting comments. This is also special because it doesn't appear in
    # the `variables` table.
    assert "comment" not in rv.keys()
    db.change_run_comment(damnit.proposal, 1, "foo")
    assert "Comment" in rv.titles()
    assert rv["comment"].read() == "foo"

    with pytest.raises(KeyError):
        rv["foo"]

def test_variable_data(mock_db_with_data, monkeypatch):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)
    damnit = Damnit(db_dir)
    rv = damnit[1]

    # We disable updates from VariableData.write() by default so we can use the
    # convenient RunVariables.__setitem__() method.
    monkeypatch.setenv("DAMNIT_API_SEND_UPDATE", "0")

    # Insert a DataSet variable
    dataset_code = """
    from damnit_ctx import Cell, Variable
    import xarray as xr

    @Variable(title="Dataset")
    def dataset(run):
        data = xr.Dataset(data_vars={
            "foo": xr.DataArray([1, 2, 3]),
            "bar/baz": xr.DataArray([1+2j, 3-4j, 5+6j]),
        })
        return Cell(data, summary_value=data['bar/baz'][2])

    @Variable(title="Summary only")
    def summary_only(run):
        return Cell(None, summary_value=7)
    """
    (db_dir / "context.py").write_text(dedent(dataset_code))
    extract_mock_run(1)

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
    assert isinstance(dataset.bar_baz, xr.DataArray)
    assert dataset.bar_baz.dtype == np.complex128

    # Datasets have a internal _damnit attribute that should be removed
    assert len(dataset.attrs) == 0

    summary = rv["dataset"].summary()
    assert isinstance(summary, complex)
    assert summary == complex(5, 6)

    fig = rv['plotly_mc_plotface'].read()
    assert isinstance(fig, PlotlyFigure)
    assert fig == px.bar(x=["a", "b", "c"], y=[1, 3, 2])

    json_str = rv["plotly_mc_plotface"].read(deserialize_plotly=False)
    assert isinstance(json_str, str)

    arr = rv["array_preview"].preview_data()
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2

    arr = rv["image"].preview_data()  # Implicit preview for 2D array
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2

    assert rv["image"].preview_data(data_fallback=False) is None

    assert isinstance(rv["image"].preview(), AxesImage)

    fig = rv["plotly_preview"].preview_data()
    assert isinstance(fig, PlotlyFigure)

    assert rv["summary_only"].summary() == 7

    # It shouldn't be possible to write to non-editable variables or the default
    # variables from DAMNIT.
    with pytest.raises(RuntimeError):
        rv["dataset"] = 1
    with pytest.raises(RuntimeError):
        rv["start_time"] = 1

    # It also shouldn't be possible to write to variables that don't exist. We
    # have to test this because RunVariables.__setitem__() will allow creating
    # VariableData objects for variables that are missing from the run but do
    # exist in the database.
    with pytest.raises(KeyError):
        rv["blah"] = 1

    # Test setting an editable value
    db.add_user_variable(UserEditableVariable("foo", "Foo", "number"))
    rv["foo"] = 3.14
    assert rv["foo"].read() == 3.14

    # Test setting a comment
    rv["comment"] = "humbug"
    assert rv["comment"].read() == "humbug"

    # Test deleting values
    del rv["comment"]
    assert "comment" not in rv.keys()

    # Test sending Kafka updates
    foo_var = rv["foo"]
    with patch("damnit.kafka.KafkaProducer") as kafka_prd:
        foo_var.write(42, send_update=True)
        kafka_prd.assert_called_once()

# These are smoke tests to ensure that writing of all variable types succeed,
# tests of special cases are above in test_variable_data.
#
# Note that we allow any value to be converted to strings for convenience, so
# its `bad_input` value is None.
@pytest.mark.parametrize("variable_name,good_input,bad_input",
                         [("boolean", True, "foo"),
                          ("integer", 42, "foo"),
                          ("number", 3.14, "foo"),
                          ("stringy", "foo", None)])
def test_writing(variable_name, good_input, bad_input, mock_db_with_data, monkeypatch):
    db_dir, db = mock_db_with_data
    monkeypatch.chdir(db_dir)
    monkeypatch.setenv("DAMNIT_API_SEND_UPDATE", "0")
    damnit = Damnit(db_dir)
    rv = damnit[1]

    # There's already a `string` variable in the test context file so we can't
    # reuse the name as the type for our editable variable.
    variable_type = "string" if variable_name == "stringy" else variable_name

    # Add the user-editable variable
    user_var = UserEditableVariable(variable_name, variable_name.capitalize(), variable_type)
    db.add_user_variable(user_var)

    rv[variable_name] = good_input
    assert rv[variable_name].read() == good_input

    if bad_input is not None:
        with pytest.raises(TypeError):
            rv[variable_name] = bad_input

def test_api_dependencies(venv):
    package_path = Path(__file__).parent.parent
    venv.install(package_path)

    # Test that we can import the module successfully and don't accidentally
    # depend on other things.
    subprocess.run([str(venv.python), "-c", "import damnit"], check=True)
