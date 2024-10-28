import socket
from unittest.mock import MagicMock

import pytest
import numpy as np

from damnit.backend.db import DamnitDB
from damnit.backend.user_variables import value_types_by_name, UserEditableVariable

from .helpers import amore_proto, mkcontext, extract_mock_run


@pytest.fixture
def mock_ctx():
    code = """
    import time
    import numpy as np
    import plotly.express as px
    import xarray as xr
    from damnit.context import Variable

    @Variable(title="Scalar1")
    def scalar1(run):
        return 42

    @Variable(title="Scalar2")
    def scalar2(run, foo: "var#scalar1"):
        return 3.14

    @Variable(title="Empty text")
    def empty_string(run):
        return ""

    # Note: we set the summary to np.size() to test that the backend can handle
    # summary values that return plain Python scalars (int, float) instead of
    # numpy scalars (np.int32, np.float32, etc).
    @Variable(title="Array", summary="size")
    def array(run, foo: "var#scalar1", bar: "var#scalar2"):
        return np.array([foo, bar])

    # Can't have a title of 'Timestamp' or it'll conflict with the GUI's
    # 'Timestamp' colummn.
    @Variable(title="Timestamp2")
    def timestamp(run):
        return time.time()

    @Variable(data="proc", summary="mean")
    def meta_array(run, baz: "var#array", run_number: "meta#run_number",
                   proposal: "meta#proposal", ts: "var#timestamp"):
        return xr.DataArray([run_number, proposal])

    @Variable(data="raw")
    def string(run, proposal_path: "meta#proposal_path"):
        return str(proposal_path)

    @Variable(data="raw")
    def plotly_mc_plotface(run):
        return px.bar(x=["a", "b", "c"], y=[1, 3, 2])
    """

    return mkcontext(code)

@pytest.fixture
def mock_user_vars():

    user_variables = {}

    for kk in value_types_by_name.keys():
        var_name = f"user_{kk}"
        user_variables[var_name] = UserEditableVariable(
            var_name,
            f"User {kk}",
            kk,
            description=f"This is a user editable variable of type {kk}"
        )

    return user_variables

@pytest.fixture
def mock_ctx_user(mock_user_vars):
    code = """
    import time
    import numpy as np
    import xarray as xr
    from damnit.context import Variable

    @Variable(title="Depend from user integer")
    #def dep_integer(run, user_integer: "input#user_integer"):
    def dep_integer(run, user_integer=12):
        return user_integer + 1

    @Variable(title="Depend from user number")
    #def dep_number(run, user_number: "input#user_number"):
    def dep_number(run, user_number=10.2):
        return user_number * 1.0

    @Variable(title="Depend from user boolean")
    #def dep_boolean(run, user_boolean: "input#user_boolean"):
    def dep_boolean(run, user_boolean=True):
        return user_boolean and False

    @Variable(title="Depend from user string")
    #def dep_string(run, user_string: "input#user_string"):
    def dep_string(run, user_string="foo"):
        return user_string * 2

    """

    return mkcontext(code)

@pytest.fixture
def mock_run():
    run = MagicMock()

    run.train_ids = np.arange(10)

    def select_trains(train_slice):
        return run

    run.select_trains.side_effect = select_trains

    def train_timestamps():
        return np.array(run.train_ids + 1493892000000000000,
                        dtype="datetime64[ns]")

    run.train_timestamps.side_effect = train_timestamps

    run.files = [MagicMock(filename="/tmp/foo/bar.h5")]

    return run

@pytest.fixture
def mock_db(tmp_path, mock_ctx, monkeypatch):
    db = DamnitDB.from_dir(tmp_path)

    (tmp_path / "context.py").write_text(mock_ctx.code)

    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        amore_proto(["read-context"])

    yield tmp_path, db

    db.close()

@pytest.fixture
def mock_db_with_data(mock_ctx, mock_db, monkeypatch):
    db_dir, db = mock_db

    with monkeypatch.context() as m:
        m.chdir(db_dir)
        amore_proto(["proposal", "1234"])
        extract_mock_run(1)

    yield mock_db

@pytest.fixture
def bound_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", 0))
    port = s.getsockname()[1]

    yield port

    s.close()
