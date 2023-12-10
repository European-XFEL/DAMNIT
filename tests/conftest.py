import os
import socket
import textwrap
from unittest.mock import MagicMock

import pytest
import numpy as np
import pandas as pd

from damnit.context import ContextFile
from damnit.backend.extract_data import extract_in_subprocess, add_to_db
from damnit.ctxsupport.damnit_ctx import types_map, UserEditableVariable
from damnit.backend.db import DamnitDB, DB_NAME


def make_mock_db(ctx, mock_db):
    db_dir, db = mock_db
    db.metameta["proposal"] = 1234

    (db_dir / "context.py").write_text(ctx.code)
    extracted_data_dir = db_dir / "extracted_data"
    extracted_data_dir.mkdir(exist_ok=True)

    cwd = os.getcwd()
    try:
        os.chdir(db_dir)
        reduced_data = extract_in_subprocess(1234, 42,
                                             extracted_data_dir / "p1234_r42.h5",
                                             mock=True, sandbox=False)
        add_to_db(reduced_data, db.conn, 1234, 42)
    finally:
        os.chdir(cwd)

def mkcontext(code, **kwargs):
    return ContextFile.from_str(textwrap.dedent(code), **kwargs)

@pytest.fixture
def mock_ctx():
    code = """
    import time
    import numpy as np
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
    """

    return mkcontext(code)

@pytest.fixture
def mock_user_vars():

    user_variables = {}

    for kk in types_map.keys():
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
    #def dep_integer(run, user_integer: "var#user_integer"):
    def dep_integer(run, user_integer=12):
        return user_integer + 1

    @Variable(title="Depend from user number")
    #def dep_number(run, user_number: "var#user_number"):
    def dep_number(run, user_number=10.2):
        return user_number * 1.0

    @Variable(title="Depend from user boolean")
    #def dep_boolean(run, user_boolean: "var#user_boolean"):
    def dep_boolean(run, user_boolean=True):
        return user_boolean and False

    @Variable(title="Depend from user string")
    #def dep_string(run, user_string: "var#user_string"):
    def dep_string(run, user_string="foo"):
        return user_string * 2

    """

    return mkcontext(code, external_vars=mock_user_vars)

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
def mock_db(tmp_path, mock_ctx):
    db = DamnitDB.from_dir(tmp_path)

    (tmp_path / "context.py").write_text(mock_ctx.code)

    yield tmp_path, db

    db.close()

@pytest.fixture
def mock_db_with_data(mock_ctx, mock_db):
    make_mock_db(mock_ctx, mock_db)
    yield mock_db

@pytest.fixture
def bound_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", 0))
    port = s.getsockname()[1]

    yield port

    s.close()