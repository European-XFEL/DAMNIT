import json

import h5py
import netCDF4
import numpy as np
import xarray as xr

from damnit.api import Damnit, submit
from damnit.context import Cell
from damnit.backend.combine import combine, FileSubmissionProcessor
from damnit.backend.db import MsgKind
from damnit.ctxsupport.damnit_writing import DamnitFileWriter
from damnit.definitions import FILE_SUBMIT_TOPIC


class SubmitHelper:
    def __init__(self, broker, db, db_dir):
        self.broker = broker
        self.db = db
        self.db_dir = db_dir
        self.combiner = FileSubmissionProcessor(
            consumer_config={'auto_offset_reset': 'earliest'}
        )

    def submit_and_combine(self, proposal, run, data, provenance='test', errors=None):
        with self.broker.assert_produces(FILE_SUBMIT_TOPIC):
            submit(proposal, run, data,
                   provenance=provenance,
                   damnit_dir=self.db_dir,
                   errors=errors,
            )

        with self.broker.assert_produces(self.db.kafka_topic) as new_records:
            self.combiner.handle_one_message()
            self.combiner.producer.flush(timeout=5)

        return new_records

    def shutdown(self):
        self.combiner.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def test_combiner_service(mock_db, mock_kafka_broker):
    db_dir, db = mock_db

    with SubmitHelper(mock_kafka_broker, db, db_dir) as sh:
        new_records = sh.submit_and_combine(1234, 56, {
            # We need start_time in the DB for the API to see the run
            "array": np.arange(10), "start_time": 1776335722
        })

        msgs = [json.loads(r.value) for r in new_records]
        assert [m['msg_kind'] for m in msgs] == [MsgKind.run_values_updated.value]
        assert set(msgs[0]['data']['values']) == {"array", "start_time"}

        api_obj = Damnit(db_dir)
        np.testing.assert_array_equal(api_obj[56, "array"].read(), np.arange(10))

        sh.submit_and_combine(1234, 56, {"array": np.arange(15)})

        np.testing.assert_array_equal(api_obj[56, "array"].read(), np.arange(15))


def test_combiner_clears_previous_data(mock_db, mock_kafka_broker):
    db_dir, db = mock_db
    h5_path = db_dir / "extracted_data" / "p1234_r56.h5"

    with SubmitHelper(mock_kafka_broker, db, db_dir) as sh:

        sh.submit_and_combine(1234, 56, {
            "array": np.arange(10),
            "start_time": 1776335722,
        })

        with h5py.File(h5_path) as f:
            assert "array" in f
            assert ".reduced/array" in f
            assert ".errors/array" not in f

        sh.submit_and_combine(
            1234, 56, {}, errors={"array": RuntimeError("boom")}
        )

        with h5py.File(h5_path) as f:
            assert "array" not in f
            assert ".reduced/array" not in f
            assert ".errors/array" in f

        attrs = db.conn.execute(
            "SELECT attributes FROM run_variables WHERE proposal=? AND run=? AND name=?",
            (1234, 56, "array"),
        ).fetchone()[0]
        assert json.loads(attrs) == {
            "error": "boom",
            "error_cls": "RuntimeError",
        }

        sh.submit_and_combine(
            1234, 56, {"array": np.arange(15)},
        )

        with h5py.File(h5_path) as f:
            assert "array" in f
            assert ".reduced/array" in f
            assert ".errors/array" not in f
            np.testing.assert_array_equal(f["array/data"][()], np.arange(15))

        attrs = db.conn.execute(
            "SELECT attributes FROM run_variables WHERE proposal=? AND run=? AND name=?",
            (1234, 56, "array"),
        ).fetchone()[0]
        assert attrs is None


def test_combine_special_group_only(mock_db, mock_kafka_broker):
    db_dir, db = mock_db
    h5_path = db_dir / "extracted_data" / "p1234_r56.h5"

    with SubmitHelper(mock_kafka_broker, db, db_dir) as sh:

        sh.submit_and_combine(1234, 56,
            {"summary_only": Cell(data=None, summary_value=7), "preview_only": Cell(data=None, preview=np.arange(4))},
            errors={"error_only": Exception("boom")}
        )

        with h5py.File(h5_path) as f:
            assert ".reduced/summary_only" in f
            assert ".preview/preview_only" in f
            assert ".errors/error_only" in f
            assert "summary_only" not in f
            assert "preview_only" not in f
            assert "error_only" not in f
            assert f[".reduced/summary_only"][()] == 7
            np.testing.assert_array_equal(f[".preview/preview_only"][()], np.arange(4))
            assert f[".errors/error_only"].asstr()[()] == "boom"


def test_combine_netcdf_dimensions(tmp_path):
    dst = tmp_path / "combined.h5"

    for name, dims, shape in [
        ("first", ("x", "y"), (2, 3)),
        ("second", ("u", "v"), (4, 5)),
        ("third", ("α", "β", "γ"), (6, 7, 8))
    ]:
        src = tmp_path / f"{name}.h5"
        with h5py.File(src, "w") as f:
            DamnitFileWriter(f).store_data(
                name, xr.DataArray(np.zeros(shape), dims=dims, name="data")
            )
        combine(src, dst)

    with netCDF4.Dataset(dst) as f:
        assert f.groups["first"].variables["data"].dimensions == ("x", "y")
        assert f.groups["second"].variables["data"].dimensions == ("u", "v")
        assert f.groups["third"].variables["data"].dimensions == ("α", "β", "γ")
