import json
from importlib.util import find_spec

import h5py
import numpy as np
import pytest
import xarray as xr

from damnit.api import Damnit, has_data, submit
from damnit.context import Cell
from damnit.backend.combine import combine, FileSubmissionProcessor
from damnit.backend.db import MsgKind, db_path
from damnit.backend.listener import ListenerDB
from damnit.ctxsupport.damnit_writing import DamnitFileWriter, PROVENANCE_ATTR
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
        new_records = sh.submit_and_combine(np.int64(1234), np.int32(56), {
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


def test_combiner_skips_unmonitored(mock_db, mock_kafka_broker, tmp_path):
    db_dir, db = mock_db
    listener_dir = tmp_path / "listener"
    listener_dir.mkdir()
    listener_db = ListenerDB(listener_dir)
    listener_db.add_proposal_db(1234, db_dir, official=True)
    listener_db.set_monitor(db_dir, False)

    combiner = FileSubmissionProcessor(
        consumer_config={'auto_offset_reset': 'earliest'}, listener_dir=listener_dir
    )
    with mock_kafka_broker.assert_produces(FILE_SUBMIT_TOPIC):
        submit(1234, 56, {"array": np.arange(10)}, provenance="test", damnit_dir=db_dir)
    combiner.handle_one_message()
    combiner.shutdown()

    # The fragment should be left alone
    h5_dir = db_dir / "extracted_data"
    assert not (h5_dir / "p1234_r56.h5").exists()
    assert len(list(h5_dir.glob("*.ready.h5"))) == 1

    # Unregistered databases are only monitored in dynamic mode
    listener_db.settings["static_mode"] = True
    assert not listener_db.is_monitored(tmp_path / "other")
    listener_db.settings["static_mode"] = False
    assert listener_db.is_monitored(tmp_path / "other")


@pytest.mark.parametrize("proposal, run", [(1234.0, 56), (1234, 56.0)])
def test_submit_rejects_non_integer_run_ids(tmp_path, proposal, run):
    with pytest.raises(TypeError):
        submit(proposal, run, {}, provenance="test", damnit_dir=tmp_path)

    assert not (tmp_path / "extracted_data").exists()


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
            assert not has_data(f["array"])
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
            assert not has_data(f["summary_only"])
            assert not has_data(f["preview_only"])
            assert not has_data(f["error_only"])
            assert f[".reduced/summary_only"][()] == 7
            np.testing.assert_array_equal(f[".preview/preview_only"][()], np.arange(4))
            assert f[".errors/error_only"].asstr()[()] == "boom"


@pytest.mark.skipif(find_spec("netCDF4") is None, reason="netCDF4 not installed")
def test_combine_netcdf_dimensions(mock_db):
    import netCDF4
    db_dir, _ = mock_db
    dst = db_dir / "combined.h5"

    for name, dims, shape in [
        ("first", ("x", "y"), (2, 3)),
        ("second", ("u", "v"), (4, 5)),
        ("third", ("α", "β", "γ"), (6, 7, 8))
    ]:
        src = db_dir / f"{name}.h5"
        with h5py.File(src, "w") as f:
            DamnitFileWriter(f).store_data(
                name, xr.DataArray(np.zeros(shape), dims=dims, name="data")
            )
            # Model an HDF5 image marker.
            if name == "second":
                f[f"{name}/data"].attrs["CLASS"] = "IMAGE"
                f[f"{name}/data"].attrs["IMAGE_VERSION"] = "1.2"
            f[name].attrs[PROVENANCE_ATTR] = "test"
        combine(src, dst, db_path(db_dir), 1234, 56)

    with netCDF4.Dataset(dst) as f:
        assert f.groups["first"].variables["data"].dimensions == ("x", "y")
        assert f.groups["second"].variables["data"].dimensions == ("u", "v")
        assert f.groups["third"].variables["data"].dimensions == ("α", "β", "γ")

    with h5py.File(dst) as f:
        assert f["second/data"].attrs["CLASS"] == "IMAGE"
        assert f["second/data"].attrs["IMAGE_VERSION"] == "1.2"
        assert f["second/u"].attrs["CLASS"] == b"DIMENSION_SCALE"


def version_list(var):
    return [(v.version, v.provenance) for v in var.versions()]


def test_versions_by_provenance(mock_db, mock_kafka_broker):
    db_dir, db = mock_db

    with SubmitHelper(mock_kafka_broker, db, db_dir) as sh:
        sh.submit_and_combine(1234, 56, {
            "x": Cell(np.arange(3), summary_value=1), "start_time": 1776335722
        }, provenance="online")
        sh.submit_and_combine(1234, 56, {
            "x": Cell(np.arange(5), summary_value=2)
        }, provenance="context.py")

        x = Damnit(db_dir)[56, "x"]
        assert x.provenance == "context.py"
        assert version_list(x) == [(1, "online"), (2, "context.py")]
        np.testing.assert_array_equal(x.read(), np.arange(5))
        np.testing.assert_array_equal(x.read(provenance="online"), np.arange(3))
        assert x.summary() == 2
        assert x.summary(provenance="online") == 1
        with pytest.raises(KeyError):
            x.read(provenance="nonexistent")

        # A new online version only replaces the old online version
        sh.submit_and_combine(1234, 56, {
            "x": Cell(np.arange(7), summary_value=3)
        }, provenance="online")

        assert version_list(x) == [(2, "context.py"), (3, "online")]
        np.testing.assert_array_equal(x.read(), np.arange(7))
        np.testing.assert_array_equal(x.read(provenance="context.py"), np.arange(5))

        # The database only has the latest version
        rows = db.conn.execute(
            "SELECT version, provenance FROM run_variables WHERE name='x'"
        ).fetchall()
        assert [tuple(r) for r in rows] == [(3, "online")]


def test_same_provenance_replaces(mock_db, mock_kafka_broker):
    db_dir, db = mock_db
    h5_path = db_dir / "extracted_data" / "p1234_r56.h5"

    with SubmitHelper(mock_kafka_broker, db, db_dir) as sh:
        sh.submit_and_combine(1234, 56, {"x": 1, "start_time": 1776335722})
        sh.submit_and_combine(1234, 56, {"x": 2})

    with h5py.File(h5_path) as f:
        assert ".versions" not in f
        assert f["x"].attrs["_damnit_version"] == 2

    assert version_list(Damnit(db_dir)[56, "x"]) == [(2, "test")]


def test_failed_version_hides_preview(mock_db, mock_kafka_broker):
    db_dir, db = mock_db
    h5_path = db_dir / "extracted_data" / "p1234_r56.h5"

    with SubmitHelper(mock_kafka_broker, db, db_dir) as sh:
        sh.submit_and_combine(1234, 56, {
            "x": Cell(np.arange(3), preview=np.arange(3)), "start_time": 1776335722
        }, provenance="online")
        sh.submit_and_combine(
            1234, 56, {}, errors={"x": RuntimeError("boom")}, provenance="context.py"
        )

    with h5py.File(h5_path) as f:
        assert ".errors/x" in f
        assert ".preview/x" not in f
        assert not has_data(f["x"])
        assert ".versions/x/1/.preview/x" in f

    x = Damnit(db_dir)[56, "x"]
    assert x.preview_data() is None
    np.testing.assert_array_equal(x.preview_data(provenance="online"), np.arange(3))
    np.testing.assert_array_equal(x.read(provenance="online"), np.arange(3))
    with pytest.raises(KeyError, match="has no data"):
        x.read()


def test_versions_xarray(mock_db, mock_kafka_broker):
    db_dir, db = mock_db
    da = xr.DataArray(np.arange(3.), dims=["t"], coords={"t": [1, 2, 3]})
    ds = xr.Dataset({"a": da})

    with SubmitHelper(mock_kafka_broker, db, db_dir) as sh:
        sh.submit_and_combine(1234, 56, {
            "da": da, "ds": ds, "start_time": 1776335722
        }, provenance="online")
        sh.submit_and_combine(1234, 56, {"da": 1, "ds": 2}, provenance="context.py")

    rv = Damnit(db_dir)[56]
    xr.testing.assert_identical(rv["da"].read(provenance="online"), da)
    xr.testing.assert_identical(rv["ds"].read(provenance="online"), ds)


def test_fill_in_legacy_versions(mock_db, mock_kafka_broker):
    db_dir, db = mock_db
    h5_path = db_dir / "extracted_data" / "p1234_r56.h5"

    with SubmitHelper(mock_kafka_broker, db, db_dir) as sh:
        sh.submit_and_combine(1234, 56, {"x": 1, "start_time": 1776335722},
                              provenance="online")

        # The first fragment is hard-linked to create the run file
        with h5py.File(h5_path, "r+") as f:
            assert f["x"].attrs["_damnit_version"] == 1

            # Make it look like a file written before versioning
            for attr in ["_damnit_version", "_damnit_provenance", "_damnit_timestamp"]:
                del f["x"].attrs[attr]

        sh.submit_and_combine(1234, 56, {"x": 2}, provenance="context.py")

    with h5py.File(h5_path) as f:
        assert f[".versions/x/1/x"].attrs["_damnit_version"] == 1
        assert f[".versions/x/1/x"].attrs["_damnit_provenance"] == "online"
        assert f["x"].attrs["_damnit_version"] == 2


def test_delete_variable_versions(mock_db, mock_kafka_broker):
    from damnit.gui.util import delete_variable

    db_dir, db = mock_db
    h5_path = db_dir / "extracted_data" / "p1234_r56.h5"

    with SubmitHelper(mock_kafka_broker, db, db_dir) as sh:
        sh.submit_and_combine(1234, 56, {"x": 1, "y": 1, "start_time": 1776335722},
                              provenance="online")
        sh.submit_and_combine(1234, 56, {"x": 2, "y": 2}, provenance="context.py")

    delete_variable(db, "x")

    with h5py.File(h5_path) as f:
        assert "x" not in f
        assert ".reduced/x" not in f
        assert ".versions/x" not in f

        assert has_data(f["y"])
        assert ".reduced/y" in f
        assert ".versions/y/1/y" in f
