import json
import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.util import find_spec
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import xarray as xr
from kafka.structs import TopicPartition

from damnit.api import Damnit, submit
from damnit.context import Cell
from damnit.backend.combine import (
    CombinedFragment,
    FileSubmissionProcessor,
    RebalanceRequested,
    combine,
    combine_fragment_worker,
)
from damnit.backend.db import MsgKind
from damnit.ctxsupport.damnit_writing import DamnitFileWriter
from damnit.definitions import FILE_SUBMIT_TOPIC


def controlled_worker(src, dst, msg_timestamp):
    dst.parent.mkdir(parents=True, exist_ok=True)
    started = dst.parent / "worker-started"
    release = dst.parent / "release-worker"
    started.touch()
    while not release.exists():
        time.sleep(0.01)
    return CombinedFragment({}, "test")


@dataclass
class ManualFuture:
    result_value: object = None
    error: Exception = None
    finished: bool = False

    def done(self):
        return self.finished

    def result(self):
        if not self.finished:
            raise AssertionError("future result requested before completion")
        if self.error is not None:
            raise self.error
        return self.result_value

    def finish(self, result=None, error=None):
        self.result_value = result
        self.error = error
        self.finished = True


class ManualExecutor:
    def __init__(self):
        self.submissions = []

    def submit(self, function, *args):
        future = ManualFuture()
        self.submissions.append((function, args, future))
        return future

    def shutdown(self, **kwargs):
        pass


class SchedulerConsumer:
    def __init__(self):
        self.commits = []
        self.pause_calls = []
        self.resume_calls = []
        self._paused_partitions = set()
        self.tp = TopicPartition(FILE_SUBMIT_TOPIC, 0)
        self.listener = None

    def subscribe(self, topics, listener=None):
        self.listener = listener

    def assignment(self):
        return {self.tp}

    def commit(self, offsets=None):
        self.commits.append(offsets)

    def pause(self, *partitions):
        self.pause_calls.append(set(partitions))
        self._paused_partitions.update(partitions)

    def resume(self, *partitions):
        self.resume_calls.append(set(partitions))
        self._paused_partitions.difference_update(partitions)

    def paused(self):
        return set(self._paused_partitions)

    def close(self):
        pass


class SchedulerProducer:
    def send(self, *args, **kwargs):
        return SimpleNamespace(get=lambda **kwargs: None)

    def flush(self, **kwargs):
        pass

    def close(self, **kwargs):
        pass


def scheduler_message(tmp_path, run, offset, proposal=1234):
    return SimpleNamespace(
        topic=FILE_SUBMIT_TOPIC,
        partition=0,
        offset=offset,
        timestamp=0,
        value=json.dumps({
            "msg_kind": MsgKind.file_submission.value,
            "data": {
                "damnit_dir": str(tmp_path),
                "new_file": str(tmp_path / f"fragment-{run}-{offset}.h5"),
                "proposal": proposal,
                "run": run,
            },
        }).encode(),
    )


def make_scheduler(
    tmp_path, monkeypatch, *, workers=2, max_pending=16, finalize=False,
):
    consumer = SchedulerConsumer()
    executor = ManualExecutor()
    monkeypatch.setattr("damnit.backend.combine.KafkaConsumer", lambda *a, **k: consumer)
    monkeypatch.setattr("damnit.backend.combine.KafkaProducer", lambda *a, **k: SchedulerProducer())
    processor = FileSubmissionProcessor(
        workers=workers,
        max_pending=max_pending,
        executor=executor,
        poll_timeout_ms=1,
    )
    if not finalize:
        processor._finalize_job = lambda job, result: None
    return processor, consumer, executor


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


@pytest.mark.skipif(find_spec("netCDF4") is None, reason="netCDF4 not installed")
def test_combine_netcdf_dimensions(tmp_path):
    import netCDF4
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
            # Model an HDF5 image marker.
            if name == "second":
                f[f"{name}/data"].attrs["CLASS"] = "IMAGE"
                f[f"{name}/data"].attrs["IMAGE_VERSION"] = "1.2"
        combine(src, dst)

    with netCDF4.Dataset(dst) as f:
        assert f.groups["first"].variables["data"].dimensions == ("x", "y")
        assert f.groups["second"].variables["data"].dimensions == ("u", "v")
        assert f.groups["third"].variables["data"].dimensions == ("α", "β", "γ")

    with h5py.File(dst) as f:
        assert f["second/data"].attrs["CLASS"] == "IMAGE"
        assert f["second/data"].attrs["IMAGE_VERSION"] == "1.2"
        assert f["second/u"].attrs["CLASS"] == b"DIMENSION_SCALE"


def test_combine_retains_source_until_finalization(tmp_path):
    src = tmp_path / "fragment.h5"
    dst = tmp_path / "combined.h5"
    with h5py.File(src, "w") as f:
        f.create_dataset("value", data=np.arange(3))

    combine(src, dst)
    assert src.exists()
    assert dst.exists()

    # A retry after the hard-link fast path must not copy a file onto itself.
    combine(src, dst)
    with h5py.File(dst) as f:
        np.testing.assert_array_equal(f["value"][()], np.arange(3))


def test_combine_fragment_worker_runs_in_spawned_process(tmp_path):
    src = tmp_path / "fragment.h5"
    dst = tmp_path / "combined.h5"
    with h5py.File(src, "w") as f:
        f.attrs["provenance"] = "test"
        reduced = f.create_group(".reduced")
        reduced.create_dataset("value", data=3)
        f.create_group(".errors")

    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        result = pool.submit(
            combine_fragment_worker,
            src,
            dst,
            datetime.now(timezone.utc),
        ).result()

    assert result.provenance == "test"
    assert result.reduced_data["value"].value == 3
    assert src.exists()
    assert dst.exists()


def test_scheduler_overlaps_destinations_and_serializes_same_destination(tmp_path, monkeypatch):
    processor, consumer, executor = make_scheduler(tmp_path, monkeypatch)
    a1 = scheduler_message(tmp_path, run=1, offset=10)
    a2 = scheduler_message(tmp_path, run=1, offset=11)
    b1 = scheduler_message(tmp_path, run=2, offset=12)

    processor._accept_record(a1)
    processor._accept_record(a2)
    processor._accept_record(b1)
    processor._schedule_available()

    assert [args[0].name for _, args, _ in executor.submissions] == [
        "fragment-1-10.h5", "fragment-2-12.h5",
    ]
    assert len(processor.active) == 2
    assert len(processor.pending[next(
        destination for destination in processor.pending
        if destination.name == "p1234_r1.h5"
    )]) == 1

    # Completing A1 releases only A's destination; A2 starts afterwards while
    # B1 was already allowed to run independently.
    executor.submissions[0][2].finish(CombinedFragment({}, "test"))
    processor._check_completed()

    assert [args[0].name for _, args, _ in executor.submissions] == [
        "fragment-1-10.h5", "fragment-2-12.h5", "fragment-1-11.h5",
    ]
    assert len(processor.active) == 2
    assert consumer.commits[-1][consumer.tp].offset == 11


def test_scheduler_failure_keeps_destination_active(tmp_path, monkeypatch):
    processor, consumer, executor = make_scheduler(tmp_path, monkeypatch, workers=1)
    first = scheduler_message(tmp_path, run=1, offset=20)
    second = scheduler_message(tmp_path, run=1, offset=21)

    processor._accept_record(first)
    processor._accept_record(second)
    processor._schedule_available()
    executor.submissions[0][2].finish(error=RuntimeError("merge failed"))

    with pytest.raises(RuntimeError, match="merge failed"):
        processor._check_completed()

    destination = next(iter(processor.active))
    assert destination.name == "p1234_r1.h5"
    assert len(processor.pending[destination]) == 1
    assert len(executor.submissions) == 1
    assert not consumer.commits


def test_scheduler_finalization_failure_keeps_fragment_and_destination_active(
    tmp_path, monkeypatch,
):
    processor, consumer, executor = make_scheduler(
        tmp_path, monkeypatch, workers=1, finalize=True,
    )
    first = scheduler_message(tmp_path, run=1, offset=22)
    second = scheduler_message(tmp_path, run=1, offset=23)
    first_source = tmp_path / "fragment-1-22.h5"
    first_source.touch()

    class FakeDB:
        kafka_topic = "run-updates"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        "damnit.backend.combine.DamnitDB.from_dir",
        lambda path: FakeDB(),
    )
    monkeypatch.setattr("damnit.backend.combine.add_to_db", lambda *args, **kwargs: None)

    def fail_notification(*args, **kwargs):
        raise RuntimeError("notification failed")

    processor.send_update = fail_notification
    processor._accept_record(first)
    processor._accept_record(second)
    processor._schedule_available()
    executor.submissions[0][2].finish(CombinedFragment({}, "test"))

    with pytest.raises(RuntimeError, match="notification failed"):
        processor._check_completed()

    destination = next(iter(processor.active))
    assert first_source.exists()
    assert not consumer.commits
    assert len(processor.active) == 1
    assert len(processor.pending[destination]) == 1
    assert len(executor.submissions) == 1


def test_scheduler_commit_failure_retains_record_window_and_blocks_destination(
    tmp_path, monkeypatch,
):
    processor, consumer, executor = make_scheduler(
        tmp_path, monkeypatch, workers=1,
    )
    first = scheduler_message(tmp_path, run=1, offset=24)
    second = scheduler_message(tmp_path, run=1, offset=25)
    processor._accept_record(first)
    processor._accept_record(second)
    processor._schedule_available()
    records = processor._partition_records[consumer.tp]

    def fail_commit(offsets=None):
        raise RuntimeError("commit failed")

    consumer.commit = fail_commit
    executor.submissions[0][2].finish(CombinedFragment({}, "test"))

    with pytest.raises(RuntimeError, match="commit failed"):
        processor._check_completed()

    destination = next(iter(processor.active))
    assert processor._partition_records[consumer.tp] is records
    assert len(records) == 2
    assert processor._uncommitted_count == 2
    assert len(processor.pending[destination]) == 1
    assert len(executor.submissions) == 1


def test_scheduler_commits_only_contiguous_completed_offsets(tmp_path, monkeypatch):
    processor, consumer, executor = make_scheduler(tmp_path, monkeypatch)
    first = scheduler_message(tmp_path, run=1, offset=30)
    second = scheduler_message(tmp_path, run=2, offset=32)

    processor._accept_record(first)
    processor._accept_record(second)
    processor._schedule_available()

    # Finish the later record first.  Its partition offset cannot be committed
    # until the earlier record has also been finalized.
    executor.submissions[1][2].finish(CombinedFragment({}, "test"))
    processor._check_completed()
    assert not consumer.commits

    executor.submissions[0][2].finish(CombinedFragment({}, "test"))
    processor._check_completed()

    assert len(consumer.commits) == 1
    committed = consumer.commits[0][consumer.tp]
    assert committed.offset == 33


def test_scheduler_bounds_received_records_behind_slow_record(tmp_path, monkeypatch):
    processor, consumer, executor = make_scheduler(
        tmp_path, monkeypatch, workers=3, max_pending=3,
    )
    records = [
        scheduler_message(tmp_path, run=run, offset=offset)
        for run, offset in enumerate((10, 12, 14), start=1)
    ]

    for record in records:
        processor._accept_record(record)
    processor._schedule_available()

    # Later jobs can finish, but their received records stay retained behind
    # the unfinished first record and continue to count against intake.
    executor.submissions[1][2].finish(CombinedFragment({}, "test"))
    executor.submissions[2][2].finish(CombinedFragment({}, "test"))
    processor._check_completed()

    assert processor._uncommitted_count == 3
    assert not consumer.commits
    processor._update_consumer_flow()
    assert consumer.pause_calls == [{consumer.tp}]

    with pytest.raises(RuntimeError, match="pending record limit"):
        processor._accept_record(scheduler_message(tmp_path, run=4, offset=16))


def test_scheduler_reconciles_actual_pause_state(tmp_path, monkeypatch):
    processor, consumer, _ = make_scheduler(
        tmp_path, monkeypatch, workers=1, max_pending=1,
    )
    processor._accept_record(scheduler_message(tmp_path, run=1, offset=50))
    processor._schedule_available()
    processor._update_consumer_flow()
    assert consumer.pause_calls == [{consumer.tp}]

    # Model a rebalance clearing kafka-python's actual pause state while the
    # coordinator still has the same assignment and pending work.
    consumer._paused_partitions.clear()
    processor._update_consumer_flow()
    assert consumer.pause_calls == [{consumer.tp}, {consumer.tp}]


def test_scheduler_stops_after_non_initial_rebalance(tmp_path, monkeypatch):
    processor, consumer, _ = make_scheduler(tmp_path, monkeypatch)
    consumer.listener.on_partitions_assigned({consumer.tp})
    assert processor._rebalance_requested is None

    consumer.listener.on_partitions_revoked({consumer.tp})

    with pytest.raises(RebalanceRequested, match="partitions revoked"):
        processor._raise_if_rebalance_requested()


def test_scheduler_rebalance_during_poll_yields_no_record(tmp_path, monkeypatch):
    processor, consumer, executor = make_scheduler(tmp_path, monkeypatch)
    consumer.listener.on_partitions_assigned({consumer.tp})

    def poll(**kwargs):
        consumer.listener.on_partitions_revoked({consumer.tp})
        return {consumer.tp: [scheduler_message(tmp_path, run=1, offset=26)]}

    consumer.poll = poll

    with pytest.raises(RebalanceRequested, match="partitions revoked"):
        list(processor._poll_records())

    assert not processor._partition_records
    assert not executor.submissions


def test_shutdown_waits_for_spawned_worker(tmp_path, monkeypatch):
    consumer = SchedulerConsumer()
    monkeypatch.setattr("damnit.backend.combine.KafkaConsumer", lambda *a, **k: consumer)
    monkeypatch.setattr("damnit.backend.combine.KafkaProducer", lambda *a, **k: SchedulerProducer())
    monkeypatch.setattr(
        "damnit.backend.combine.combine_fragment_worker", controlled_worker,
    )
    processor = FileSubmissionProcessor(workers=1, max_pending=1)
    started = tmp_path / "extracted_data" / "worker-started"
    release = tmp_path / "extracted_data" / "release-worker"
    started.parent.mkdir(parents=True, exist_ok=True)
    job = processor._make_job({
        "damnit_dir": str(tmp_path),
        "new_file": str(tmp_path / "fragment.h5"),
        "proposal": 1234,
        "run": 1,
    }, datetime.now(timezone.utc))
    processor._enqueue_job(job)
    processor._schedule_available()

    marker = started
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    future = next(iter(processor.active.values()))[1]
    shutdown_started = threading.Event()
    shutdown_error = []
    shutdown_thread = None

    try:
        assert marker.exists()

        def shutdown():
            shutdown_started.set()
            try:
                processor.shutdown()
            except BaseException as exc:
                shutdown_error.append(exc)

        shutdown_thread = threading.Thread(target=shutdown)
        shutdown_thread.start()
        assert shutdown_started.wait(timeout=2)
        time.sleep(0.1)
        assert shutdown_thread.is_alive()
        assert not future.done()
    finally:
        release.touch()
        if shutdown_thread is None:
            processor.shutdown()
        else:
            shutdown_thread.join(timeout=5)

    if shutdown_thread is not None:
        assert not shutdown_thread.is_alive()
    assert not shutdown_error
    assert future.result(timeout=0) == CombinedFragment({}, "test")


def test_scheduler_pauses_at_pending_limit_and_resumes_after_completion(tmp_path, monkeypatch):
    processor, consumer, executor = make_scheduler(
        tmp_path, monkeypatch, workers=1, max_pending=1,
    )
    processor._accept_record(scheduler_message(tmp_path, run=1, offset=40))
    processor._schedule_available()
    processor._update_consumer_flow()

    assert consumer.pause_calls == [{consumer.tp}]

    executor.submissions[0][2].finish(CombinedFragment({}, "test"))
    processor._check_completed()
    processor._update_consumer_flow()

    assert consumer.resume_calls == [{consumer.tp}]
