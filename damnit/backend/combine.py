"""Read temporary HDF5 files, combine them into a single file per run, updating DB"""
import json
import logging
import multiprocessing
import posixpath
import re
import signal
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
from kafka import KafkaConsumer, KafkaProducer
from kafka.structs import OffsetAndMetadata, TopicPartition

from ..context import DataType
from ..definitions import FILE_SUBMIT_TOPIC, update_brokers
from .db import DamnitDB, MsgKind, msg_dict
from .extract_data import add_to_db, load_reduced_data
from .service import notify_ready

FRAGMENT_PATTERN = re.compile(r"p(\d+)_r(\d+).(.+).ready.h5$")
SPECIAL_GROUPS = (".reduced", ".preview", ".errors")

# If the file doesn't exist N seconds after the Kafka message was sent, move on
NO_FILE_TIMEOUT = 30

DEFAULT_WORKERS = 4
DEFAULT_MAX_PENDING = 16
DEFAULT_POLL_TIMEOUT_MS = 1000
NOTIFICATION_TIMEOUT = 30

_DIMENSION_SCALE_ATTRS = {
    # CLASS is a special attribute used by HDF5 dimension scales, but it is also
    # used by other HDF5 conventions. We only skip it when the destination
    # object already has it.
    # "CLASS",
    "NAME",
    "REFERENCE_LIST",
    "DIMENSION_LIST",
    "DIMENSION_LABELS",
}

log = logging.getLogger(__name__)


def _dimension_scale_name(dataset: h5py.Dataset) -> str:
    """Return the HDF5 dimension-scale name as a Python string."""
    name = dataset.attrs.get("NAME", "")

    if isinstance(name, bytes):
        return name.decode("utf-8", "surrogateescape")

    return str(name)


# HDF5's copy function (H5Ocopy) doesn't work correctly with dimension scales,
# which are used on NetCDF4 data (saved xarray objects). We work around that by
# rebuilding manually its dimension-scale metadata.
# https://github.com/HDFGroup/hdf5/issues/6200
def copy_h5_obj_rebuild_dimscales(fsrc: h5py.File, fdst: h5py.File, path: str) -> None:
    """
    Copy an HDF5 object and rebuild its dimension-scale metadata.

    The source and destination paths are identical. The destination object
    must not already exist, and all attached dimension scales must be inside
    the copied subtree.
    """
    source = fsrc[path]
    path = source.name

    objects: list[h5py.Group | h5py.Dataset] = [source]

    if isinstance(source, h5py.Group):
        source.visititems(lambda _, obj: objects.append(obj))

    datasets = [
        obj for obj in objects
        if isinstance(obj, h5py.Dataset)
    ]

    scale_names = {
        dataset.name: _dimension_scale_name(dataset)
        for dataset in datasets
        if dataset.is_scale
    }

    dimensions: list[tuple[str, int, str, list[str]]] = []

    for dataset in datasets:
        for index in range(dataset.ndim):
            dimension = dataset.dims[index]
            scale_paths = [scale.name for scale in dimension.values()]

            if dimension.label or scale_paths:
                dimensions.append(
                    (dataset.name, index, dimension.label, scale_paths)
                )

    # NetCDF-C treats dimension IDs as file-wide, while each fragment starts
    # allocating them independently.
    destination_dimids: list[int] = []

    def collect_dimension_ids(_, obj):
        if (
            isinstance(obj, h5py.Dataset)
            and obj.is_scale
            and "_Netcdf4Dimid" in obj.attrs
        ):
            destination_dimids.append(int(obj.attrs["_Netcdf4Dimid"]))

    fdst.visititems(collect_dimension_ids)
    dimid_offset = max(destination_dimids, default=-1) + 1

    parent = posixpath.dirname(path)

    if parent != "/":
        fdst.require_group(parent)

    # Copy everything except attributes. Dimension-scale attributes contain
    # source-file object references and must be recreated.
    fsrc.copy(
        path,
        fdst,
        path,
        expand_refs=True,
        without_attrs=True,
    )

    # Mark dimension-scale datasets before attaching them.
    for scale_path, scale_name in scale_names.items():
        fdst[scale_path].make_scale(scale_name)

    # Recreate scale attachments and dimension labels.
    for dataset_path, index, label, scale_paths in dimensions:
        dimension = fdst[dataset_path].dims[index]

        for scale_path in scale_paths:
            dimension.attach_scale(fdst[scale_path])

        if label:
            dimension.label = label

    # Copy all ordinary attributes while preserving their dtype and shape.
    for src_obj in objects:
        dst_obj = fdst[src_obj.name]

        for name in src_obj.attrs:
            if (
                name in _DIMENSION_SCALE_ATTRS
                or (name == "CLASS" and name in dst_obj.attrs)
            ):
                continue

            attr_id = src_obj.attrs.get_id(name)
            value = src_obj.attrs[name]

            if (
                isinstance(src_obj, h5py.Dataset)
                and name in ("_Netcdf4Dimid", "_Netcdf4Coordinates")
            ):
                value = value + dimid_offset

            dst_obj.attrs.create(
                name,
                value,
                shape=attr_id.shape,
                dtype=attr_id.dtype,
            )


def copy_h5_obj(fsrc: h5py.File, fdst: h5py.File, path: str) -> None:
    objtype = fsrc[path].attrs.get("_damnit_objtype", "")

    if objtype in (DataType.DataArray.value, DataType.Dataset.value):
        copy_h5_obj_rebuild_dimscales(fsrc, fdst, path)
    else:
        fsrc.copy(path, fdst, path, expand_refs=True)


def fragment_variables(fsrc: h5py.File) -> set[str]:
    """Return names of all modified variables from the fragment file.
    """
    names = {key for key in fsrc if not key.startswith(".")}
    for special_grp in SPECIAL_GROUPS:
        names |= set(fsrc.get(special_grp, ()))
    return names


def delete_variable(fdst: h5py.File, name):
    fdst.pop(name, None)
    for special_grp in SPECIAL_GROUPS:
        fdst.pop(f"{special_grp}/{name}", None)


def copy_variable(fsrc: h5py.File, fdst: h5py.File, name: str):
    if name in fsrc:
        copy_h5_obj(fsrc, fdst, name)

    for special_grp in SPECIAL_GROUPS:
        if name in fsrc.get(special_grp, ()):
            copy_h5_obj(fsrc, fdst, f"{special_grp}/{name}") 


def combine(src: Path, dst: Path):
    """Combine the contents of ``src`` (an HDF5 file) into ``dst``.

    The source is deliberately not deleted here.  The combiner service deletes
    it only after the database update and notification have succeeded, while
    offline callers do the same after their own finalization.
    """
    # If a previous attempt created the destination by hard-linking the
    # fragment, the two paths name the same file.  This can happen when the
    # service is restarted after the HDF5 merge but before finalization.
    try:
        if src.samefile(dst):
            log.debug("Source %r and destination %r are already the same file", src, dst)
            return
    except FileNotFoundError:
        pass

    # Shortcut: if the destination file doesn't exist, hard-link the source.
    try:
        dst.hardlink_to(src)
    except FileExistsError:
        pass
    else:
        log.debug("Created %r by hard-linking %r", dst, src)
        return

    t0 = time.perf_counter()
    with h5py.File(src) as fsrc, h5py.File(dst, 'r+') as fdst:
        for name in fragment_variables(fsrc):
            delete_variable(fdst, name)
            copy_variable(fsrc, fdst, name)

    t1 = time.perf_counter()
    log.debug("Finished combining in %.3f s", t1 - t0)


@dataclass
class CombinedFragment:
    """Picklable data returned by a worker after an HDF5 merge."""

    reduced_data: dict
    provenance: str


@dataclass
class RecordState:
    topic_partition: TopicPartition
    offset: int
    completed: bool = False


@dataclass
class SubmissionJob:
    source: Path
    destination: Path
    damnit_dir: Path
    proposal: int
    run: int
    msg_timestamp: datetime
    record: RecordState | None = None


def wait_file_exists(p: Path, msg_timestamp: datetime):
    if p.exists():
        return True

    time_limit = msg_timestamp + timedelta(seconds=NO_FILE_TIMEOUT)
    t0 = time.monotonic()
    while not p.exists():
        if datetime.now(timezone.utc) > time_limit:
            return False
        time.sleep(0.5)

    t1 = time.monotonic()
    since_msg = datetime.now(timezone.utc) - msg_timestamp
    log.info(
        "Fragment file %s found after %d s (%d s after notification)",
        p, (t1 - t0), since_msg.total_seconds()
    )
    return True


def offset_metadata(offset: int) -> OffsetAndMetadata:
    """Build offset metadata for supported kafka-python versions."""
    try:
        # Recent kafka-python releases include the leader epoch in this
        # namedtuple.  -1 means that the consumer has no epoch to provide.
        return OffsetAndMetadata(offset, None, -1)
    except TypeError:
        return OffsetAndMetadata(offset, None)


def combine_fragment_worker(
    src: Path, dst: Path, msg_timestamp: datetime
) -> CombinedFragment | None:
    """Wait for, read, and combine one fragment in a worker process."""
    if not wait_file_exists(src, msg_timestamp):
        return None

    with h5py.File(src) as f:
        provenance = f.attrs.get("provenance", "")
    new_data = load_reduced_data(src)
    combine(src, dst)

    # Returning only ordinary, picklable values keeps HDF5 handles and the
    # processor's Kafka clients in the coordinator process.
    return CombinedFragment(new_data, provenance)


class FileSubmissionProcessor:
    def __init__(
        self,
        consumer_config=None,
        *,
        workers=DEFAULT_WORKERS,
        max_pending=DEFAULT_MAX_PENDING,
        poll_timeout_ms=DEFAULT_POLL_TIMEOUT_MS,
        executor=None,
    ):
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if max_pending < 1:
            raise ValueError("max_pending must be at least 1")
        if poll_timeout_ms < 1:
            raise ValueError("poll_timeout_ms must be at least 1")

        self.workers = workers
        self.max_pending = max_pending
        self.poll_timeout_ms = poll_timeout_ms

        consumer_config = {
            # The coordinator polls regularly while workers copy files, so
            # heartbeats do not depend on the duration of an HDF5 merge.
            'session_timeout_ms': 120_000,
            # Keep the scheduler's pending bound exact.  A single record also
            # makes arrival order unambiguous for each destination.
            'max_poll_records': 1,
            'enable_auto_commit': False,
            # Default timeout is 300s (5 minutes), which should be enough.
        } | (consumer_config or {})
        # This service commits only after finalization.  Do not allow a caller
        # to accidentally turn Kafka's background commits back on.
        consumer_config['enable_auto_commit'] = False

        self.consumer = KafkaConsumer(
            FILE_SUBMIT_TOPIC,
            bootstrap_servers=update_brokers(),
            group_id='xfel-da-damnit-combiner',
            consumer_timeout_ms=600_000,
            **consumer_config
        )
        self.producer = KafkaProducer(
            bootstrap_servers=update_brokers(),
            value_serializer=lambda d: json.dumps(d).encode('utf-8')
        )

        self.pool = executor or ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        self._owns_pool = executor is None

        # Destination scheduling state.  A destination has at most one active
        # future; later fragments remain in its coordinator-owned FIFO queue.
        self.pending: dict[Path, deque[SubmissionJob]] = {}
        self.active: dict[Path, tuple[SubmissionJob, Future]] = {}
        self._inflight_count = 0

        # Offset state is kept separately from jobs because different
        # destinations can complete out of order while each Kafka partition
        # must still be committed as one contiguous prefix.
        self._partition_states: dict[TopicPartition, dict[int, RecordState]] = {}
        self._next_commit_offsets: dict[TopicPartition, int] = {}
        self._paused_partitions: set[TopicPartition] = set()
        self._fatal_error = None
        self._shutdown = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        if self._shutdown:
            return
        self._shutdown = True
        self.consumer.close()
        if self._owns_pool:
            # Do not leave old workers running across a service restart.  Any
            # completed-but-unfinalized result remains uncommitted and will be
            # replayed from Kafka on the next start.
            try:
                self.pool.shutdown(wait=True, cancel_futures=True)
            except TypeError:  # pragma: no cover - for older executors
                self.pool.shutdown(wait=True)
        self.producer.flush(timeout=10)
        self.producer.close(timeout=10)

    def run(self):
        while True:
            self._check_completed()
            self._schedule_available()
            self._update_consumer_flow()

            for record in self._poll_records():
                self._accept_record(record)

            # A worker can finish during the poll timeout.  Handle it without
            # waiting for another Kafka record to arrive.
            self._check_completed()

    def handle_one_message(self):
        """Process one polled record and wait for it to be finalized.

        This is retained as a small synchronous convenience for callers and
        tests.  The service entry point uses :meth:`run`, which continues
        polling while worker futures are in flight.
        """
        self._check_completed()
        self._schedule_available()
        self._update_consumer_flow()
        records = list(self._poll_records())
        if not records:
            return False

        states = [self._accept_record(record) for record in records]
        self._schedule_available()
        self._wait_for_records(states)
        return True

    def process_file_submission_msg(self, d: dict, msg_timestamp: datetime):
        """Process one submission synchronously for direct callers."""
        job = self._make_job(d, msg_timestamp)
        log.info(
            "Combining %r into %r; event pending for %.3f s before processing",
            job.source, job.destination,
            (datetime.now(timezone.utc) - msg_timestamp).total_seconds()
        )

        result = self.pool.submit(
            combine_fragment_worker,
            job.source,
            job.destination,
            job.msg_timestamp,
        ).result()
        self._finalize_job(job, result)

    @staticmethod
    def wait_file_exists(p: Path, msg_timestamp: datetime):
        return wait_file_exists(p, msg_timestamp)

    def send_update(self, reduced_data, topic, proposal, run):
        update_msg = msg_dict(MsgKind.run_values_updated, {
            'run': run, 'proposal': proposal, 'values': {
                name: None for name in reduced_data.keys()
            }
        })
        return self.producer.send(topic, update_msg)

    def _make_job(self, d: dict, msg_timestamp: datetime, record=None):
        damnit_dir = Path(d['damnit_dir']).resolve()
        destination = (
            damnit_dir
            / "extracted_data"
            / f"p{d['proposal']}_r{d['run']}.h5"
        ).resolve()
        return SubmissionJob(
            source=Path(d['new_file']).resolve(),
            destination=destination,
            damnit_dir=damnit_dir,
            proposal=d['proposal'],
            run=d['run'],
            msg_timestamp=msg_timestamp,
            record=record,
        )

    def _poll_records(self):
        try:
            records = self.consumer.poll(
                timeout_ms=self.poll_timeout_ms,
                max_records=1,
            )
        except TypeError:
            # Keep simple consumer doubles compatible with the coordinator's
            # polling contract.
            records = self.consumer.poll(timeout_ms=self.poll_timeout_ms)

        for partition_records in records.values():
            yield from partition_records

    @staticmethod
    def _record_timestamp(record):
        timestamp = getattr(record, "timestamp", None)
        if timestamp is None or timestamp < 0:
            return datetime.now(timezone.utc)
        return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)

    @staticmethod
    def _record_message(record):
        value = record.value
        if isinstance(value, bytes):
            value = value.decode()
        return json.loads(value)

    def _record_state(self, record):
        topic_partition = TopicPartition(
            getattr(record, "topic", FILE_SUBMIT_TOPIC),
            getattr(record, "partition", 0),
        )
        offset = record.offset
        partition_states = self._partition_states.setdefault(topic_partition, {})
        state = partition_states.get(offset)
        if state is None:
            state = partition_states[offset] = RecordState(topic_partition, offset)
            self._next_commit_offsets.setdefault(topic_partition, offset)
        return state

    def _accept_record(self, record):
        state = self._record_state(record)
        try:
            msg = self._record_message(record)
            if msg['msg_kind'] != MsgKind.file_submission.value:
                log.info("Unexpected message kind %r", msg['msg_kind'])
                self._mark_record_complete(state)
                return state

            job = self._make_job(msg['data'], self._record_timestamp(record), state)
            queue = self.pending.setdefault(job.destination, deque())
            queue.append(job)
            self._inflight_count += 1
            log.info(
                "Queued %r for %r; event pending for %.3f s before processing",
                job.source, job.destination,
                (datetime.now(timezone.utc) - job.msg_timestamp).total_seconds()
            )
            return state
        except Exception as exc:
            self._fatal_error = exc
            log.error("Unexpected error accepting file submission message", exc_info=True)
            raise

    def _schedule_available(self):
        if self._fatal_error is not None:
            return

        while len(self.active) < self.workers:
            for destination, queue in self.pending.items():
                if destination in self.active or not queue:
                    continue

                job = queue[0]
                try:
                    future = self.pool.submit(
                        combine_fragment_worker,
                        job.source,
                        job.destination,
                        job.msg_timestamp,
                    )
                except Exception as exc:
                    self._fatal_error = exc
                    log.error("Unable to submit combiner worker", exc_info=True)
                    raise

                queue.popleft()
                self.active[destination] = (job, future)
                break
            else:
                break

    def _check_completed(self):
        if self._fatal_error is not None:
            raise RuntimeError("Combiner stopped after an earlier failure") from self._fatal_error

        for destination, (job, future) in list(self.active.items()):
            if not future.done():
                continue

            try:
                result = future.result()
                self._finalize_job(job, result)
                if job.record is not None:
                    self._mark_record_complete(job.record)
            except Exception as exc:
                # Keep this destination active.  Its queued fragments must not
                # be submitted after a failed merge or finalization.
                self._fatal_error = exc
                log.error(
                    "Unexpected error finalizing fragment %s for %s",
                    job.source, destination, exc_info=True,
                )
                raise

            self.active.pop(destination)
            self._inflight_count -= 1
            if not self.pending.get(destination):
                self.pending.pop(destination, None)

        self._schedule_available()

    def _finalize_job(self, job: SubmissionJob, result: CombinedFragment | None):
        if result is None:
            log.warning(
                "File %s not present %d s after notification, skipping",
                job.source, NO_FILE_TIMEOUT,
            )
            return

        with DamnitDB.from_dir(job.damnit_dir) as db:
            log.info(
                "Updating database in %s with %s variables for p%d r%d from %s",
                job.damnit_dir, len(result.reduced_data),
                job.proposal, job.run, result.provenance,
            )
            add_to_db(
                result.reduced_data,
                db,
                job.proposal,
                job.run,
                provenance=result.provenance,
            )
            notification = self.send_update(
                result.reduced_data,
                db.kafka_topic,
                job.proposal,
                job.run,
            )
            if notification is not None and hasattr(notification, "get"):
                notification.get(timeout=NOTIFICATION_TIMEOUT)

        # The destination remains through a hard link when it was newly
        # created.  Removing only the fragment path is therefore safe.
        if job.source != job.destination:
            job.source.unlink(missing_ok=True)

    def _mark_record_complete(self, state: RecordState):
        state.completed = True
        states = self._partition_states[state.topic_partition]
        next_offset = self._next_commit_offsets[state.topic_partition]
        old_next_offset = next_offset

        while (next_state := states.get(next_offset)) is not None and next_state.completed:
            del states[next_offset]
            next_offset += 1

        self._next_commit_offsets[state.topic_partition] = next_offset
        if next_offset == old_next_offset:
            return

        self.consumer.commit(offsets={
            state.topic_partition: offset_metadata(next_offset),
        })

    def _update_consumer_flow(self):
        should_pause = self._inflight_count >= self.max_pending
        assigned = set(self.consumer.assignment())

        if should_pause:
            to_pause = assigned - self._paused_partitions
            if to_pause:
                self.consumer.pause(*to_pause)
                self._paused_partitions.update(to_pause)
        elif self._paused_partitions:
            self.consumer.resume(*self._paused_partitions)
            self._paused_partitions.clear()

    def _wait_for_records(self, states):
        while not all(state.completed for state in states):
            self._check_completed()
            if not all(state.completed for state in states):
                time.sleep(0.01)


def gather_all_fragments(damnit_dir: Path):
    h5_dir = damnit_dir / "extracted_data"

    frag_files_matches = sorted(
        [(p, m) for p in h5_dir.iterdir() if (m := FRAGMENT_PATTERN.match(p.name))],
        # Sort by mtime to process files in order written
        key=lambda t: t[0].stat().st_mtime
    )

    with DamnitDB.from_dir(damnit_dir) as db:
        for p, m in frag_files_matches:
            proposal = int(m[1])
            run = int(m[2])

            new_data = load_reduced_data(p)
            with h5py.File(p) as f:
                provenance = f.attrs.get("provenance", "")
            combine(p, h5_dir / f"p{proposal}_r{run}.h5")
            add_to_db(new_data, db, proposal, run, provenance=provenance)
            p.unlink(missing_ok=True)


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main(*, workers=DEFAULT_WORKERS, max_pending=DEFAULT_MAX_PENDING):
    logging.basicConfig(level=logging.DEBUG)
    # Exclude debug level logging from kafka-python
    logging.getLogger('kafka').setLevel(logging.INFO)

    # Treat SIGTERM like SIGINT (Ctrl-C) & do a clean shutdown
    signal.signal(signal.SIGTERM, interrupted)

    with FileSubmissionProcessor(
        workers=workers,
        max_pending=max_pending,
    ) as processor:
        notify_ready()
        try:
            log.info("Waiting for file submission messages")
            processor.run()
        except KeyboardInterrupt:
            log.info("Stopping on Ctrl + C")
        except Exception:
            log.error("Stopping on unexpected error", exc_info=True)
            raise


if __name__ == "__main__":
    main()
