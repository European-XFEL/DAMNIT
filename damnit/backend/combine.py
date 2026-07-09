"""Read temporary HDF5 files, combine them into a single file per run, updating DB"""
import multiprocessing

import json
import logging
import re
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5netcdf
import h5py
import xarray as xr
from kafka import KafkaConsumer, KafkaProducer
from xarray.backends import H5NetCDFStore

from ..context import DataType
from ..definitions import FILE_SUBMIT_TOPIC, update_brokers
from .db import DamnitDB, MsgKind, msg_dict
from .extract_data import add_to_db, load_reduced_data
from .service import notify_ready

FRAGMENT_PATTERN = re.compile(r"p(\d+)_r(\d+).(.+).ready.h5$")
SPECIAL_GROUPS = (".reduced", ".preview", ".errors")

# If the file doesn't exist N seconds after the Kafka message was sent, move on
NO_FILE_TIMEOUT = 30

# The combiner service will spread jobs over this many worker processes
NUM_COMBINER_PROCESSES = 16

log = logging.getLogger(__name__)


# HDF5's copy function (H5Ocopy) doesn't work correctly with dimension scales,
# which are used on NetCDF4 data (saved xarray objects). We work around that by
# reading such objects back to an xarray dataset and saving them again.
# https://github.com/HDFGroup/hdf5/issues/6200
def copy_h5_obj(fsrc: h5py.File, fdst: h5py.File, path: str):
    objtype = fsrc[path].attrs.get('_damnit_objtype', '')
    if objtype in (DataType.DataArray.value, DataType.Dataset.value):
        with h5netcdf.File(fsrc) as nf:
            with H5NetCDFStore(nf, path, mode='r') as store:
                xobj = xr.load_dataset(store)

        with h5netcdf.File(fdst, 'a') as nf:
            with H5NetCDFStore(nf, path, mode='a') as store:
                xobj.dump_to_store(store)
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
    """Combine the the contents of src (an HDF5 file) into dst"""
    # Shortcut: if the destination file doesn't exist, rename src
    try:
        dst.hardlink_to(src)
    except FileExistsError:
        pass
    else:
        src.unlink()
        log.debug("Created %r by renaming", dst)
        return

    with h5py.File(src) as fsrc, h5py.File(dst, 'r+') as fdst:
        for name in fragment_variables(fsrc):
            delete_variable(fdst, name)
            copy_variable(fsrc, fdst, name)

    src.unlink()


class FileSubmissionProcessor:
    def __init__(self, consumer_config=None, n_workers=NUM_COMBINER_PROCESSES):
        self.consumer = KafkaConsumer(
            FILE_SUBMIT_TOPIC,
            bootstrap_servers=update_brokers(),
            group_id='xfel-da-damnit-combiner',
            consumer_timeout_ms=15_000,
            **(consumer_config or {})
        )
        # forkserver doesn't work for the tests, because it doesn't pick up
        # changing environment variables after the server process starts.
        self.mp_ctx = mp_ctx = multiprocessing.get_context('spawn')
        self.workers = []
        for i in range(n_workers):
            self.workers.append(cw := CombinerWorker(f"combiner-worker-{i}", mp_ctx))
            cw.start_process()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        self.consumer.close()
        for w in self.workers:
            w.stop()
        for w in self.workers:
            w.process.join()
            w.task_q.close()
            w.task_q.join_thread()

    def wait_all_done(self):  # For tests
        for w in self.workers:
            w.task_q.join()

    def run(self):
        while True:
            try:
                self.handle_one_message()
            except StopIteration:
                pass  # consumer timeout was reached

            self.ensure_workers()

    def ensure_workers(self):
        for w in self.workers:
            if not w.process.is_alive():
                log.warning(f"{w.name} died unexpectedly; restarting")
                # Also recreate the queue to reset the task_done count
                w.task_q.close()
                w.task_q = self.mp_ctx.JoinableQueue()
                w.start_process()

    def handle_one_message(self):
        record  = next(self.consumer)
        try:
            ts = datetime.fromtimestamp(record.timestamp / 1000, tz=timezone.utc)
            msg = json.loads(record.value.decode())
            if msg['msg_kind'] == MsgKind.file_submission.value:
                self.process_file_submission_msg(msg['data'], ts)
            else:
                log.info("Unexpected message kind %r", msg['msg_kind'])
        except Exception:
            log.error(
                "Unexpected error processing file submission message",
                exc_info=True,
            )

    def process_file_submission_msg(self, d: dict, msg_timestamp: datetime):
        # Assign the task to worker deterministically based on the destination
        # HDF5 file. This ensures jobs for the same destination file can't run
        # in parallel, avoiding locking/corruption issues.
        dst_key = d['damnit_dir'], d['proposal'], d['run']
        use_worker_num = hash(dst_key) % len(self.workers)
        log.info("Submission in %s (p%d r%d) assigned to worker %d",
                 *dst_key, use_worker_num)
        self.workers[use_worker_num].task_q.put((d, msg_timestamp))


class CombinerWorker:
    process = None

    def __init__(self, name, mp_ctx):
        self.name = name
        self.process_cls = mp_ctx.Process
        self.task_q = mp_ctx.JoinableQueue()

    def start_process(self):
        self.process = self.process_cls(name=self.name, target=self.proc_main, daemon=True)
        self.process.start()

    def stop(self):
        self.task_q.put(None)

    # Methods below run in the worker process -------------------------------

    def proc_main(self):
        producer = KafkaProducer(
            bootstrap_servers=update_brokers(),
            value_serializer=lambda d: json.dumps(d).encode('utf-8')
        )
        try:
            self.loop(producer)
        finally:
            producer.close(timeout=5)

    def loop(self, producer):
        while True:
            if (task := self.task_q.get()) is None:
                return

            try:
                d, msg_timestamp = task
                self.process_combine_task(d, msg_timestamp, producer)
            except Exception:
                log.error("Exception while combining file", exc_info=True)
            finally:
                self.task_q.task_done()

    def process_combine_task(self, d: dict, msg_timestamp: datetime, producer):
        """Handle a notification from Kafka"""
        damnit_dir = Path(d['damnit_dir'])
        h5_dir = damnit_dir / "extracted_data"
        src = Path(d['new_file'])
        dst = h5_dir / f"p{d['proposal']}_r{d['run']}.h5"
        log.info("Combining %r into %r", src, dst)

        prop, run = d['proposal'], d['run']

        if not self.wait_file_exists(src, msg_timestamp):
            log.warning(
                "File %s not present %d s after notification, skipping",
                src, NO_FILE_TIMEOUT
            )
            return

        with h5py.File(src) as f:
            provenance = f.attrs.get("provenance", "")
        new_data = load_reduced_data(src)
        combine(src, dst)

        with DamnitDB.from_dir(damnit_dir) as db:
            log.info("Updating database in %s with %s variables for p%d r%d from %s",
                     damnit_dir, len(new_data), prop, run, provenance)
            add_to_db(new_data, db, prop, run, provenance=provenance)
            producer.send(db.kafka_topic, update_msg(new_data, prop, run))
            producer.flush(timeout=5)

    @staticmethod
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


def update_msg(reduced_data, proposal: int, run: int):
    return msg_dict(MsgKind.run_values_updated, {
        'run': run, 'proposal': proposal, 'values': {
            name: None for name in reduced_data.keys()
        }
    })


def gather_all_fragments(damnit_dir: Path):
    db = DamnitDB.from_dir(damnit_dir)
    h5_dir = damnit_dir / "extracted_data"

    frag_files_matches = sorted(
        [(p, m) for p in h5_dir.iterdir() if (m := FRAGMENT_PATTERN.match(p.name))],
        # Sort by mtime to process files in order written
        key=lambda t: t[0].stat().st_mtime
    )

    for p, m in frag_files_matches:
        proposal = int(m[1])
        run = int(m[2])

        new_data = load_reduced_data(p)
        with h5py.File(p) as f:
            provenance = f.attrs.get("provenance", "")
        combine(p, h5_dir / f"p{proposal}_r{run}.h5")
        add_to_db(new_data, db, proposal, run, provenance=provenance)


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main():
    logging.basicConfig(level=logging.DEBUG)
    # Exclude debug level logging from kafka-python
    logging.getLogger('kafka').setLevel(logging.INFO)

    # Treat SIGTERM like SIGINT (Ctrl-C) & do a clean shutdown
    signal.signal(signal.SIGTERM, interrupted)

    with FileSubmissionProcessor() as processor:
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
