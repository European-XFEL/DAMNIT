"""Read temporary HDF5 files, combine them into a single file per run, updating DB"""
import json
import logging
import multiprocessing
import re
import queue
import signal
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread

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
    def __init__(self, consumer_config=None):
        self.consumer = KafkaConsumer(
            FILE_SUBMIT_TOPIC,
            bootstrap_servers=update_brokers(),
            group_id='xfel-da-damnit-combiner',
            consumer_timeout_ms=600_000,
            **(consumer_config or {})
        )
        self.task_q = multiprocessing.JoinableQueue(maxsize=64)
        self.finished_q = multiprocessing.SimpleQueue()
        self.holding = {}
        self.holding_lock = multiprocessing.Lock()
        self.finished_thread = Thread(target=self.watch_finished, daemon=True)
        self.finished_thread.start()
        self.workers = [CombinerWorker(
            name=f"combiner-worker-{i}", task_q=self.task_q, finished_q=self.finished_q,
        ) for i in range(16)]
        for w in self.workers:
            w.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        self.consumer.close()
        while True:
            try:  # Discard any tasks not yet started
                self.task_q.get_nowait()
                self.task_q.task_done()
            except queue.Empty:
                break
        for _ in self.workers:
            self.task_q.put(None)
        for w in self.workers:
            w.join(timeout=5)

        self.task_q.join()
        self.task_q.close()
        self.task_q.join_thread()

        self.finished_q.put(None)
        self.finished_thread.join()
        self.finished_q.close()


    def run(self):
        while True:
            try:
                self.handle_one_message()
            except StopIteration:
                pass  # consumer timeout was reached

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

    def process_file_submission_msg(self, d, msg_timestamp):
        damnit_dir = d['damnit_dir']
        prop, run = d['proposal'], d['run']
        key = (damnit_dir, prop, run)

        with self.holding_lock:
            if (deq := self.holding.get(key)) is not None:
                log.info("New submission for p%r r%r in %s will be held until another "
                         "job finishes for the same file", prop, run, damnit_dir)
                deq.append((d, msg_timestamp))
                return

            self.holding[key] = deque()
            self.task_q.put((d, msg_timestamp))

    def watch_finished(self):
        """Runs in thread to submit held tasks when one finishes"""
        while True:
            key = self.finished_q.get()  # damnit_dir, prop, run
            if key is None:
                return

            with self.holding_lock:
                if (deq := self.holding.get(key)) is None:
                    continue

                if len(deq) > 0:
                    self.task_q.put(deq.popleft())
                else:
                    # If the finished task was the last for this key, we don't
                    # need to hold up further jobs.
                    del self.holding[key]



class CombinerWorker(multiprocessing.Process):
    def __init__(self, task_q, finished_q, name=None):
        super().__init__(name=name, daemon=True)
        self.task_q = task_q
        self.finished_q = finished_q

    def run(self):
        prod = KafkaProducer(
            bootstrap_servers=update_brokers(),
            value_serializer=lambda d: json.dumps(d).encode('utf-8')
        )
        while True:
            if (task := self.task_q.get()) is None:
                self.task_q.task_done()
                return

            d, msg_timestamp = task
            try:
                damnit_dir = Path(d['damnit_dir'])
                src = Path(d['new_file'])
                prop, run = d['proposal'], d['run']
                self.process_one(damnit_dir, src, prop, run, msg_timestamp, prod)
            except Exception:
                log.error("Error combining file %r", d, exc_info=True)
            self.task_q.task_done()
            key = d.get('damnit_dir'), d.get('proposal'), d.get('run')
            self.finished_q.put(key)

    def process_one(self, damnit_dir, src, prop, run, msg_timestamp, producer):
        """Handle a notification from Kafka"""
        h5_dir = damnit_dir / "extracted_data"
        dst = h5_dir / f"p{prop}_r{run}.h5"
        log.info("Combining %r into %r", src, dst)

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
            update_msg = msg_dict(MsgKind.run_values_updated, {
                'run': run, 'proposal': prop, 'values': {
                    name: None for name in new_data.keys()
                }
            })
            producer.send(db.kafka_topic, update_msg)
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

    multiprocessing.set_start_method('forkserver')

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
