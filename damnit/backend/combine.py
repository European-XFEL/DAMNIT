"""Read temporary HDF5 files, combine them into a single file per run, updating DB"""
import json
import logging
import multiprocessing
import os.path
import re
import signal
import time
from concurrent.futures import ProcessPoolExecutor
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


def _try_rename(src: Path, dst: Path) -> bool:
    try:
        dst.hardlink_to(src)
    except FileExistsError:
        return False
    else:
        src.unlink()
        log.debug("Created %r by renaming", dst)
        return True


def combine_contents(fsrc: h5py.File, fdst: h5py.File):
    for name in fragment_variables(fsrc):
        delete_variable(fdst, name)
        copy_variable(fsrc, fdst, name)


class FileSubmissionProcessor:
    def __init__(self, consumer_config=None):
        self.consumer = KafkaConsumer(
            FILE_SUBMIT_TOPIC,
            bootstrap_servers=update_brokers(),
            group_id='xfel-da-damnit-combiner',
            consumer_timeout_ms=600_000,
            **(consumer_config or {})
        )
        self.executor = ProcessPoolExecutor(
            max_workers=16,
            mp_context=multiprocessing.get_context('spawn'),
            max_tasks_per_child=100,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        self.consumer.close()
        # With the default wait=True, this will finish anything that has started
        # running before shutting down.
        self.executor.shutdown(cancel_futures=True)

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
                return self.executor.submit(process_file_submission_msg, msg['data'], ts)
            else:
                log.info("Unexpected message kind %r", msg['msg_kind'])
        except Exception:
            log.error(
                "Unexpected error processing file submission message",
                exc_info=True,
            )


def process_file_submission_msg(d: dict, msg_timestamp: datetime):
    """Handle a notification from Kafka. This runs in a worker process"""
    damnit_dir = Path(d['damnit_dir'])
    h5_dir = damnit_dir / "extracted_data"
    src = Path(d['new_file'])
    dst = h5_dir / f"p{d['proposal']}_r{d['run']}.h5"
    log.info("Combining %r into %r", src, dst)

    prop, run = d['proposal'], d['run']

    if not wait_file_exists(src, msg_timestamp):
        log.warning(
            "File %s not present %d s after notification, skipping",
            src, NO_FILE_TIMEOUT
        )
        return

    try:
        combine_one_file(src, dst, damnit_dir, prop, run)
    except BlockingIOError:
        log.warning(
            "Output file %s appears to be locked by another worker, not combining %s",
            dst, src
        )
        return

    # Now combine any further files with the same destination which another
    # worker may have given up on while this one had the lock.
    extra_srcs = sorted(dst.parent.glob(dst.stem + '*.ready.h5'), key=os.path.getmtime)
    for extra_src in extra_srcs:
        log.info("Found additional file %s waiting to be combined", extra_src)
        try:
            combine_one_file(extra_src, dst, damnit_dir, prop, run, dst_retries=1)
        except BlockingIOError:
            log.warning(
                "Output file %s appears to be locked by another worker, not combining %s",
                dst, src
            )


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


def combine_one_file(src, dst, damnit_dir, prop, run, dst_retries=10):
    with h5py.File(src) as fsrc:
        provenance = fsrc.attrs.get("provenance", "")
        new_data = load_reduced_data(fsrc)

        if not _try_rename(src, dst):
            with open_dst_file(dst, retries=dst_retries) as fdst:
                combine_contents(fsrc, fdst)

    with DamnitDB.from_dir(damnit_dir) as db:
        log.info("Updating database in %s with %s variables for p%d r%d from %s",
                 damnit_dir, len(new_data), prop, run, provenance)
        add_to_db(new_data, db, prop, run, provenance=provenance)
        send_update(new_data, db.kafka_topic, prop, run)


def open_dst_file(p: Path, retries=10):
    while True:
        try:
            return h5py.File(p, 'r+')
        except BlockingIOError:  # File locked by another process
            retries -= 1
            if retries <= 0:
                raise
            time.sleep(1)


def send_update(reduced_data, topic, proposal, run):
    update_msg = msg_dict(MsgKind.run_values_updated, {
        'run': run, 'proposal': proposal, 'values': {
            name: None for name in reduced_data.keys()
        }
    })
    producer = KafkaProducer(
        bootstrap_servers=update_brokers(),
        value_serializer=lambda d: json.dumps(d).encode('utf-8')
    )
    producer.send(topic, update_msg)
    producer.close(timeout=5)


def gather_all_fragments(damnit_dir: Path):
    h5_dir = damnit_dir / "extracted_data"

    frag_files_matches = sorted(
        [(p, m) for p in h5_dir.iterdir() if (m := FRAGMENT_PATTERN.match(p.name))],
        # Sort by mtime to process files in order written
        key=lambda t: t[0].stat().st_mtime
    )

    for p, m in frag_files_matches:
        proposal = int(m[1])
        run = int(m[2])

        dst = h5_dir / f"p{proposal}_r{run}.h5"
        combine_one_file(p, dst, damnit_dir, proposal, run, dst_retries=1)


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
