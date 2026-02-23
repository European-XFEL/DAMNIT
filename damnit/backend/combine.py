"""Read temporary HDF5 files, combine them into a single file per run, updating DB"""
import json
import logging
import re
import signal
from pathlib import Path

import h5py
import h5netcdf
import xarray as xr
from kafka import KafkaConsumer, KafkaProducer
from xarray.backends import H5NetCDFStore

from ..context import DataType
from ..definitions import FILE_SUBMIT_TOPIC, UPDATE_BROKERS
from .db import DamnitDB, MsgKind, msg_dict
from .extract_data import load_reduced_data, add_to_db
from .service import notify_ready

FRAGMENT_PATTERN = re.compile(r"p(\d+)_r(\d+).(.+).ready.h5$")

log = logging.getLogger(__name__)


# HDF5's copy function (H5Ocopy) doesn't work correctly with dimension scales,
# which are used on NetCDF4 data (saved xarray objects). We work around that by
# reading such objects back to an xarray dataset and saving them again.
# https://github.com/HDFGroup/hdf5/issues/6200
def copy_h5_obj(fsrc: h5py.File, fdst: h5py.File, path: str):
    fdst.pop(path, None)

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
        for grp in fsrc:
            if not grp.startswith("."):
                copy_h5_obj(fsrc, fdst, grp)

        for special_grp in [".reduced", ".preview", ".errors"]:
            for k in fsrc[special_grp]:
                copy_h5_obj(fsrc, fdst, f"{special_grp}/{k}")

    src.unlink()


class FileSubmissionProcessor:
    def __init__(self):
        self.consumer = KafkaConsumer(
            FILE_SUBMIT_TOPIC,
            bootstrap_servers=UPDATE_BROKERS,
            consumer_timeout_ms=600_000,
        )
        self.producer = KafkaProducer(
            bootstrap_servers=UPDATE_BROKERS,
            value_serializer=lambda d: json.dumps(d).encode('utf-8')
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        self.consumer.close()
        self.producer.flush(timeout=10)
        self.producer.close(timeout=10)

    def run(self):
        while True:
            for record in self.consumer:
                try:
                    msg = json.loads(record.value.decode())
                    self.process_file_submission_msg(msg)
                except Exception:
                    log.error(
                        "Unexpected error processing file submission message",
                        exc_info=True,
                    )

    def process_file_submission_msg(self, d: dict):
        """Handle a notification from Kafka"""
        damnit_dir = Path(d['damnit_dir'])
        db = DamnitDB.from_dir(damnit_dir)
        h5_dir = damnit_dir / "extracted_data"
        src = Path(d['new_file'])
        dst = h5_dir / f"p{d['proposal']}_r{d['run']}.h5"
        log.info(f"Combining %r into %r", src, dst)

        prop, run = d['proposal'], d['run']

        new_data = load_reduced_data(src)
        add_to_db(new_data, db, prop, run)
        self.send_update(new_data, db.kafka_topic, prop, run)
        combine(src, dst)

    def send_update(self, reduced_data, topic, proposal, run):
        update_msg = msg_dict(MsgKind.run_values_updated, {
            'run': run, 'proposal': proposal, 'values': {
                name: None for name in reduced_data.keys()
            }
        })
        self.producer.send(topic, update_msg)


def update_db(db: DamnitDB, proposal: int, run: int, src: Path):
    new_data = load_reduced_data(src)
    add_to_db(new_data, db, proposal, run)


def gather_all_fragments(damnit_dir: Path):
    db = DamnitDB.from_dir(damnit_dir)
    h5_dir = damnit_dir / "extracted_data"

    for p in h5_dir.iterdir():
        if not (m := FRAGMENT_PATTERN.match(p.name)):
            continue

        proposal = int(m[1])
        run = int(m[2])
        dst = h5_dir / f"p{proposal}_r{run}.h5"
        update_db(db, proposal, run, p)
        combine(p, dst)


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main():
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
