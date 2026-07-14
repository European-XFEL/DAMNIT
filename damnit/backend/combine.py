"""Read temporary HDF5 files, combine them into a single file per run, updating DB"""
import json
import logging
import posixpath
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

    t0 = time.perf_counter()
    with h5py.File(src) as fsrc, h5py.File(dst, 'r+') as fdst:
        for name in fragment_variables(fsrc):
            delete_variable(fdst, name)
            copy_variable(fsrc, fdst, name)

    src.unlink()
    t1 = time.perf_counter()
    log.debug("Finished combining in %.3f s", t1 - t0)


class FileSubmissionProcessor:
    def __init__(self, consumer_config=None):
        self.consumer = KafkaConsumer(
            FILE_SUBMIT_TOPIC,
            bootstrap_servers=update_brokers(),
            group_id='xfel-da-damnit-combiner',
            consumer_timeout_ms=600_000,
            **(consumer_config or {})
        )
        self.producer = KafkaProducer(
            bootstrap_servers=update_brokers(),
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

    def process_file_submission_msg(self, d: dict, msg_timestamp: datetime):
        """Handle a notification from Kafka"""
        damnit_dir = Path(d['damnit_dir'])
        h5_dir = damnit_dir / "extracted_data"
        src = Path(d['new_file'])
        dst = h5_dir / f"p{d['proposal']}_r{d['run']}.h5"
        log.info(
            "Combining %r into %r; event pending for %.3f s before processing",
            src, dst, (datetime.now(timezone.utc) - msg_timestamp).total_seconds()
        )

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
            self.send_update(new_data, db.kafka_topic, prop, run)

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

    def send_update(self, reduced_data, topic, proposal, run):
        update_msg = msg_dict(MsgKind.run_values_updated, {
            'run': run, 'proposal': proposal, 'values': {
                name: None for name in reduced_data.keys()
            }
        })
        self.producer.send(topic, update_msg)


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
