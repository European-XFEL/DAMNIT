import os
import logging
import pickle
import re
import sqlite3
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import extra_data
import h5py
import numpy as np
import xarray
from scipy import ndimage
from kafka import KafkaProducer

from ..context import ContextFile
from ..definitions import UPDATE_BROKERS, UPDATE_TOPIC
from .db import open_db, get_meta

log = logging.getLogger(__name__)


class RunData(Enum):
    RAW = "raw"
    PROC = "proc"
    ALL = "all"

def get_start_time(xd_run):
    ts = xd_run.select_trains(np.s_[:1]).train_timestamps()[0]

    if np.isnan(ts):
        # If the timestamp information is not present (i.e. on old files), then
        # we take the timestamp from the oldest raw file as an approximation.
        files = sorted([f.filename for f in xd_run.files if "raw" in f.filename])
        first_file = Path(files[0])

        # Use the modified timestamp
        return first_file.stat().st_mtime
    else:
        # Convert np datetime64 [ns] -> [us] -> datetime -> float  :-/
        return np.datetime64(ts, 'us').item().timestamp()


class Results:
    def __init__(self, data, ctx):
        self.data = data
        self.ctx = ctx

    @classmethod
    def create(cls, ctx_file: ContextFile, xd_run):
        res = {'start_time': np.asarray(get_start_time(xd_run))}
        for name, var in ctx_file.vars.items():
            try:
                data = var.func(xd_run)
                if not isinstance(data, (xarray.DataArray, str)):
                    data = np.asarray(data)
            except Exception:
                log.error("Could not get data for %s", name, exc_info=True)
            else:
                res[name] = data
        return Results(res, ctx_file)

    @staticmethod
    def _datasets_for_arr(name, arr):
        if isinstance(arr, xarray.DataArray):
            return [
                (f'{name}/data', arr.values),
            ] + [
                (f'{name}/{dim}', coords.values)
                for dim, coords in arr.coords.items()
            ]
        else:
            value = arr if isinstance(arr, str) else np.asarray(arr)
            return [
                (f'{name}/data', value)
            ]

    def summarise(self, name):
        data = self.data[name]

        if isinstance(data, str):
            return data
        elif data.ndim == 0:
            return data
        elif data.ndim == 2 and self.ctx.vars[name].summary is None:
            # For the sake of space and memory we downsample images to a
            # resolution of 150x150.
            zoom_ratio = 150 / max(data.shape)
            if zoom_ratio < 1:
                data = ndimage.zoom(np.nan_to_num(data),
                                    zoom_ratio)

            return data
        else:
            summary_method = self.ctx.vars[name].summary
            if summary_method is None:
                return None
            return getattr(np, summary_method)(data)

    def save_hdf5(self, hdf5_path):
        dsets = []
        for name, arr in self.data.items():
            reduced = self.summarise(name)
            if reduced is not None:
                dsets.append((f'.reduced/{name}', reduced))
            dsets.extend(self._datasets_for_arr(name, arr))

        log.info("Writing %d variables to %d datasets in %s",
                 len(self.data), len(dsets), hdf5_path)

        # We need to open the files in append mode so that when proc Variable's
        # are processed after raw ones, the raw ones won't be lost.
        with h5py.File(hdf5_path, 'a') as f:
            # Create datasets before filling them, so metadata goes near the
            # start of the file.
            for path, arr in dsets:
                # Delete the existing datasets so we can overwrite them
                if path in f:
                    del f[path]

                if isinstance(arr, str):
                    f.create_dataset(path, shape=(1,), dtype=h5py.string_dtype(length=len(arr)))
                else:
                    f.create_dataset(path, shape=arr.shape, dtype=arr.dtype)

            for path, arr in dsets:
                f[path][()] = arr

        os.chmod(hdf5_path, 0o666)


def run_and_save(proposal, run, out_path, run_data=RunData.ALL, match=[]):
    run_dc = extra_data.open_run(proposal, run, data="all")

    ctx_file = ContextFile.from_py_file(Path('context.py'))

    # Filter variables
    for name in list(ctx_file.vars.keys()):
        title = ctx_file.vars[name].title or name
        var_data = ctx_file.vars[name].data

        # If this is being triggered by a migration/calibration message for
        # raw/proc data, then only process the Variable's that require that data.
        data_mismatch = run_data != RunData.ALL and var_data != run_data.value
        # Skip Variable's that don't match the match list
        name_mismatch = len(match) > 0 and not any(m.lower() in title.lower() for m in match)

        if data_mismatch or name_mismatch:
            del ctx_file.vars[name]

    res = Results.create(ctx_file, run_dc)
    res.save_hdf5(out_path)


def load_reduced_data(h5_path):
    def get_dset_value(ds):
        # If it's a string, extract the string
        if h5py.check_string_dtype(ds.dtype) is not None:
            return ds.asstr()[0]
        else:
            value = ds[()]
            # SQlite doesn't like np.float32; .item() converts to Python numbers
            return value.item() if np.isscalar(value) else value

    with h5py.File(h5_path, 'r') as f:
        return {
            name: get_dset_value(dset) for name, dset in f['.reduced'].items()
        }

def add_to_db(reduced_data, db: sqlite3.Connection, proposal, run):
    log.info("Adding p%d r%d to database, with %d columns",
             proposal, run, len(reduced_data))

    # We're going to be formatting column names as strings into SQL code,
    # so check that they are simple identifiers before we get there.
    for name in reduced_data:
        assert re.match(r'[a-zA-Z][a-zA-Z0-9_]*$', name), f"Bad field name {name}"

    cursor = db.execute("SELECT * FROM runs")
    cols = [c[0] for c in cursor.description]

    for missing_col in set(reduced_data) - set(cols):
        db.execute(f"ALTER TABLE runs ADD COLUMN {missing_col}")

    col_names = list(reduced_data.keys())
    cols_sql = ", ".join(col_names)
    values_sql = ", ".join([f':{c}' for c in col_names])
    updates_sql = ", ".join([f'{c} = :{c}' for c in col_names])

    db_data = reduced_data.copy()
    db_data.update({'proposal': proposal, 'run': run})

    # Serialize non-SQLite-supported types
    for key, value in db_data.items():
        if not isinstance(value, (type(None), int, float, str, bytes)):
            db_data[key] = pickle.dumps(value)

    with db:
        db.execute(f"""
            INSERT INTO runs (proposal, runnr, {cols_sql})
            VALUES (:proposal, :run, {values_sql})
            ON CONFLICT (proposal, runnr) DO UPDATE SET {updates_sql}
        """, db_data)


def extract_and_ingest(proposal, run, run_data=RunData.ALL, match=[]):
    db = open_db()
    if proposal is None:
        proposal = get_meta(db, 'proposal')

    with db:
        db.execute("""
            INSERT INTO runs (proposal, runnr, added_at) VALUES (?, ?, ?)
            ON CONFLICT (proposal, runnr) DO NOTHING
        """, (proposal, run, datetime.now(tz=timezone.utc).timestamp()))
    log.info("Ensured p%d r%d in database", proposal, run)

    out_path = Path('extracted_data', f'p{proposal}_r{run}.h5')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(out_path.parent, 0o777)

    run_and_save(proposal, run, out_path, run_data, match)
    reduced_data = load_reduced_data(out_path)
    log.info("Reduced data has %d fields", len(reduced_data))
    add_to_db(reduced_data, db, proposal, run)

    # Send update via Kafka
    kafka_prd = KafkaProducer(
        bootstrap_servers=UPDATE_BROKERS,
        value_serializer=lambda d: pickle.dumps(d),
    )
    update_topic = UPDATE_TOPIC.format(get_meta(db, 'db_id'))
    reduced_data['Proposal'] = proposal
    reduced_data['Run'] = run
    kafka_prd.send(update_topic, reduced_data)
    log.info("Sent Kafka update")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    proposal = int(sys.argv[1])
    run = int(sys.argv[2])
    run_data = RunData(sys.argv[3])
    extract_and_ingest(proposal, run, run_data)
