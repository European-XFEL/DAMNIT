import argparse
import getpass
import os
import functools
import logging
import pickle
import re
import shlex
import sqlite3
import subprocess
import sys
import time
from ctypes import CDLL
from datetime import datetime, timezone
from pathlib import Path

import extra_data
import h5py
import numpy as np
import xarray
from scipy import ndimage
from kafka import KafkaProducer

from ..context import ContextFile, RunData
from ..definitions import UPDATE_BROKERS, UPDATE_TOPIC
from .db import open_db, get_meta

log = logging.getLogger(__name__)


# Python innetgr wrapper after https://github.com/wcooley/netgroup-python/
def innetgr(netgroup: bytes, host=None, user=None, domain=None):
    libc = CDLL("libc.so.6")
    return bool(libc.innetgr(netgroup, host, user, domain))

def default_slurm_partition():
    username = getpass.getuser().encode()
    if innetgr(b'exfel-wgs-users', user=username):
        return 'exfel'
    elif innetgr(b'upex-users', user=username):
        return 'upex'
    return 'all'

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

def get_proposal_path(xd_run):
    files = [f.filename for f in xd_run.files]
    p = Path(files[0])

    return Path(*p.parts[:7])

def add_to_h5_file(path) -> h5py.File:
    """Open the file with exponential backoff if it's locked"""
    for i in range(6):
        try:
            return h5py.File(path, 'a')
        except BlockingIOError as e:
            # File is locked for writing; wait 1, 2, 4, ... seconds
            time.sleep(2 ** i)
    raise e

class Results:
    def __init__(self, data, ctx):
        self.data = data
        self.ctx = ctx
        self._reduced = None

    @classmethod
    def create(cls, ctx_file: ContextFile, xd_run, run_number, proposal):
        res = {'start_time': np.asarray(get_start_time(xd_run))}

        for name in ctx_file.ordered_vars():
            var = ctx_file.vars[name]

            try:
                # Add all variable dependencies
                kwargs = { arg_name: res.get(dep_name)
                           for arg_name, dep_name in var.arg_dependencies().items() }

                # If any are None, skip this variable since we're missing a dependency
                missing_deps = [key for key, value in kwargs.items() if value is None]
                if len(missing_deps) > 0:
                    log.warning(f"Skipping {name} because of missing dependencies: {', '.join(missing_deps)}")
                    continue

                # And all meta dependencies
                for arg_name, annotation in var.annotations().items():
                    if not annotation.startswith("meta#"):
                        continue

                    if annotation == "meta#run_number":
                        kwargs[arg_name] = run_number
                    elif annotation == "meta#proposal":
                        kwargs[arg_name] = proposal
                    elif annotation == "meta#proposal_path":
                        kwargs[arg_name] = get_proposal_path(xd_run)
                    else:
                        raise RuntimeError(f"Unknown path '{annotation}' for variable '{var.title}'")

                func = functools.partial(var.func, **kwargs)

                data = func(xd_run)
                if not isinstance(data, (xarray.DataArray, str, type(None))):
                    data = np.asarray(data)
            except Exception:
                log.error("Could not get data for %s", name, exc_info=True)
            else:
                # Only save the result if it's not None
                if data is not None:
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

    @property
    def reduced(self):
        if self._reduced is None:
            r = {}
            for name in self.data:
                v = self.summarise(name)
                if v is not None:
                    r[name] = v
            self._reduced = r
        return self._reduced

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

            if isinstance(data, xarray.DataArray):
                data = data.data

            return getattr(np, summary_method)(data)

    def save_hdf5(self, hdf5_path):
        dsets = [(f'.reduced/{name}', v) for name, v in self.reduced.items()]
        for name, arr in self.data.items():
            dsets.extend(self._datasets_for_arr(name, arr))

        log.info("Writing %d variables to %d datasets in %s",
                 len(self.data), len(dsets), hdf5_path)

        # We need to open the files in append mode so that when proc Variable's
        # are processed after raw ones, the raw ones won't be lost.
        with add_to_h5_file(hdf5_path) as f:
            # Create datasets before filling them, so metadata goes near the
            # start of the file.
            for path, arr in dsets:
                # Delete the existing datasets so we can overwrite them
                if path in f:
                    del f[path]

                if isinstance(arr, str):
                    f.create_dataset(path, shape=(1,), dtype=h5py.string_dtype())
                else:
                    f.create_dataset(path, shape=arr.shape, dtype=arr.dtype)

            for path, arr in dsets:
                f[path][()] = arr

        os.chmod(hdf5_path, 0o666)


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

    for key, value in db_data.items():
        # For convenience, all returned values from Variables are stored in
        # arrays. If a Variable returns a scalar then we'll end up with a
        # zero-dimensional array here which will be pickled, so we unbox all
        # zero-dimensional arrays first.
        if isinstance(value, (np.ndarray, np.generic)) and value.ndim == 0:
            value = value.item()

        # Serialize non-SQLite-supported types
        if not isinstance(value, (type(None), int, float, str, bytes)):
            value = pickle.dumps(value)

        db_data[key] = value

    with db:
        db.execute(f"""
            INSERT INTO runs (proposal, runnr, {cols_sql})
            VALUES (:proposal, :run, {values_sql})
            ON CONFLICT (proposal, runnr) DO UPDATE SET {updates_sql}
        """, db_data)


class Extractor:
    _proposal = None

    def __init__(self):
        self.db = open_db()
        self.kafka_prd = KafkaProducer(
            bootstrap_servers=UPDATE_BROKERS,
            value_serializer=lambda d: pickle.dumps(d),
        )
        self.update_topic = UPDATE_TOPIC.format(get_meta(self.db, 'db_id'))
        self.ctx_whole = ContextFile.from_py_file(Path('context.py'))

    @property
    def proposal(self):
        if self._proposal is None:
            self._proposal = get_meta(self.db, 'proposal')
        return self._proposal

    def slurm_options(self):
        if reservation := get_meta(self.db, 'slurm_reservation', ''):
            return ['--reservation', reservation]
        partition = get_meta(self.db, 'slurm_partition', '') or default_slurm_partition()
        return ['--partition', partition]

    def extract_and_ingest(self, proposal, run, cluster=False,
                           run_data=RunData.ALL, match=()):
        if proposal is None:
            proposal = self.proposal

        with self.db:
            self.db.execute("""
                INSERT INTO runs (proposal, runnr, added_at) VALUES (?, ?, ?)
                ON CONFLICT (proposal, runnr) DO NOTHING
            """, (proposal, run, datetime.now(tz=timezone.utc).timestamp()))
        log.info("Ensured p%d r%d in database", proposal, run)

        out_path = Path('extracted_data', f'p{proposal}_r{run}.h5')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(out_path.parent, 0o777)

        ctx = self.ctx_whole.filter(run_data=run_data, cluster=cluster, name_matches=match)

        run_dc = extra_data.open_run(proposal, run, data="all")
        res = Results.create(ctx, run_dc, run, proposal)
        res.save_hdf5(out_path)
        reduced_data = res.reduced
        log.info("Reduced data has %d fields", len(reduced_data))
        add_to_db(reduced_data, self.db, proposal, run)

        # Send update via Kafka
        reduced_data['Proposal'] = proposal
        reduced_data['Run'] = run
        self.kafka_prd.send(self.update_topic, reduced_data).get(timeout=30)
        log.info("Sent Kafka update to topic %r", self.update_topic)

        # Launch a Slurm job if there are any 'cluster' variables to evaluate
        ctx_slurm = self.ctx_whole.filter(run_data=run_data, name_matches=match, cluster=True)
        if set(ctx_slurm.vars) > set(ctx.vars):
            python_cmd = [sys.executable, '-m', 'amore_mid_prototype.backend.extract_data',
                          '--cluster-job', str(proposal), str(run), run_data.value]
            res = subprocess.run([
                'sbatch', '--parsable',
                *self.slurm_options(),
                '--wrap', shlex.join(python_cmd)
            ], stdout=subprocess.PIPE, text=True)
            job_id = res.stdout.partition(';')[0]
            log.info("Launched Slurm job %s to calculate cluster variables", job_id)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('proposal', type=int)
    ap.add_argument('run', type=int)
    ap.add_argument('run_data', choices=('raw', 'proc', 'all'))
    ap.add_argument('--cluster-job', action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    Extractor().extract_and_ingest(args.proposal, args.run,
                                   cluster=args.cluster_job,
                                   run_data=RunData(args.run_data))
