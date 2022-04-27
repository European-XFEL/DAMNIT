import pickle
import types
import logging
import re
import sqlite3
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from unittest.mock import patch

import extra_data
import h5py
import numpy as np
import xarray
from scipy import ndimage

from ..context import ContextFile, RectROI
from .db import open_db, get_meta

log = logging.getLogger(__name__)


def get_start_time(xd_run):
    ts = xd_run.select_trains(np.s_[:1]).train_timestamps()[0]
    # Convert np datetime64 [ns] -> [us] -> datetime -> float  :-/
    return np.datetime64(ts, 'us').item().timestamp()


class Results:
    def __init__(self, data, ctx):
        self.data = data
        self.ctx = ctx

    @classmethod
    def create(cls, ctx_file: ContextFile, xd_run, run_number, out_path):
        res = {'start_time': np.asarray(get_start_time(xd_run))}
        db = open_db()
        proposal = get_meta(db, "proposal")

        # First evaluate the parameters
        params = { }
        for name, param in ctx_file.params.items():
            result = None
            with db:
                cursor = db.cursor()
                try:
                    cursor.execute(
                        f"""
                        SELECT {name} FROM runs WHERE runnr=?
                        """, (run_number,))
                except sqlite3.OperationalError:
                    # This will fail if the column doesn't exist yet
                    pass
                else:
                    result = cursor.fetchone()
                finally:
                    cursor.close()

            try:
                if result is not None and result[0] is not None:
                    value = result[0]
                    if isinstance(value, bytes):
                        value = pickle.loads(value)

                    if isinstance(value, RectROI):
                        with h5py.File(out_path) as f:
                            value.image = f[name]["data"][:]

                    params[name] = value
                else:
                    # Store the raw output of the function in the params dict, this
                    # will get passed to the variables when they are executed.
                    params[name] = param.func(xd_run)
            except Exception:
                log.error(f"Could not evaluate parameters '{name}', skipping rest of run", exc_info=True)

                # Right now we don't do anything fancy to check which variables
                # depend on which parameters, so we can only fail on the whole
                # run.
                return Results(res, ctx_file)
            else:
                # But always convert the result to an array when preparing to
                # store it in the database.
                data = params[name]
                if not isinstance(data, xarray.DataArray) and not isinstance(data, RectROI):
                    data = np.asarray(data)

                res[name] = data

        for name, var in ctx_file.vars.items():
            try:
                # We make a copy of the function object so that we can inject
                # the result of the parameters into the globals dictionary.
                func = var.func
                new_globals = func.__globals__.copy()
                new_globals.update(params)
                patched_func = types.FunctionType(func.__code__,
                                                  new_globals,
                                                  func.__name__,
                                                  func.__defaults__,
                                                  func.__closure__)
                data = patched_func(xd_run)

                if not isinstance(data, xarray.DataArray):
                    data = np.asarray(data)
            except Exception:
                log.error("Could not get data for %s", name, exc_info=True)
            else:
                res[name] = data
        return Results(res, ctx_file)

    @staticmethod
    def _datasets_for_arr(name, arr):
        if isinstance(arr, RectROI):
            roi = arr
            return [
                (f"{name}/{roi.__class__.__name__}_coords", np.array([roi.x, roi.y, roi.width, roi.height]))
            ] + Results._datasets_for_arr(name, roi.image)
        elif isinstance(arr, xarray.DataArray):
            return [
                (f'{name}/data', arr.values),
            ] + [
                (f'{name}/{dim}', coords.values)
                for dim, coords in arr.coords.items()
            ]
        else:
            return [
                (f'{name}/data', np.asarray(arr))
            ]

    def summarise(self, name):
        data = self.data[name]

        if isinstance(data, RectROI):
            data = data.image

        if data.ndim == 0:
            return data
        elif data.ndim == 2:
            # For the sake of space and memory we downsample images to a
            # resolution of 300x300.
            zoom_ratio = 300 / max(data.shape)
            if zoom_ratio < 1:
                data = ndimage.zoom(np.nan_to_num(data),
                                    zoom_ratio)

            return data
        else:
            summary_method = self.ctx.vars[name].summary
            if summary_method is None:
                return None
            return getattr(np, summary_method)(data)

    def save_hdf5(self, path):
        dsets = []
        for name, arr in self.data.items():
            reduced = self.summarise(name)
            if reduced is not None:
                dsets.append((f'.reduced/{name}', reduced))
            dsets.extend(self._datasets_for_arr(name, arr))

        log.info("Writing %d variables to %d datasets in %s",
                 len(self.data), len(dsets), path)

        with h5py.File(path, 'w') as f:
            # Create datasets before filling them, so metadata goes near the
            # start of the file.
            for path, arr in dsets:
                f.create_dataset(path, shape=arr.shape, dtype=arr.dtype)

            for path, arr in dsets:
                f[path][()] = arr


def run_and_save(proposal, run, out_path):
    run_dc = extra_data.open_run(proposal, run, data="all")

    ctx_file = ContextFile.from_py_file(Path('context.py'))
    res = Results.create(ctx_file, run_dc, run, out_path)
    res.save_hdf5(out_path)


def load_reduced_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        # SQlite doesn't like np.float32; .item() converts to Python numbers
        to_safe_type = lambda x: x.item() if np.isscalar(x) else x

        reduced_data = {
            name: to_safe_type(dset[()]) for name, dset in f['.reduced'].items()
        }

        for name, value in reduced_data.copy().items():
            roi_ds = f"{name}/RectROI_coords"
            if roi_ds in f:
                reduced_data[name] = RectROI(*f[roi_ds])

        return reduced_data

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
        if not isinstance(value, (type(None), int, float, str, bytes)):
            db_data[key] = pickle.dumps(value)

    with db:
        db.execute(f"""
            INSERT INTO runs (proposal, runnr, {cols_sql})
            VALUES (:proposal, :run, {values_sql})
            ON CONFLICT (proposal, runnr) DO UPDATE SET {updates_sql}
        """, db_data)


def extract_and_ingest(proposal, run):
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

    # run_dir = glob(f'/gpfs/exfel/exp/*/*/p{proposal:>06}/raw/r{run:>04}')[0]
    run_and_save(proposal, run, out_path)
    reduced_data = load_reduced_data(out_path)
    log.info("Reduced data has %d fields", len(reduced_data))
    add_to_db(reduced_data, db, proposal, run)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    proposal = int(sys.argv[1])
    run = int(sys.argv[2])
    out_path = sys.argv[3]
    run_and_save(proposal, run, out_path)
