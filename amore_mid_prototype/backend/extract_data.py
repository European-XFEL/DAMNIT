import logging
import re
import sqlite3
import sys
from pathlib import Path

import extra_data
import h5py
import numpy as np
import xarray

from .context import Variable

log = logging.getLogger(__name__)


class Context:
    def __init__(self, vars):
        self.vars = vars

    @classmethod
    def from_py_file(cls, path: Path):
        code = path.read_bytes()
        d = {}
        exec(code, d)
        vars = {v.name: v for v in d.values() if isinstance(v, Variable)}
        log.debug("Loaded context from %s: %d variables", path, len(vars))
        return cls(vars)

    def process(self, xd_run):
        res = {}
        for name, var in self.vars.items():
            try:
                data = var.func(xd_run)
                if not isinstance(data, xarray.DataArray):
                    data = np.asarray(data)
            except Exception:
                log.error("Could not get data for %s", name, exc_info=True)
            else:
                res[name] = data
        return Results(res, self)


class Results:
    def __init__(self, data, ctx):
        self.data = data
        self.ctx = ctx

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
            return [
                (f'{name}/data', np.asarray(arr))
            ]

    def summarise(self, name):
        data = self.data[name]
        if data.ndim == 0:
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


def run_and_save(run_path, out_path):
    ctx = Context.from_py_file(Path('context.py'))
    run = extra_data.RunDirectory(run_path)
    res = ctx.process(run)
    res.save_hdf5(out_path)


def load_reduced_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        # SQlite doesn't like np.float32; .item() converts to Python numbers
        return {
            name: dset[()].item() for name, dset in f['.reduced'].items()
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

    with db:
        db.execute(f"""
            INSERT INTO runs (proposal, runnr, {cols_sql})
            VALUES (:proposal, :run, {values_sql})
            ON CONFLICT (proposal, runnr) DO UPDATE SET {updates_sql}
        """, reduced_data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_and_save(sys.argv[1], sys.argv[2])
