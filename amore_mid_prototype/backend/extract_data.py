import argparse
import getpass
import os
import logging
import pickle
import re
import shlex
import sqlite3
import subprocess
import sys
from ctypes import CDLL
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

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


def extract_in_subprocess(
        proposal, run, out_path, cluster=False, run_data=RunData.ALL, match=(),
        python_exe=None,
):
    if not python_exe:
        python_exe = sys.executable
    env = os.environ.copy()
    ctxsupport_dir = str(Path(__file__).parents[1] / 'ctxsupport')
    env['PYTHONPATH'] = ctxsupport_dir + (
        os.pathsep + env['PYTHONPATH'] if 'PYTHONPATH' in env else ''
    )
    args = [python_exe, '-m', 'ctxrunner', str(proposal), str(run), run_data.value,
            '--save', out_path]
    if cluster:
        args.append('--cluster-job')
    for m in match:
        args.extend(['--match', m])

    with TemporaryDirectory() as td:
        # Save a separate copy of the reduced data, so we can send an update
        # with only the variables that we've extracted.
        reduced_out_path = Path(td, 'reduced.h5')
        args.extend(['--save-reduced', str(reduced_out_path)])

        subprocess.run(args, env=env, check=True)

        return load_reduced_data(reduced_out_path)


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

        python_exe = get_meta(self.db, 'context_python', '')
        reduced_data = extract_in_subprocess(
            proposal, run, out_path, cluster=cluster, run_data=run_data,
            match=match, python_exe=python_exe
        )
        log.info("Reduced data has %d fields", len(reduced_data))
        add_to_db(reduced_data, self.db, proposal, run)

        # Send update via Kafka
        reduced_data['Proposal'] = proposal
        reduced_data['Run'] = run
        self.kafka_prd.send(self.update_topic, reduced_data).get(timeout=30)
        log.info("Sent Kafka update to topic %r", self.update_topic)

        # Launch a Slurm job if there are any 'cluster' variables to evaluate
        ctx =       self.ctx_whole.filter(run_data=run_data, name_matches=match, cluster=cluster)
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
