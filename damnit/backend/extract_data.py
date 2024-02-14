import argparse
import copy
import getpass
import os
import logging
import pickle
import re
import shlex
import socket
import subprocess
import sys
from contextlib import contextmanager
from ctypes import CDLL
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from typing import Optional

import h5py
import numpy as np
from extra_data.read_machinery import find_proposal

from kafka import KafkaProducer

from ..context import ContextFile, RunData
from ..definitions import UPDATE_BROKERS
from .db import DamnitDB, ReducedData, BlobTypes, MsgKind, msg_dict


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

def run_in_subprocess(args, **kwargs):
    env = os.environ.copy()
    ctxsupport_dir = str(Path(__file__).parents[1] / 'ctxsupport')
    env['PYTHONPATH'] = ctxsupport_dir + (
        os.pathsep + env['PYTHONPATH'] if 'PYTHONPATH' in env else ''
    )

    return subprocess.run(args, env=env, **kwargs)

def process_log_path(run, proposal, ctx_dir=Path('.'), create=True):
    p = ctx_dir.absolute() / 'process_logs' / f"r{run}-p{proposal}.out"
    if create:
        p.parent.mkdir(exist_ok=True)
        if p.parent.stat().st_uid == os.getuid():
            p.parent.chmod(0o777)
        p.touch(exist_ok=True)
        if p.stat().st_uid == os.getuid():
            p.chmod(0o666)
    return p


@contextmanager
def tee(path: Optional[Path]):
    if path is None:
        yield None
        return

    with path.open('ab') as fout:
        r, w = os.pipe()
        def loop():
            while b := os.read(r, 4096):
                fout.write(b)
                sys.stdout.buffer.write(b)

        thread = Thread(target=loop)
        thread.start()
        try:
            yield w
        finally:
            os.close(w)
            thread.join()
            os.close(r)


def extract_in_subprocess(
        proposal, run, out_path, cluster=False, run_data=RunData.ALL, match=(),
        python_exe=None, mock=False, tee_output=None
):
    if not python_exe:
        python_exe = sys.executable

    args = [python_exe, '-m', 'ctxrunner', 'exec', str(proposal), str(run), run_data.value,
            '--save', out_path]
    if cluster:
        args.append('--cluster-job')
    if mock:
        args.append("--mock")
    for m in match:
        args.extend(['--match', m])

    with TemporaryDirectory() as td:
        # Save a separate copy of the reduced data, so we can send an update
        # with only the variables that we've extracted.
        reduced_out_path = Path(td, 'reduced.h5')
        args.extend(['--save-reduced', str(reduced_out_path)])

        with tee(tee_output) as pipe:
            run_in_subprocess(args, check=True, stdout=pipe, stderr=subprocess.STDOUT)

        return load_reduced_data(reduced_out_path)

class ContextFileUnpickler(pickle.Unpickler):
    """
    Unpickler class to allow unpickling ContextFile's from any module location.

    See: https://stackoverflow.com/a/51397373
    """
    def find_class(self, module, name):
        if name == 'ContextFile':
            return ContextFile
        else:
            return super().find_class(module, name)

def get_context_file(ctx_path: Path, context_python=None):
    ctx_path = ctx_path.absolute()
    db_dir = ctx_path.parent

    if context_python is None:
        db = DamnitDB.from_dir(ctx_path.parent)
        with db.conn:
            ctx = ContextFile.from_py_file(ctx_path)

        db.close()
        return ctx, None
    else:
        with TemporaryDirectory() as d:
            out_file = Path(d) / "context.pickle"
            run_in_subprocess([context_python, "-m", "ctxrunner", "ctx", str(ctx_path), str(out_file)],
                              cwd=db_dir, check=True)

            with out_file.open("rb") as f:
                unpickler = ContextFileUnpickler(f)
                ctx, error_info = unpickler.load()

                return ctx, error_info

def load_reduced_data(h5_path):
    def get_dset_value(ds):
        # If it's a string, extract the string
        if h5py.check_string_dtype(ds.dtype) is not None:
            return ds.asstr()[()]
        elif (ds.ndim == 1 and ds.dtype == np.uint8
              and BlobTypes.identify(ds[:8].tobytes()) is BlobTypes.png):
            # PNG: pass around as bytes
            return ds[()].tobytes()
        else:
            value = ds[()]
            # SQlite doesn't like np.float32; .item() converts to Python numbers
            return value.item() if (value.ndim == 0) else value

    with h5py.File(h5_path, 'r') as f:
        return {
            name: ReducedData(
                get_dset_value(dset),
                max_diff=dset.attrs.get("max_diff", np.array(None)).item(),
                summary_method=dset.attrs.get("summary_method", "")
            )
            for name, dset in f['.reduced'].items()
        }

def add_to_db(reduced_data, db: DamnitDB, proposal, run):
    db.ensure_run(proposal, run)
    log.info("Adding p%d r%d to database, with %d columns",
             proposal, run, len(reduced_data))

    # Check that none of the values are None. We don't support None for the
    # rather bland reason of there not being a type for it in the DataType enum,
    # and right now it doesn't really make sense to add one.
    for name, reduced in reduced_data.items():
        if reduced.value is None:
            raise RuntimeError(f"Variable '{name}' has value None, this is unsupported")

    # We're going to be formatting column names as strings into SQL code,
    # so check that they are simple identifiers before we get there.
    for name in reduced_data:
        assert re.match(r'[a-zA-Z][a-zA-Z0-9_]*$', name), f"Bad field name {name}"

    # Make a deepcopy before making modifications to the dictionary, such as
    # removing `start_time` and pickling non-{array, scalar} values.
    reduced_data = copy.deepcopy(reduced_data)

    # Handle the start_time variable specially
    start_time = reduced_data.pop("start_time", None)
    if start_time is not None:
        db.ensure_run(proposal, run, start_time=start_time.value)

    for name, reduced in reduced_data.items():
        if not isinstance(reduced.value, (int, float, str, bytes)):
            raise TypeError(f"Unsupported type for database: {type(reduced.value)}")

        db.set_variable(proposal, run, name, reduced)


class Extractor:
    _proposal = None

    def __init__(self):
        self.db = DamnitDB()
        self.kafka_prd = KafkaProducer(
            bootstrap_servers=UPDATE_BROKERS,
            value_serializer=lambda d: pickle.dumps(d),
        )
        context_python = self.db.metameta.get("context_python")
        self.ctx_whole, error_info = get_context_file(Path('context.py'), context_python=context_python)
        assert error_info is None, error_info

    def update_db_vars(self):
        updates = self.db.update_computed_variables(self.ctx_whole.vars_to_dict())

        for name, var in updates.items():
            self.kafka_prd.send(self.db.kafka_topic, msg_dict(
                MsgKind.variable_set, {'name': name} | var
            ))
        self.kafka_prd.flush()

    @property
    def proposal(self):
        if self._proposal is None:
            self._proposal = self.db.metameta['proposal']
        return self._proposal

    def slurm_options(self):
        opts = ["--time", self.db.metameta.get("slurm_time", "02:00:00")]

        if reservation := self.db.metameta.get('slurm_reservation', ''):
            opts.extend(['--reservation', reservation])
        else:
            partition = self.db.metameta.get('slurm_partition', '') or default_slurm_partition()
            opts.extend(['--partition', partition])

        return opts

    def extract_and_ingest(self, proposal, run, cluster=False,
                           run_data=RunData.ALL, match=(), mock=False, tee_output=None):
        if proposal is None:
            proposal = self.proposal

        self.update_db_vars()

        out_path = Path('extracted_data', f'p{proposal}_r{run}.h5')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.parent.stat().st_uid == os.getuid():
            os.chmod(out_path.parent, 0o777)

        python_exe = self.db.metameta.get('context_python', '')
        reduced_data = extract_in_subprocess(
            proposal, run, out_path, cluster=cluster, run_data=run_data,
            match=match, python_exe=python_exe, mock=mock, tee_output=tee_output,
        )
        log.info("Reduced data has %d fields", len(reduced_data))
        add_to_db(reduced_data, self.db, proposal, run)

        # Send update via Kafka
        update_msg = msg_dict(MsgKind.run_values_updated, {
            'run': run, 'proposal': proposal, 'values': {
                name: reduced.value for name, reduced in reduced_data.items()
        }})
        self.kafka_prd.send(self.db.kafka_topic, update_msg).get(timeout=30)
        log.info("Sent Kafka update to topic %r", self.db.kafka_topic)

        # Launch a Slurm job if there are any 'cluster' variables to evaluate
        ctx =       self.ctx_whole.filter(run_data=run_data, name_matches=match, cluster=cluster)
        ctx_slurm = self.ctx_whole.filter(run_data=run_data, name_matches=match, cluster=True)
        if set(ctx_slurm.vars) > set(ctx.vars):
            python_cmd = [sys.executable, '-m', 'damnit.backend.extract_data',
                          '--cluster-job', str(proposal), str(run), run_data.value]
            for m in match:
                python_cmd.extend(["--match", m])

            res = subprocess.run([
                'sbatch', '--parsable',
                *self.slurm_options(),
                '-o', process_log_path(run, proposal),
                '--open-mode=append',
                # Note: we put the run number first so that it's visible in
                # squeue's default 11-character column for the JobName.
                '--job-name', f"r{run}-p{proposal}-damnit",
                '--wrap', shlex.join(python_cmd)
            ], stdout=subprocess.PIPE, text=True)
            job_id = res.stdout.partition(';')[0].strip()
            log.info("Launched Slurm job %s to calculate cluster variables", job_id)


def proposal_runs(proposal):
    proposal_name = f"p{int(proposal):06d}"
    raw_dir = Path(find_proposal(proposal_name)) / "raw"
    return set(int(p.stem[1:]) for p in raw_dir.glob("*"))


def reprocess(runs, proposal=None, match=(), mock=False):
    """Called by the 'amore-proto reprocess' subcommand"""
    extr = Extractor()
    if proposal is None:
        proposal = extr.proposal

    if runs == ['all']:
        rows = extr.db.conn.execute("SELECT proposal, run FROM runs").fetchall()

        # Dictionary of proposal numbers to sets of available runs
        available_runs = {}
        # Lists of (proposal, run) tuples
        props_runs = []
        unavailable_runs = []

        for proposal, run in rows:
            if not mock and proposal not in available_runs:
                available_runs[proposal] = proposal_runs(proposal)

            if mock or run in available_runs[proposal]:
                props_runs.append((proposal, run))
            else:
                unavailable_runs.append((proposal, run))

        print(f"Reprocessing {len(props_runs)} runs already recorded, skipping {len(unavailable_runs)}...")
    else:
        try:
            runs = set([int(r) for r in runs])
        except ValueError as e:
            sys.exit(f"Run numbers must be integers ({e})")

        if mock:
            available_runs = runs
        else:
            available_runs = proposal_runs(proposal)

        unavailable_runs = runs - available_runs
        if len(unavailable_runs) > 0:
            # Note that we print unavailable_runs as a list so it's enclosed
            # in [] brackets, which is more recognizable than the {} braces
            # that sets are enclosed in.
            print(
                f"Warning: skipping {len(unavailable_runs)} runs because they don't exist: {sorted(unavailable_runs)}")

        props_runs = [(proposal, r) for r in sorted(runs & available_runs)]

    log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for prop, run in props_runs:
        log_path = process_log_path(run, prop)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(log_formatter)
        logging.getLogger().addHandler(file_handler)
        try:
            log.info("\n\n----- Reprocessing r%s (p%s) -----", run, prop)
            log.info(f"match={match}")
            extr.extract_and_ingest(prop, run, match=match, mock=mock, tee_output=log_path)
        except Exception:
            log.error("Exception while extracting p%s r%s", run, prop, exc_info=True)
            raise
        finally:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()


if __name__ == '__main__':
    # This runs when extraction is launched by the listener:
    ap = argparse.ArgumentParser()
    ap.add_argument('proposal', type=int)
    ap.add_argument('run', type=int)
    ap.add_argument('run_data', choices=('raw', 'proc', 'all'))
    ap.add_argument('--cluster-job', action="store_true")
    ap.add_argument('--match', action="append", default=[])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Hide some logging from Kafka to make things more readable
    logging.getLogger('kafka').setLevel(logging.WARNING)

    print(f"\n----- Processing r{args.run} (p{args.proposal}) -----", file=sys.stderr)
    log.info(f"run_data={args.run_data}, match={args.match}")
    if args.cluster_job:
        log.info("Extracting cluster variables in Slurm job %s on %s",
                 os.environ.get('SLURM_JOB_ID', '?'), socket.gethostname())

    Extractor().extract_and_ingest(args.proposal, args.run,
                                   cluster=args.cluster_job,
                                   run_data=RunData(args.run_data),
                                   match=args.match)
