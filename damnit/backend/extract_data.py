"""Machinery run inside a Slurm job to process data

This code runs in the DAMNIT application Python environment. It launches
a subprocess in the chosen 'context_python' environment to run the context
file (see ctxrunner.py).
"""
import argparse
import copy
import os
import logging
import pickle
import re
import socket
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from kafka import KafkaProducer

from ..context import ContextFile, RunData
from ..definitions import UPDATE_BROKERS
from .db import DamnitDB, ReducedData, BlobTypes, MsgKind, msg_dict
from .extraction_control import ExtractionRequest, ExtractionSubmitter

log = logging.getLogger(__name__)


def run_in_subprocess(args, **kwargs):
    env = os.environ.copy()
    ctxsupport_dir = str(Path(__file__).parents[1] / 'ctxsupport')
    env['PYTHONPATH'] = ctxsupport_dir + (
        os.pathsep + env['PYTHONPATH'] if 'PYTHONPATH' in env else ''
    )

    return subprocess.run(args, env=env, **kwargs)


def extract_in_subprocess(
        proposal, run, out_path, cluster=False, run_data=RunData.ALL, match=(),
        python_exe=None, mock=False
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

        run_in_subprocess(args, check=True)

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

    def get_attrs(ds):
        d = {}
        for name, value in ds.attrs.items():
            if name in {"max_diff", "summary_method"}:
                continue  # These are stored separately

            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):  # Scalar
                value = value.item()
            d[name] = value
        return d

    with h5py.File(h5_path, 'r') as f:
        return {
            name: ReducedData(
                get_dset_value(dset),
                max_diff=dset.attrs.get("max_diff", np.array(None)).item(),
                summary_method=dset.attrs.get("summary_method", ""),
                attributes=get_attrs(dset),
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

    def extract_and_ingest(self, proposal, run, cluster=False,
                           run_data=RunData.ALL, match=(), mock=False):
        if proposal is None:
            proposal = self.db.metameta['proposal']

        out_path = Path('extracted_data', f'p{proposal}_r{run}.h5')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.parent.stat().st_uid == os.getuid():
            os.chmod(out_path.parent, 0o777)

        python_exe = self.db.metameta.get('context_python', '')
        reduced_data = extract_in_subprocess(
            proposal, run, out_path, cluster=cluster, run_data=run_data,
            match=match, python_exe=python_exe, mock=mock,
        )
        log.info("Reduced data has %d fields", len(reduced_data))
        add_to_db(reduced_data, self.db, proposal, run)

        # Send all the updates for scalars
        image_values = { name: reduced for name, reduced in reduced_data.items()
                         if isinstance(reduced.value, bytes) }
        update_msg = msg_dict(MsgKind.run_values_updated, {
            'run': run, 'proposal': proposal, 'values': { name: reduced.value for name, reduced in reduced_data.items()
                                                          if name not in image_values }})
        self.kafka_prd.send(self.db.kafka_topic, update_msg).get(timeout=30)

        # And each image update separately so we don't hit any size limits
        for name, reduced in image_values.items():
            update_msg = msg_dict(MsgKind.run_values_updated, {
                'run': run, 'proposal': proposal, 'values': { name: reduced.value }})
            self.kafka_prd.send(self.db.kafka_topic, update_msg).get(timeout=30)

        log.info("Sent Kafka updates to topic %r", self.db.kafka_topic)

        # Launch a Slurm job if there are any 'cluster' variables to evaluate
        ctx =       self.ctx_whole.filter(run_data=run_data, name_matches=match, cluster=cluster)
        ctx_slurm = self.ctx_whole.filter(run_data=run_data, name_matches=match, cluster=True)
        if set(ctx_slurm.vars) > set(ctx.vars):
            submitter = ExtractionSubmitter(Path.cwd(), self.db)
            cluster_req = ExtractionRequest(
                run, proposal, run_data, cluster=True, match=match, mock=mock
            )
            submitter.submit(cluster_req)


def main(argv=None):
    # This runs inside the Slurm job
    ap = argparse.ArgumentParser()
    ap.add_argument('proposal', type=int)
    ap.add_argument('run', type=int)
    ap.add_argument('run_data', choices=('raw', 'proc', 'all'))
    # cluster-job means we've got a full Maxwell node to run cluster=True
    # variables (confusing because all extraction now runs in cluster jobs)
    ap.add_argument('--cluster-job', action="store_true")
    ap.add_argument('--match', action="append", default=[])
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--update-vars', action='store_true')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Hide some logging from Kafka to make things more readable
    logging.getLogger('kafka').setLevel(logging.WARNING)

    print(f"\n----- Processing r{args.run} (p{args.proposal}) -----", file=sys.stderr)
    log.info(f"run_data={args.run_data}, match={args.match}")
    if args.mock:
        log.info("Using mock run object for testing")
    if args.cluster_job:
        log.info("Extracting cluster variables in Slurm job %s on %s",
                 os.environ.get('SLURM_JOB_ID', '?'), socket.gethostname())

    extr = Extractor()
    if args.update_vars:
        extr.update_db_vars()

    extr.extract_and_ingest(args.proposal, args.run,
                            cluster=args.cluster_job,
                            run_data=RunData(args.run_data),
                            match=args.match,
                            mock=args.mock)


if __name__ == '__main__':
    main()
