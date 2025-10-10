"""Machinery run inside a Slurm job to process data

This code runs in the DAMNIT application Python environment. It launches
a subprocess in the chosen 'context_python' environment to run the context
file (see ctxrunner.py).
"""
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
from getpass import getuser
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import h5py
import numpy as np

from kafka import KafkaProducer

from ..context import ContextFile, RunData
from ..definitions import UPDATE_BROKERS
from .db import DamnitDB, ReducedData, BlobTypes, MsgKind, msg_dict
from .extraction_control import ExtractionRequest, ExtractionSubmitter

log = logging.getLogger(__name__)


def prepare_env():
    # Ensure subprocess can import ctxrunner & damnit_ctx
    env = os.environ.copy()
    ctxsupport_dir = str(Path(__file__).parents[1] / 'ctxsupport')
    env['PYTHONPATH'] = ctxsupport_dir + (
        os.pathsep + env['PYTHONPATH'] if 'PYTHONPATH' in env else ''
    )
    return env


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
            subprocess.run([context_python, "-m", "ctxrunner", "ctx", str(ctx_path), str(out_file)],
                            cwd=db_dir, env=prepare_env(), check=True)

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
        } | {
            name: ReducedData(None, attributes={
                'error': get_dset_value(dset),
                'error_cls': dset.attrs.get("type", "")
            })
            for name, dset in f['.errors'].items()
        }

def add_to_db(reduced_data, db: DamnitDB, proposal, run):
    db.ensure_run(proposal, run)
    log.info("Adding p%d r%d to database, with %d columns",
             proposal, run, len(reduced_data))

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
        if not isinstance(reduced.value, (int, float, str, bytes, complex, type(None))):
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

class RunExtractor(Extractor):
    def __init__(self, proposal, run, cluster=False, run_data=RunData.ALL,
                 match=(), variables=(), mock=False, uuid=None, sandbox_args=None):
        super().__init__()
        self.proposal = proposal
        self.run = run
        self.cluster = cluster
        self.run_data = run_data
        self.match = match
        self.variables = variables
        self.mock = mock
        self.uuid = uuid or str(uuid4())
        self.sandbox_args = sandbox_args
        self.running_msg = msg_dict(MsgKind.processing_state_set, {
            'processing_id': self.uuid,
            'proposal': proposal,
            'run': run,
            'data': run_data.value,
            'status': 'RUNNING',
            'hostname': socket.gethostname(),
            'username': getuser(),
            'slurm_cluster': self._slurm_cluster(),
            'slurm_job_id': self._slurm_job_id(),
        })

    @staticmethod
    def _slurm_cluster():
        # For some reason, SLURM_CLUSTER_NAME is '(null)'. This is a workaround:
        if not os.environ.get('SLURM_JOB_ID', ''):
            return None
        partition = os.environ.get('SLURM_JOB_PARTITION', '')
        return 'solaris' if (partition == 'solcpu') else 'maxwell'

    @staticmethod
    def _slurm_job_id():
        # Get Slurm job ID in the same format that squeue uses
        array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID', '')
        array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '')
        if array_job_id and array_task_id:
            # In an array job, e.g. '12380337_308'
            return f"{array_job_id}_{array_task_id}"
        else:
            # Not an array job - just use the regular job ID
            return os.environ.get('SLURM_JOB_ID', '')

    @property
    def out_path(self):
        return Path('extracted_data', f'p{self.proposal}_r{self.run}.h5')

    def _notify_running(self):
        self.kafka_prd.send(self.db.kafka_topic, self.running_msg)

    def _notify_finished(self):
        self.kafka_prd.send(self.db.kafka_topic, msg_dict(
            MsgKind.processing_finished, {'processing_id': self.uuid}
        ))

    def extract_in_subprocess(self):
        python_exe = self.db.metameta.get('context_python', '') or sys.executable

        args = []
        if self.sandbox_args is not None:
            args.extend(shlex.split(sandbox_args))
            args.append(str(self.proposal))
            args.append("--")
        args.extend([python_exe, '-m', 'ctxrunner', 'exec', str(self.proposal), str(self.run),
                     self.run_data.value, '--save', self.out_path])
        if self.cluster:
            args.append('--cluster-job')
        if self.mock:
            args.append("--mock")
        if self.variables:
            for v in self.variables:
                args.extend(['--var', v])
        else:
            for m in self.match:
                args.extend(['--match', m])

        with TemporaryDirectory() as td:
            # Save a separate copy of the reduced data, so we can send an update
            # with only the variables that we've extracted.
            reduced_out_path = Path(td, 'reduced.h5')
            args.extend(['--save-reduced', str(reduced_out_path)])

            p = subprocess.Popen(args, env=prepare_env(), stdin=subprocess.DEVNULL)

            while True:
                try:
                    retcode = p.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    self._notify_running()

            if retcode:
                raise subprocess.CalledProcessError(retcode, p.args)

            return load_reduced_data(reduced_out_path)

    def extract_and_ingest(self):
        self._notify_running()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.out_path.parent.stat().st_uid == os.getuid():
            os.chmod(self.out_path.parent, 0o777)

        reduced_data = self.extract_in_subprocess()
        log.info("Reduced data has %d fields", len(reduced_data))
        add_to_db(reduced_data, self.db, self.proposal, self.run)

        # Send all the updates for scalars
        image_values = { name: reduced for name, reduced in reduced_data.items()
                         if isinstance(reduced.value, bytes) }
        update_msg = msg_dict(MsgKind.run_values_updated, {
            'run': self.run, 'proposal': self.proposal, 'values': {
                name: reduced.value for name, reduced in reduced_data.items()
                if name not in image_values
            }
        })
        self.kafka_prd.send(self.db.kafka_topic, update_msg).get(timeout=30)

        # And each image update separately so we don't hit any size limits
        for name, reduced in image_values.items():
            update_msg = msg_dict(MsgKind.run_values_updated, {
                'run': self.run, 'proposal': self.proposal, 'values': { name: reduced.value }})
            self.kafka_prd.send(self.db.kafka_topic, update_msg).get(timeout=30)

        log.info("Sent Kafka updates to topic %r", self.db.kafka_topic)

        # Launch a Slurm job if there are any 'cluster' variables to evaluate
        if not self.cluster:
            ctx_slurm = self.ctx_whole.filter(
                run_data=self.run_data, name_matches=self.match, variables=self.variables, cluster=True
            )
            ctx_no_slurm = ctx_slurm.filter(cluster=False)
            if set(ctx_slurm.vars) > set(ctx_no_slurm.vars):
                submitter = ExtractionSubmitter(Path.cwd(), self.db)
                cluster_req = ExtractionRequest(
                    self.run, self.proposal, mock=self.mock, run_data=self.run_data,
                    cluster=True, match=self.match, variables=self.variables
                )
                job_id, cluster = submitter.submit(cluster_req)

                # Announce the newly submitted follow up job
                self.kafka_prd.send(self.db.kafka_topic, msg_dict(
                    MsgKind.processing_state_set,
                    cluster_req.submitted_info(cluster, job_id)
                ))

        self._notify_finished()


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
    ap.add_argument('--var',  action="append", default=[])
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--update-vars', action='store_true')
    ap.add_argument('--processing-id', type=str)
    ap.add_argument('--sandbox-args', type=str)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Hide some logging from Kafka to make things more readable
    logging.getLogger('kafka').setLevel(logging.WARNING)

    username = getpass.getuser()
    hostname = socket.gethostname()

    if args.sandbox_args is not None:
        p = subprocess.run([*shlex.split(args.sandbox_args), str(args.proposal), "--", "whoami"],
                           capture_output=True, check=True, text=True)
        username = p.stdout.strip()

    print(f"\n----- Processing r{args.run} (p{args.proposal}) as {username} on {hostname} -----", file=sys.stderr)
    log.info(f"run_data={args.run_data}, match={args.match}")
    if args.mock:
        log.info("Using mock run object for testing")
    if args.cluster_job:
        log.info("Extracting cluster variables in Slurm job %s",
                 os.environ.get('SLURM_JOB_ID', '?'))

    extr = RunExtractor(args.proposal, args.run,
                        cluster=args.cluster_job,
                        run_data=RunData(args.run_data),
                        match=args.match,
                        variables=args.var,
                        mock=args.mock,
                        uuid=args.processing_id,
                        sandbox_args=args.sandbox_args)
    if args.update_vars:
        extr.update_db_vars()

    extr.extract_and_ingest()
    extr.kafka_prd.flush(timeout=10)


if __name__ == '__main__':
    main()
