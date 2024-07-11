"""Machinery to run Slurm jobs for extracting data

See extract_data.py for what happens inside the jobs this launches.
"""
import getpass
import logging
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from ctypes import CDLL
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

from extra_data.read_machinery import find_proposal

from .db import DamnitDB
from ..context import RunData

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
def tee(path: Path):
    with path.open('ab') as fout:
        r, w = os.pipe()
        def loop():
            while b := os.read(r, 4096):
                fout.write(b)
                sys.stdout.buffer.write(b)
                sys.stdout.flush()

        thread = Thread(target=loop)
        thread.start()
        try:
            yield w
        finally:
            os.close(w)
            thread.join()
            os.close(r)


def proposal_runs(proposal):
    proposal_name = f"p{int(proposal):06d}"
    raw_dir = Path(find_proposal(proposal_name)) / "raw"
    return set(int(p.stem[1:]) for p in raw_dir.glob("*"))


@dataclass
class ExtractionRequest:
    """Description of some data we want to extract"""
    run: int
    proposal: int
    run_data: RunData
    cluster: bool = False
    match: tuple = ()
    mock: bool = False
    update_vars: bool = True

    def python_cmd(self):
        """Creates the command for a process to do this extraction"""
        cmd = [
            sys.executable, '-m', 'damnit.backend.extract_data',
            str(self.proposal), str(self.run), self.run_data.value
        ]
        if self.cluster:
            cmd.append('--cluster-job')
        for m in self.match:
            cmd.extend(['--match', m])
        if self.mock:
            cmd.append('--mock')
        if self.update_vars:
            cmd.append('--update-vars')
        return cmd


class ExtractionSubmitter:
    """Submits extraction jobs to Slurm"""
    def __init__(self, context_dir: Path, db=None):
        self.context_dir = context_dir
        self.db = db or DamnitDB.from_dir(context_dir)

    _proposal = None

    @property
    def proposal(self):
        if self._proposal is None:
            self._proposal = self.db.metameta['proposal']
        return self._proposal

    def submit(self, req: ExtractionRequest):
        """Submit a Slurm job to extract data from a run

        Returns the job ID & cluster
        """
        res = subprocess.run(
            self.sbatch_cmd(req), stdout=subprocess.PIPE, text=True, check=True
        )
        job_id, _, cluster = res.stdout.partition(';')
        job_id = job_id.strip()
        cluster = cluster.strip() or 'maxwell'
        log.info("Launched Slurm (%s) job %s to run context file", cluster, job_id)
        return job_id, cluster

    def sbatch_cmd(self, req: ExtractionRequest):
        """Make the sbatch command to extract data from a run"""
        log_path = process_log_path(req.run, req.proposal, self.context_dir)
        log.info("Processing output will be written to %s",
                 log_path.relative_to(self.context_dir.absolute()))

        return [
            'sbatch', '--parsable',
            *self._resource_opts(req.cluster),
            '-o', log_path,
            '--open-mode=append',
            # Note: we put the run number first so that it's visible in
            # squeue's default 11-character column for the JobName.
            '--job-name', f"r{req.run}-p{req.proposal}-damnit",
            '--wrap', shlex.join(req.python_cmd())
        ]

    def execute_in_slurm(self, req: ExtractionRequest):
        """Run an extraction job in srun with live output"""
        log_path = process_log_path(req.run, req.proposal, self.context_dir)
        log.info("Processing output will be written to %s",
                 log_path.relative_to(self.context_dir.absolute()))

        # Duplicate output to the log file
        with tee(log_path) as pipe:
            subprocess.run(
                self.srun_cmd(req), stdout=pipe, stderr=subprocess.STDOUT, check=True
            )

    def execute_direct(self, req: ExtractionRequest):
        log_path = process_log_path(req.run, req.proposal, self.context_dir)
        log.info("Processing output will be written to %s",
                 log_path.relative_to(self.context_dir.absolute()))

        # Duplicate output to the log file
        with tee(log_path) as pipe:
            subprocess.run(
                req.python_cmd(), stdout=pipe, stderr=subprocess.STDOUT, check=True
            )

    def srun_cmd(self, req: ExtractionRequest):
        return [
            'srun', *self._resource_opts(req.cluster),
            '--job-name', f"r{req.run}-p{req.proposal}-damnit",
            *req.python_cmd()
        ]

    def _resource_opts(self, cluster: bool):
        if cluster:
            return self._slurm_cluster_opts()
        else:
            return self._slurm_shared_opts()

    def _slurm_shared_opts(self):
        return [
            "--clusters", "solaris",
            # Default 4 CPU cores & 25 GB memory, can be overridden
            '--cpus-per-task', str(self.db.metameta.get('noncluster_cpus', '4')),
            '--mem', self.db.metameta.get('noncluster_mem', '25G'),
        ]

    def _slurm_cluster_opts(self):
        # Maxwell (dedicated node)
        opts = [
            "--clusters", "maxwell",
            "--time", self.db.metameta.get("slurm_time", "02:00:00")
        ]

        if reservation := self.db.metameta.get('slurm_reservation', ''):
            opts.extend(['--reservation', reservation])
        else:
            partition = self.db.metameta.get('slurm_partition', '') or default_slurm_partition()
            opts.extend(['--partition', partition])

        return opts


def reprocess(runs, proposal=None, match=(), mock=False, watch=False, direct=False):
    """Called by the 'amore-proto reprocess' subcommand"""
    submitter = ExtractionSubmitter(Path.cwd())
    if proposal is None:
        proposal = submitter.proposal

    if runs == ['all']:
        rows = submitter.db.conn.execute("SELECT proposal, run FROM runs").fetchall()

        # Dictionary of proposal numbers to sets of available runs
        available_runs = {}
        # Lists of (proposal, run) tuples
        props_runs = []
        unavailable_runs = []

        if mock:
            props_runs = rows
        else:
            for proposal, run in rows:
                if proposal not in available_runs:
                    available_runs[proposal] = proposal_runs(proposal)

                if run in available_runs[proposal]:
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

    reqs = [
        ExtractionRequest(run, prop, RunData.ALL, match=match, mock=mock)
        for prop, run in props_runs
    ]
    # To reduce DB write contention, only update the computed variables in the
    # first job when we're submitting a whole bunch.
    for req in reqs[1:]:
        req.update_vars = False

    for prop, run in props_runs:
        req = ExtractionRequest(run, prop, RunData.ALL, match=match, mock=mock)
        if direct:
            submitter.execute_direct(req)
        elif watch:
            submitter.execute_in_slurm(req)
        else:
            submitter.submit(req)
