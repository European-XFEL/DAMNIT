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
from dataclasses import dataclass, field
from pathlib import Path
from secrets import token_hex
from threading import Thread
from uuid import uuid4

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

def batches(l, n):
    start = 0
    while True:
        end = start + n
        batch = l[start:end]
        if not batch:
            return
        yield batch
        start = end


@dataclass
class ExtractionRequest:
    """Description of some data we want to extract"""
    run: int
    proposal: int
    run_data: RunData
    sandbox_args: str = ""
    damnit_python: str = sys.executable
    cluster: bool = False
    match: tuple = ()
    variables: tuple = ()   # Overrides match if present
    mock: bool = False
    update_vars: bool = True
    processing_id: str = field(default_factory=lambda : str(uuid4()))

    def python_cmd(self):
        """Creates the command for a process to do this extraction"""
        cmd = [
            self.damnit_python, '-m', 'damnit.backend.extract_data',
            str(self.proposal), str(self.run), self.run_data.value,
            '--processing-id', self.processing_id,
        ]
        if self.cluster:
            cmd.append('--cluster-job')
        if self.variables:
            for v in self.variables:
                cmd.extend(['--var', v])
        else:
            for m in self.match:
                cmd.extend(['--match', m])
        if self.mock:
            cmd.append('--mock')
        if self.update_vars:
            cmd.append('--update-vars')
        if len(self.sandbox_args) > 0:
            cmd.extend(["--sandbox-args", self.sandbox_args])

        return cmd

    def submitted_info(self, cluster, job_id):
        return {
            'processing_id': self.processing_id,
            'proposal': self.proposal,
            'run': self.run,
            'data': self.run_data.value,
            'status': 'PENDING',
            'hostname': '',
            'username': '',
            'slurm_cluster': cluster,
            'slurm_job_id': job_id,
        }


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

    def _filter_env(self):
        env = os.environ.copy()
        # A non-array job submitted from an array job will inherit these
        # variables if they're not cleared.
        for v in ['SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID']:
            env.pop(v, None)
        # Slurm jobs shouldn't attempt to use your X session
        env.pop('DISPLAY', None)
        # Some LD_PRELOAD entries from FastX (?) cause spurious warnings
        env.pop('LD_PRELOAD', None)
        return env

    def submit(self, req: ExtractionRequest):
        """Submit a Slurm job to extract data from a run

        Returns the job ID & cluster
        """
        res = subprocess.run(
            self.sbatch_cmd(req), stdout=subprocess.PIPE, text=True, check=True,
            cwd=self.context_dir, env=self._filter_env()
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

        if req.run == -1:
            job_name = f"p{req.proposal}-damnit"
        else:
            # We put the run number first so that it's visible in
            # squeue's default 11-character column for the JobName.
            job_name = f"r{req.run}-p{req.proposal}-damnit"

        return [
            'sbatch', '--parsable',
            *self._resource_opts(req.cluster),
            '-o', log_path,
            '--open-mode=append',
            '--job-name', job_name,
            '--wrap', shlex.join(req.python_cmd())
        ]

    def submit_multi(self, reqs: list[ExtractionRequest], limit_running=-1):
        """Submit multiple requests using Slurm job arrays.
        """
        out = []

        if limit_running == -1:
            limit_running = self.db.metameta.setdefault("concurrent_jobs", 15)

        assert len({r.cluster for r in reqs}) <= 1  # Don't mix cluster/non-cluster

        # Array jobs are limited to 1001 in Slurm config (MaxArraySize)
        for req_group in batches(reqs, 1000):
            grpid = token_hex(8)   # random unique string
            scripts_dir = self.context_dir / '.tmp'
            scripts_dir.mkdir(exist_ok=True)
            if scripts_dir.stat().st_uid == os.getuid():
                scripts_dir.chmod(0o777)

            for i, req in enumerate(req_group):
                script_file = scripts_dir / f'launch-{grpid}-{i}.sh'
                log_path = process_log_path(req.run, req.proposal, self.context_dir)
                script_file.write_text(
                    'rm "$0"\n'  # Script cleans itself up
                    f'{shlex.join(req.python_cmd())} >>"{log_path}" 2>&1'
                )
                script_file.chmod(0o777)

            script_expr = f".tmp/launch-{grpid}-$SLURM_ARRAY_TASK_ID.sh"
            cmd = self.sbatch_array_cmd(script_expr, req_group, limit_running)
            if out:
                # 1 batch at a time, to simplify limiting concurrent jobs
                prev_job = out[-1][0]
                cmd.append(f"--dependency=afterany:{prev_job}")
            res = subprocess.run(
                cmd, stdout=subprocess.PIPE, text=True, check=True,
                cwd=self.context_dir, env=self._filter_env(),
            )
            job_id, _, cluster = res.stdout.partition(';')
            job_id = job_id.strip()
            cluster = cluster.strip() or 'maxwell'
            log.info("Launched Slurm (%s) job array %s (%d runs) to run context file",
                     cluster, job_id, len(req_group))
            out.extend([(f"{job_id}_{i}", cluster) for i in range(len(req_group))])

        return out

    def sbatch_array_cmd(self, script_expr, reqs, limit_running=30):
        """Make the sbatch command for an array job"""
        req = reqs[0]  # This should never be called with an empty list
        return [
            'sbatch', '--parsable',
            *self._resource_opts(req.cluster),
            # Slurm doesn't know the run number, so we redirect inside the job
            '-o', '/dev/null',
            '--open-mode=append',
            '--job-name', f"p{req.proposal}-damnit",
            '--array', f"0-{len(reqs)-1}%{limit_running}",
            '--wrap', f'exec {script_expr}'
        ]

    def execute_in_slurm(self, req: ExtractionRequest):
        """Run an extraction job in srun with live output"""
        log_path = process_log_path(req.run, req.proposal, self.context_dir)
        log.info("Processing output will be written to %s",
                 log_path.relative_to(self.context_dir.absolute()))

        # Duplicate output to the log file
        with tee(log_path) as pipe:
            subprocess.run(
                self.srun_cmd(req), stdout=pipe, stderr=subprocess.STDOUT, check=True,
                cwd=self.context_dir, env=self._filter_env(),
            )

    def execute_direct(self, req: ExtractionRequest):
        log_path = process_log_path(req.run, req.proposal, self.context_dir)
        log.info("Processing output will be written to %s",
                 log_path.relative_to(self.context_dir.absolute()))

        # Duplicate output to the log file
        with tee(log_path) as pipe:
            subprocess.run(
                req.python_cmd(), stdout=pipe, stderr=subprocess.STDOUT, check=True,
                cwd=self.context_dir,
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


def reprocess(runs, proposal=None, match=(), mock=False, watch=False, direct=False, limit_running=-1):
    """Called by the 'damnit reprocess' subcommand"""
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

        props_runs.sort()
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

    if direct:
        for req in reqs:
            submitter.execute_direct(req)
    elif watch:
        for req in reqs:
            submitter.execute_in_slurm(req)
    else:
        submitter.submit_multi(reqs, limit_running=limit_running)


class ExtractionJobTracker:
    """Track running extraction jobs using their running/finished messages"""
    def __init__(self):
        self.jobs = {}  # keyed by processing_id

    def on_processing_state_set(self, info):
        proc_id = info['processing_id']
        if info != self.jobs.get(proc_id, None):
            self.jobs[proc_id] = info
            self.on_run_jobs_changed(info['proposal'], info['run'])
            log.debug("Processing %s for p%s r%s on %s (%s)",
                      info['status'], info['proposal'], info['run'], info['hostname'], proc_id)

    def on_processing_finished(self, info):
        proc_id = info['processing_id']
        info = self.jobs.pop(proc_id, None)
        if info is not None:
            self.on_run_jobs_changed(info['proposal'], info['run'])
            log.debug("Processing finished for p%s r%s (%s)",
                      info['proposal'], info['run'], proc_id)

    def on_run_jobs_changed(self, proposal, run):
        pass   # Implement in subclass

    def check_slurm_jobs(self):
        """Check for any Slurm jobs that exited without a 'finished' message"""
        jobs_by_cluster = {}
        for info in self.jobs.values():
            if cluster := info['slurm_cluster']:
                jobs_by_cluster.setdefault(cluster, []).append(info)

        for cluster, infos in jobs_by_cluster.items():
            jids = [i['slurm_job_id'] for i in infos]
            # Passing 1 Job ID can give an 'Invalid job id' error if it has
            # already left the queue. With multiple, we always get a list back.
            if len(jids) == 1:
                jids.append("1")

            cmd = ["squeue", "--clusters", cluster, "--jobs=" + ",".join(jids),
                   "--format=%i %T", "--noheader"]
            self.squeue_check_jobs(cmd, infos)

    # Running the squeue subprocess is separated here so GUI code can override
    # it, to avoid blocking the event loop if squeue is slow for any reason.
    def squeue_check_jobs(self, cmd, jobs_to_check):
        res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        if res.returncode != 0:
            log.warning("Error calling squeue")
            return

        self.process_squeue_output(res.stdout, jobs_to_check)

    def process_squeue_output(self, stdout: str, jobs_to_check):
        """Inspect squeue output to clean up crashed jobs"""
        still_running = set()
        for line in stdout.splitlines():
            job_id, status = line.strip().split()
            if status in ('RUNNING', 'PENDING'):
                still_running.add(job_id)

        for info in jobs_to_check:
            proc_id = info['processing_id']
            job_id = info['slurm_job_id']
            if (proc_id in self.jobs) and (job_id not in still_running):
                del self.jobs[proc_id]
                self.on_run_jobs_changed(info['proposal'], info['run'])
                log.info("Slurm job %s on %s (%s) crashed or was cancelled",
                         info['slurm_job_id'], info['slurm_cluster'], proc_id)
