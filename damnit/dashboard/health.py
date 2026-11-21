"""Sample current service health on the dashboard host, independently of historical filters."""
import json
import os
import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

from .journal import JournalError, _journal_records, _time_options

SERVICES = ("combiner", "listener")
ACTIVITY_PATTERNS = {
    "combiner": {
        "last_input": r"Combining .+ into .+p\d+_r\d+\.h5",
        "last_success": r"DB change \(set_variables(?: p\d+ r\d+)?\): .* ms commit$",
    },
    "listener": {
        "last_input": r"Added p\d+ r\d+ \((?:raw|proc) data\) to database$",
        "last_success": r"Launched Slurm \([^)]+\) job \S+ to run context file$",
    },
}
CACHE_SECONDS = 30
_snapshot = None
_expires = 0
_lock = Lock()


def _unit_health(service):
    properties = "LoadState,ActiveState,SubState,Result,NRestarts,ActiveEnterTimestamp,WorkingDirectory,ExecStart"
    unit = dict(status="unavailable", load_state=None, active_state=None, sub_state=None,
                result=None, restarts=None, active_since=None, error=None)
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "show", "--all",
             "--property=" + properties, f"damnit-{service}.service"],
            capture_output=True, text=True, timeout=4, env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
        )
        data = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
        if not data.get("LoadState") or not data.get("ActiveState") or (
            proc.returncode and data["LoadState"] != "not-found"
        ):
            raise ValueError("No unit state")
        active_since = None
        stamp = data.get("ActiveEnterTimestamp", "")
        if data["ActiveState"] == "active" and stamp and stamp != "n/a":
            # systemctl's default presentation is available on older service hosts.
            try:
                since = datetime.strptime(stamp, "%a %Y-%m-%d %H:%M:%S UTC")
                active_since = since.replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                pass  # An unfamiliar timestamp can still accompany a valid unit state.
        unit.update(status="ok", load_state=data["LoadState"], active_state=data["ActiveState"],
                    sub_state=data.get("SubState"), result=data.get("Result") or None,
                    restarts=int(data["NRestarts"]) if data.get("NRestarts") else None,
                    active_since=active_since)
        return unit, data
    except (OSError, ValueError, OverflowError, subprocess.TimeoutExpired):
        unit["error"] = "Could not read the systemd user unit."
        return unit, {}


def _last_activity(service, pattern, start, end):
    options = [f"--user-unit=damnit-{service}.service", "--output-fields=MESSAGE",
               "--reverse", "--lines=1", "--grep=" + pattern, "--case-sensitive=yes",
               *_time_options(start, end)]
    # Exhaust the generator so that journalctl's exit status is checked even when it emits a record.
    records = list(_journal_records(options, allow_no_matches=True, timeout_seconds=4))
    if records:
        micros = int(records[0]["__REALTIME_TIMESTAMP"])
        return (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=micros)).isoformat()
    return None


def _probe_python(exec_start):
    if configured := os.getenv("DAMNIT_DASHBOARD_KAFKA_PYTHON"):
        return configured
    # The two shipped units run a Python interpreter or its sibling console script.
    if match := re.search(r"\bpath=(/[^; ]+)\s*;", exec_start):
        path = Path(match[1])
        if path.name == "damnit":
            path = path.with_name("python")
        if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", path.name) and os.access(path, os.X_OK):
            return str(path)
    return sys.executable


def _kafka_health(service, properties):
    unavailable = dict(status="unavailable", lag=None, topics=[], error=None)
    if service == "listener":
        # Match backend.listener.EventProcessor's input stream for this service host.
        brokers = ["exflwgs06:9091"]
        if socket.gethostname().startswith("exflonc"):
            topics = ["test.euxfel.hed.daq", "test.euxfel.hed.cal"]
        else:
            topics = ["test.r2d2", "cal.offline-corrections"]
        working_dir = properties.get("WorkingDirectory")
        group = os.getenv("DAMNIT_DASHBOARD_LISTENER_GROUP")
        if not group and working_dir:
            try:
                group = "xfel-da-damnit-" + str(Path(working_dir).resolve()).replace("/", "_")
            except (OSError, RuntimeError):
                return {**unavailable, "error": "Could not resolve the Listener working directory."}
        if not group:
            return {**unavailable, "error": "Could not identify the Listener consumer group."}
    else:
        brokers = [os.environ.get("AMORE_BROKER", "exflwgs06.desy.de:9091")]
        topics = ["test.damnit.file_submissions"]
        group = "xfel-da-damnit-combiner"

    # The library's request timeout does not bound coordinator discovery or DNS.
    # Isolate each sample to enforce a deadline and release sockets on timeout.
    try:
        proc = subprocess.run(
            [_probe_python(properties.get("ExecStart", "")), str(Path(__file__).with_name("_kafka_lag.py"))],
            input=json.dumps(dict(brokers=brokers, topics=topics, group=group)),
            capture_output=True, text=True, check=True, timeout=8,
        )
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return {**unavailable, "error": "The Kafka offset probe timed out."}
    except (OSError, ValueError, subprocess.CalledProcessError):
        return {**unavailable, "error": "The Kafka offset probe could not run."}


def _collect():
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=24)
    services = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        units = {s: pool.submit(_unit_health, s) for s in SERVICES}
        observations = {
            s: {key: pool.submit(_last_activity, s, pattern, start, end)
                for key, pattern in ACTIVITY_PATTERNS[s].items()} for s in SERVICES
        }
        probes = {}
        for s in SERVICES:
            unit, properties = units[s].result()
            services[s] = {"unit": unit}
            probes[s] = pool.submit(_kafka_health, s, properties)
        for s in SERVICES:
            activity = dict(status="ok", last_input=None, last_success=None, since=start.isoformat(), error=None)
            for key, future in observations[s].items():
                try:
                    activity[key] = future.result()
                except (JournalError, KeyError, ValueError, OverflowError, TypeError):
                    activity.update(status="unavailable", error="Could not read recent service activity.")
            services[s].update(activity=activity, kafka=probes[s].result())
    return dict(sampled_at=end.isoformat(), services=services)


def get_health():
    """Coalesce concurrent requests and sample each live source at most once per refresh interval."""
    global _snapshot, _expires
    with _lock:
        now = time.monotonic()
        if _snapshot is None or now >= _expires:
            _snapshot = _collect()
            _expires = now + CACHE_SECONDS
        return _snapshot
