"""Extract compact observations from the two DAMNIT user-unit journals."""
import hashlib
import re


# The combiner uses basicConfig's default format, the listener the CLI formatter.
LOG_PREFIX = re.compile(
    r"^(?:(?:DEBUG|INFO|WARNING|ERROR|CRITICAL):[\w.]+:"
    r"|\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+\w+\s+[\w.]+\s+)"
)
NUMBER = r"\d+(?:\.\d+)?"
COMBINING = re.compile(
    rf"""Combining (?P<source>.+) into .*?p(?P<p>\d+)_r(?P<r>\d+)\.h5['"]?\)?"""
    rf"(?:; event pending for (?P<pending>-?{NUMBER}) s before processing)?"
)
FILE_TIME = re.compile(rf"Finished combining in ({NUMBER}) s")
CREATED = re.compile(r"""Created .*?p(?P<p>\d+)_r(?P<r>\d+)\.h5['"]?\)? by renaming""")
MISSING = re.compile(r"File .*?p(?P<p>\d+)_r(?P<r>\d+)\.[^/]* not present \d+ s after notification, skipping")
DB_UPDATE = re.compile(
    r"Updating database in .* with (?P<count>\d+) variables for p(?P<p>\d+) r(?P<r>\d+)(?: from(?: .*)?)?"
)
DB_TIME = re.compile(
    rf"DB change \(set_variables(?: p(?P<p>\d+) r(?P<r>\d+))?\): "
    rf"(?P<transaction>{NUMBER}) ms in transaction, (?P<finish>{NUMBER}) ms (?P<result>commit|rollback)"
)
ADDED = re.compile(r"Added p(?P<p>\d+) r(?P<r>\d+) \((?P<data>raw|proc) data\) to database")
LAUNCHED = re.compile(r"Launched Slurm \([^)]+\) job \S+ to run context file")
PROCESS_FAILED = re.compile(r"Processing p(?P<p>\d+), r(?P<r>\d+) for .* failed:")
LOCAL_FAILED = re.compile(r"Local extraction of p(?P<p>\d+), r(?P<r>\d+) failed:")
TRACEBACK_ISSUES = {
    "permission_error": re.compile(r"^\s*PermissionError:", re.M),
    "database_locked": re.compile(r"^\s*sqlite3\.OperationalError: database is locked\b", re.M),
}
CRASH = re.compile(r"Main process exited, code=(?:dumped|killed|exited, status=[1-9]\d*)\b")


def _key(match):
    if match and match["p"] is not None:
        return [int(match["p"]), int(match["r"])]


def _combiner(line):
    if line.startswith("Combining "):
        if match := COMBINING.fullmatch(line):
            # A digest identifies repeated source files without persisting paths.
            digest = hashlib.sha256(match["source"].encode("utf-8", "surrogatepass")).hexdigest()
            return dict(kind="fragment", key=_key(match), path_hash=digest,
                        pending=float(match["pending"]) if match["pending"] else None)
        return dict(kind="reset")
    if match := FILE_TIME.fullmatch(line):
        return dict(kind="merge", seconds=float(match[1]))
    if line.startswith("Created ") and line.endswith(" by renaming"):
        return dict(kind="rename", key=_key(CREATED.fullmatch(line)))
    if line.startswith("File ") and line.endswith(" after notification, skipping"):
        return dict(kind="skip", key=_key(MISSING.fullmatch(line)))
    if match := DB_UPDATE.fullmatch(line):
        return dict(kind="update", key=_key(match), variables=int(match["count"]))
    if match := DB_TIME.fullmatch(line):
        seconds = (float(match["transaction"]) + float(match["finish"])) / 1000
        return dict(kind=match["result"], key=_key(match), seconds=seconds)
    if line.startswith("Unexpected error processing file submission message"):
        return dict(kind="processing_error")
    if "Heartbeat session expired" in line:
        return dict(kind="heartbeat_expired")
    if line.startswith("Offset commit failed:"):
        return dict(kind="offset_commit_failure")
    if line.startswith(("Waiting for file submission", "Stopping on ")):
        return dict(kind="reset")


def _listener(line):
    if match := ADDED.fullmatch(line):
        return dict(kind="added", key=_key(match), data=match["data"])
    if LAUNCHED.fullmatch(line):
        return dict(kind="launch")
    if match := PROCESS_FAILED.fullmatch(line):
        return dict(kind="failure", key=_key(match))
    if match := LOCAL_FAILED.fullmatch(line):
        return dict(kind="local_failure", key=_key(match))
    if line.startswith("Unexpected error handling Kafka event"):
        return dict(kind="failure")
    if line.startswith(("Slurm job submission failed", "Started listener", "Stopping on ")):
        return dict(kind="reset")


def _service(record):
    # User-manager messages describe the target through USER_UNIT, while the
    # manager itself belongs to a different _SYSTEMD_USER_UNIT.
    for field in ("USER_UNIT", "OBJECT_SYSTEMD_USER_UNIT", "_SYSTEMD_USER_UNIT", "_SYSTEMD_UNIT"):
        unit = record.get(field)
        if unit in ("damnit-listener.service", "damnit-combiner.service"):
            return unit.removeprefix("damnit-").removesuffix(".service")
    identifier = record.get("SYSLOG_IDENTIFIER")
    if identifier in ("listener", "combiner"):
        return identifier


def parse_record(record):
    """Return (epoch microseconds, JSON-safe observation) or None for unrelated lines."""
    if not isinstance(record, dict):
        return
    service = _service(record)
    text = record.get("MESSAGE")
    if service is None or not isinstance(text, str) or not text:
        return
    try:
        timestamp = int(record["__REALTIME_TIMESTAMP"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return
    text = LOG_PREFIX.sub("", text, count=1)
    line = text.partition("\n")[0]
    if service == "combiner":
        if line.startswith("Starting ") and "DAMNIT output file combiner service" in line:
            event = dict(kind="service_start")
        elif CRASH.search(line):
            event = dict(kind="service_crash")
        else:
            event = _combiner(line)
        details = [name for name, pattern in TRACEBACK_ISSUES.items() if pattern.search(text)]
        if details:
            if event is None:
                event = dict(kind="traceback")
            event["details"] = details
    else:
        event = _listener(line)

    if event:
        source = [record.get(field) for field in ("_BOOT_ID", "_SYSTEMD_INVOCATION_ID", "_PID")]
        if all(isinstance(field, (str, int, type(None))) for field in source):
            return timestamp, {"service": service, "source": source, **event}
