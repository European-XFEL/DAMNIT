"""Statistics from the journal messages emitted by DAMNIT's central services."""
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone


# systemd stores Python's formatted log line in MESSAGE. The combiner uses
# basicConfig's default format and the listener uses the damnit CLI formatter.
LOG_PREFIX = re.compile(
    r"^(?:(?:DEBUG|INFO|WARNING|ERROR|CRITICAL):[\w.]+:"
    r"|\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+\w+\s+[\w.]+\s+)"
)
NUMBER = r"\d+(?:\.\d+)?"
COMBINING = re.compile(
    rf"""Combining .* into .*?p(\d+)_r(\d+)\.h5['"]?\)?; """
    rf"event pending for (-?{NUMBER}) s before processing"
)
FILE_TIME = re.compile(rf"Finished combining in ({NUMBER}) s")
CREATED = re.compile(r"""Created .*?p(\d+)_r(\d+)\.h5['"]?\)? by renaming""")
MISSING = re.compile(r"File .*?p(\d+)_r(\d+)\.[^/]* not present \d+ s after notification, skipping")
DB_UPDATE = re.compile(
    r"Updating database in .* with (\d+) variables for p(\d+) r(\d+) from(?: .*)?"
)
DB_TIME = re.compile(
    rf"DB change \(set_variables p(\d+) r(\d+)\): ({NUMBER}) ms in transaction, "
    rf"({NUMBER}) ms (commit|rollback)"
)
ADDED = re.compile(r"Added p(\d+) r(\d+) \((raw|proc) data\) to database")
LAUNCHED = re.compile(r"Launched Slurm \([^)]+\) job \S+ to run context file")
PROCESS_FAILED = re.compile(r"Processing p(\d+), r(\d+) for .* failed:")
LOCAL_FAILED = re.compile(r"Local extraction of p(\d+), r(\d+) failed:")

COUNTS = (
    "submissions", "raw_submissions", "proc_submissions", "unclassified_submissions",
    "listener_failures", "fragments", "renames", "skipped", "variables", "variable_samples",
)
DURATIONS = ("pending", "file", "database")


def _combiner_event(message, source, active):
    """Return the affected run and the metric changes for one message."""
    if message.startswith("Combining "):
        active.pop(source, None)
        if match := COMBINING.fullmatch(message):
            key = int(match[1]), int(match[2])
            active[source] = key
            return key, {"fragments": 1, "pending_seconds": float(match[3]), "pending_samples": 1}
    elif match := FILE_TIME.fullmatch(message):
        return active.pop(source, None), {"file_seconds": float(match[1]), "file_samples": 1}
    elif message.startswith("Created ") and message.endswith(" by renaming"):
        key = active.pop(source, None)
        if match := CREATED.fullmatch(message):
            key = int(match[1]), int(match[2])
        return key, {"renames": 1}
    elif message.startswith("File ") and message.endswith(" after notification, skipping"):
        key = active.pop(source, None)
        if key is None and (match := MISSING.fullmatch(message)):
            key = int(match[1]), int(match[2])
        return key, {"skipped": 1}
    elif match := DB_UPDATE.fullmatch(message):
        active.pop(source, None)
        return (int(match[2]), int(match[3])), {"variables": int(match[1]), "variable_samples": 1}
    elif match := DB_TIME.fullmatch(message):
        if match[5] == "commit":
            seconds = (float(match[3]) + float(match[4])) / 1000
            return (int(match[1]), int(match[2])), {"database_seconds": seconds, "database_samples": 1}
    elif message.startswith(("Unexpected error processing", "Waiting for file submission", "Stopping on ")):
        active.pop(source, None)


def _listener_event(message, source, active):
    if match := ADDED.fullmatch(message):
        active[source] = ((int(match[1]), int(match[2])), match[3])
    elif LAUNCHED.fullmatch(message):
        key, kind = active.pop(source, (None, "unclassified"))
        return key, {"submissions": 1, f"{kind}_submissions": 1}
    elif match := PROCESS_FAILED.fullmatch(message):
        active.pop(source, None)
        return (int(match[1]), int(match[2])), {"listener_failures": 1}
    elif match := LOCAL_FAILED.fullmatch(message):
        # Local extractions run in separate threads. A failure may be logged
        # while the listener is submitting another run.
        return (int(match[1]), int(match[2])), {"listener_failures": 1}
    elif message.startswith("Unexpected error handling Kafka event"):
        active.pop(source, None)
        return None, {"listener_failures": 1}
    elif message.startswith(("Slurm job submission failed", "Started listener", "Stopping on ")):
        active.pop(source, None)


def _metrics(totals):
    result = {name: totals[name] for name in COUNTS}
    for name in DURATIONS:
        samples = totals[f"{name}_samples"]
        result[f"{name}_samples"] = samples
        result[f"{name}_seconds"] = totals[f"{name}_seconds"] / samples if samples else None
    return result


def _bucket_seconds(span):
    for seconds in (60, 300, 900, 3600, 21600, 86400):
        if span <= 96 * seconds:
            return seconds
    return 86400


def summarize(records, start, end, proposal=None, run=None):
    """Aggregate chronological journal records in the half-open interval [start, end)."""
    first = start.timestamp()
    last = end.timestamp()
    bucket_seconds = _bucket_seconds(last - first)
    first_bucket = int(first // bucket_seconds) * bucket_seconds
    totals = Counter()
    by_run = defaultdict(Counter)
    by_time = defaultdict(Counter)
    listeners = {}
    combiners = {}

    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            timestamp = int(record["__REALTIME_TIMESTAMP"]) / 1_000_000
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if timestamp >= last:
            continue
        message = record.get("MESSAGE")
        if not isinstance(message, str) or not message:
            continue
        message = LOG_PREFIX.sub("", message.partition("\n")[0])
        source = (
            record.get("_BOOT_ID"), record.get("_SYSTEMD_INVOCATION_ID"), record.get("_PID")
        )
        if not all(isinstance(field, (str, int, type(None))) for field in source):
            continue
        unit = record.get("_SYSTEMD_USER_UNIT") or record.get("_SYSTEMD_UNIT")
        identifier = record.get("SYSLOG_IDENTIFIER")
        if unit == "damnit-combiner.service" or identifier == "combiner":
            event = _combiner_event(message, source, combiners)
        elif unit == "damnit-listener.service" or identifier == "listener":
            event = _listener_event(message, source, listeners)
        else:
            continue
        if event is None or timestamp < first:
            continue

        key, changes = event
        if proposal is not None and (key is None or key[0] != proposal):
            continue
        if run is not None and (key is None or key[1] != run):
            continue
        totals.update(changes)
        bucket = int(timestamp // bucket_seconds) * bucket_seconds
        by_time[bucket].update(changes)
        if key is not None:
            by_run[key].update(changes)

    series = []
    bucket = first_bucket
    while bucket < last:
        time = datetime.fromtimestamp(bucket, timezone.utc).isoformat()
        series.append({"time": time, **_metrics(by_time[bucket])})
        bucket += bucket_seconds
    return {
        "start": start.astimezone(timezone.utc).isoformat(),
        "end": end.astimezone(timezone.utc).isoformat(),
        "bucket_seconds": bucket_seconds,
        "summary": {
            **_metrics(totals),
            "proposal_count": len({p for p, r in by_run}),
            "run_count": len(by_run),
        },
        "series": series,
        "runs": [
            {"proposal": p, "run": r, **_metrics(counts)}
            for (p, r), counts in sorted(by_run.items())
        ],
    }
