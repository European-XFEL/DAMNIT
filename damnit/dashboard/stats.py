"""Statistics from independent journal observations in chronological order."""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .events import parse_record


COUNTS = (
    "submissions", "raw_submissions", "proc_submissions", "unclassified_submissions",
    "dispatch_attempts", "local_fallbacks", "local_failures",
    "capacity_skips", "db_after_file_failures", "pending_exceedances", "unattributed_file_samples",
    "listener_failures", "fragments", "incomplete", "renames", "skipped", "variables", "variable_samples",
)
DURATIONS = ("pending", "file", "database")
OPERATIONS = (
    "heartbeat_expired", "offset_commit_failure", "unexpected_processing_error",
    "missing_notified_file", "permission_error", "database_locked", "database_rollback",
    "service_start", "service_crash", "duplicate_source_paths", "extra_deliveries",
)
SYSTEM_EVENTS = {"heartbeat_expired", "offset_commit_failure", "service_start", "service_crash"}
MAX_SCATTER_POINTS = 1000
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MICROSECOND = timedelta(microseconds=1)


@dataclass
class Measurement:
    timestamp: int
    key: tuple | None
    counts: dict
    path_hash: str | None = None
    paired_variables: int | None = None
    attempt: object | None = None
    issue: str | None = None
    stage: str | None = None
    data_type: str | None = None
    fragment: str | None = None


@dataclass
class Fragment:
    key: tuple | None = None
    awaiting_file_since: int | None = None
    variables: int | None = None
    failure_key: tuple | None = None

    attempt: object | None = None
    began: int | None = None
    file_stage: str | None = None
    terminal: bool = False
    error_seen: bool = False
    filename: str | None = None

    def close(self):
        if self.began is not None and not self.terminal:
            stage = "Database write" if self.file_stage else "File completion"
            yield Measurement(self.began, self.key, {}, attempt=self.attempt, issue="unobserved",
                              stage=stage, fragment=self.filename)
        self.began = None
        yield from self.incomplete()

    def incomplete(self):
        if self.awaiting_file_since is not None:
            yield Measurement(self.awaiting_file_since, self.key, {"incomplete": 1})
            self.awaiting_file_since = None

    def observe(self, timestamp, event, key):
        kind = event["kind"]
        known_key = self.key or self.failure_key
        if (key is not None and known_key is not None and key != known_key) or (
                kind == "processing_error" and self.error_seen):
            # A keyed record for another run, or a new independent error after
            # an exception, cannot complete the previous worker attempt.
            yield from self.close()
            self.key = self.variables = self.failure_key = self.attempt = None
            self.file_stage = self.filename = None
            self.terminal = self.error_seen = False
        if kind == "fragment":
            yield from self.close()
            self.filename = event.get("fragment")
            self.began, self.attempt = timestamp, object()
            self.file_stage = None
            self.terminal = self.error_seen = False
            self.key, self.awaiting_file_since = key, timestamp
            self.variables = self.failure_key = None
            changes = {"fragments": 1}
            if event["pending"] is not None:
                changes["pending_seconds"] = event["pending"]
            yield Measurement(timestamp, key, changes, path_hash=event["path_hash"],
                              attempt=self.attempt, fragment=self.filename)
        elif kind in ("merge", "rename"):
            if self.awaiting_file_since is not None:
                key = self.key
            self.awaiting_file_since = None
            self.key = key
            self.file_stage = "File merged" if kind == "merge" else "File created"
            changes = {"renames": 1} if kind == "rename" else {"file_seconds": event["seconds"]}
            if kind == "merge" and key is None:
                changes["unattributed_file_samples"] = 1
            yield Measurement(timestamp, key, changes)
        elif kind == "skip":
            self.terminal = True
            yield Measurement(timestamp, key or self.key, {"skipped": 1, "missing_notified_file": 1},
                              attempt=self.attempt, issue="missing_notified_file", stage="File availability",
                              fragment=self.filename)
            yield from self.incomplete()
            self.key = self.variables = self.failure_key = None
        elif kind == "update":
            yield from self.incomplete()
            self.key, self.variables = key, event["variables"]
            self.failure_key = None
            yield Measurement(timestamp, key, {"variables": self.variables, "variable_samples": 1})
        elif kind in ("commit", "rollback"):
            yield from self.incomplete()
            key = key or self.key
            if kind == "commit":
                self.terminal = True
                matched = self.variables if key == self.key else None
                yield Measurement(timestamp, key, {"database_seconds": event["seconds"]},
                                  paired_variables=matched)
                self.filename = None
                self.key = self.failure_key = None
                self.attempt = None
                self.file_stage = None
            else:
                self.failure_key = key
                changes = {"database_rollback": 1}
                if self.attempt is None:
                    self.attempt = object()
                if self.file_stage and not self.terminal:
                    changes["db_after_file_failures"] = 1
                self.terminal = True
                yield Measurement(timestamp, key, changes, attempt=self.attempt,
                                  issue="db_after_file_failure" if self.file_stage else "database_rollback",
                                  stage=f"{self.file_stage} · DB rolled back" if self.file_stage else "DB rolled back",
                                  fragment=self.filename)
            self.variables = None
        elif kind in ("processing_error", "reset"):
            yield from self.incomplete()
            if kind == "processing_error":
                self.error_seen = True
                self.failure_key = key or self.key or self.failure_key
                if self.attempt is None:
                    self.attempt = object()
                changes = {"unexpected_processing_error": 1}
                # The update message marks the start of the database write.
                # An earlier error, even a lock opening the DB, does not prove a write failed.
                database_failure = self.file_stage is not None and self.variables is not None
                if database_failure and not self.terminal:
                    changes["db_after_file_failures"] = 1
                self.terminal = True
                stage = "Fragment processing"
                if self.file_stage:
                    stage = self.file_stage + (" · DB write failed" if database_failure else " · Processing failed")
                yield Measurement(timestamp, self.failure_key, changes, attempt=self.attempt,
                                  issue="db_after_file_failure" if database_failure else "unexpected_processing_error",
                                  stage=stage, fragment=self.filename)
            else:
                yield from self.close()
                self.attempt = self.failure_key = None
                self.file_stage = self.filename = None
                self.error_seen = self.terminal = False
            self.key = self.variables = None

        if details := event.get("details"):
            yield Measurement(timestamp, self.failure_key or self.key, dict.fromkeys(details, 1), attempt=self.attempt)


@dataclass
class PendingDispatch:
    timestamp: int
    key: tuple | None
    data: str | None
    kind: str

    def matches(self, key, data):
        return (key is None or self.key is None or key == self.key) and (
            data is None or self.data is None or data == self.data)

    def unobserved(self):
        # A fallback request is a recorded dispatch decision. The current
        # listener never logs a reliable local-worker start or successful exit.
        if self.kind == "added":
            yield Measurement(self.timestamp, self.key, {}, issue="unobserved",
                              stage="Run registered · Dispatch outcome unobserved", data_type=self.data)


def _listener_measurements(pending, source, timestamp, event, key):
    kind = event["kind"]
    previous = pending.get(source)
    if kind == "added":
        if previous:
            yield from previous.unobserved()
        pending[source] = PendingDispatch(timestamp, key, event["data"], kind)
        yield Measurement(timestamp, key, {"dispatch_attempts": 1})
    elif kind == "fallback":
        data = None
        if previous and previous.kind == "added":
            key, data = previous.key, previous.data
        pending[source] = PendingDispatch(timestamp, key, data, kind)
        yield Measurement(timestamp, key, {"local_fallbacks": 1})
    elif kind == "launch":
        data = "unclassified"
        if previous and previous.kind == "added":
            pending.pop(source)
            key, data = previous.key, previous.data
        yield Measurement(timestamp, key, {"submissions": 1, f"{data}_submissions": 1})
    elif kind == "capacity_skip":
        if previous and previous.matches(key, event["data"]):
            pending.pop(source)
        yield Measurement(timestamp, key, {"capacity_skips": 1}, issue="capacity_skip",
                          stage="Local extraction", data_type=event["data"])
    elif kind == "failure":
        data = None
        changes = {"listener_failures": 1}
        if key is not None:
            if previous and previous.matches(key, None):
                data = pending.pop(source).data
            else:
                # A per-database failure can precede the registration message.
                changes["dispatch_attempts"] = 1
        yield Measurement(timestamp, key, changes, issue="listener_failure", stage="Dispatch", data_type=data)
    elif kind == "local_failure":
        # The background worker's message includes the run but no data type or
        # dispatch identity. Another dispatch for the same run can be in flight.
        yield Measurement(timestamp, key, {"local_failures": 1, "listener_failures": 1},
                          issue="local_failure", stage="Local extraction")
    elif kind == "reset" and previous:
        yield from pending.pop(source).unobserved()


def _measurements(events):
    listeners = {}
    fragments = defaultdict(Fragment)
    for timestamp, event in events:
        source = tuple(event["source"])
        if not any(source[1:]):
            # A boot ID alone cannot tie two records to one worker.
            source = object()
        key = tuple(event["key"]) if event.get("key") is not None else None
        kind = event["kind"]
        if kind in SYSTEM_EVENTS:
            # Kafka and systemd records have no proposal/run identity.
            yield Measurement(timestamp, None, {kind: 1}, issue=kind if kind == "service_crash" else None,
                              stage="Service")
        elif event["service"] == "combiner":
            yield from fragments[source].observe(timestamp, event, key)
        else:
            yield from _listener_measurements(listeners, source, timestamp, event, key)
    for dispatch in listeners.values():
        yield from dispatch.unobserved()
    for fragment in fragments.values():
        yield from fragment.close()


@dataclass
class Metrics:
    counts: Counter = field(default_factory=Counter)
    times: dict = field(default_factory=lambda: {name: [] for name in DURATIONS})

    def add(self, counts):
        self.counts.update(counts)
        for name, values in self.times.items():
            if (seconds := counts.get(name + "_seconds")) is not None:
                values.append(seconds)

    def result(self, operations=False, trends=False):
        result = {name: self.counts[name] for name in COUNTS}
        for name, values in self.times.items():
            result[name + "_samples"] = len(values)
            result[name + "_seconds"] = self.counts[name + "_seconds"] / len(values) if values else None
            if trends:
                p50, p95 = np.percentile(values, [50, 95]).tolist() if values else (None, None)
                result[name + "_p50_seconds"] = p50
                result[name + "_p95_seconds"] = p95
        if operations:
            result["operations"] = {name: self.counts[name] for name in OPERATIONS}
        return result

    def distributions(self):
        result = {}
        for name, values in self.times.items():
            percentiles = np.percentile(values, [50, 95, 99, 100]).tolist() if values else [None] * 4
            result[name] = dict(zip(("p50", "p95", "p99", "max"), percentiles),
                                count=len(values), mean=sum(values) / len(values) if values else None)
        return result


def _scaling(points):
    """Spearman's rho uses every matched update; sample only the plotted points."""
    correlation = None
    if len(points) >= 2:
        ranks = pd.DataFrame(points, columns=["variables", "database_seconds"]).rank()
        if ranks.nunique().min() > 1:
            correlation = float(ranks.corr().iat[0, 1])
    count = len(points)
    if count > MAX_SCATTER_POINTS:
        ordered = sorted(points, key=lambda item: item["database_seconds"])
        # Sample evenly across observed latencies, preserving both extremes
        # and the full range of variable counts.
        positions = set(np.linspace(0, count - 1, MAX_SCATTER_POINTS - 2, dtype=int))
        positions.add(min(range(count), key=lambda i: ordered[i]["variables"]))
        positions.add(max(range(count), key=lambda i: ordered[i]["variables"]))
        points = [ordered[index] for index in sorted(positions)]
    return {"count": count, "points": points, "spearman": correlation}


def _bucket_seconds(span):
    for seconds in (60, 300, 900, 3600, 21600, 86400):
        if span <= 96 * seconds:
            return seconds
    return 86400


INCIDENT_LIMIT = 50
ISSUE_REASONS = {
    "db_after_file_failure": "File completed, database write failed",
    "database_rollback": "Database transaction rolled back",
    "missing_notified_file": "Notified fragment was missing",
    "unexpected_processing_error": "Fragment processing failed",
    "listener_failure": "Listener processing failed",
    "local_failure": "Local extraction failed",
    "capacity_skip": "Local worker limit reached",
    "service_crash": "Service process crashed",
    "unobserved": "Completion was not observed by the cutoff",
    "slow_pending": "Pending delay exceeded the selected threshold",
}


def _iso(stamp):
    return (EPOCH + stamp * MICROSECOND).isoformat() if stamp is not None else None


def _incident(measurement, kind, stage):
    proposal, run = measurement.key or (None, None)
    return {"kind": kind, "stage": stage, "reason": ISSUE_REASONS[kind], "time": _iso(measurement.timestamp),
            "proposal": proposal, "run": run, "seconds": measurement.counts.get("pending_seconds"),
            "data_type": measurement.data_type, "fragment": measurement.fragment}


def summarize_events(events, start, end, proposal=None, run=None, *, service=None, pending_threshold_seconds=60):
    """Aggregate [start, end). Earlier observations can provide correlation context."""
    first = (start - EPOCH) // MICROSECOND
    last = (end - EPOCH) // MICROSECOND
    bucket_seconds = _bucket_seconds((last - first) / 1_000_000)
    width = bucket_seconds * 1_000_000
    totals = Metrics()
    by_time = defaultdict(Metrics)
    by_run = defaultdict(Metrics)
    deliveries = Counter()
    points = []
    service_operations = Counter()
    global_by_time = defaultdict(Counter)
    failures, slow, unobserved = {}, {}, []
    eligible = (
        (ts, event) for ts, event in events
        if ts < last and (service is None or event["service"] == service)
    )
    for measurement in _measurements(eligible):
        key = measurement.key
        if measurement.timestamp < first:
            continue
        changes = measurement.counts
        bucket_index = measurement.timestamp // width
        for name in SYSTEM_EVENTS.intersection(changes):
            service_operations[name] += changes[name]
            global_by_time[bucket_index][name] += changes[name]
        if proposal is not None and (key is None or key[0] != proposal):
            continue
        if run is not None and (key is None or key[1] != run):
            continue
        attempt = measurement.attempt
        issue = measurement.issue
        if issue == "unobserved":
            unobserved.append(_incident(measurement, issue, measurement.stage))
        elif issue:
            identity = attempt if attempt is not None else object()
            incident = failures.setdefault(identity, _incident(measurement, issue, measurement.stage))
            if measurement.fragment:
                incident["fragment"] = measurement.fragment
        if attempt in failures:
            if "database_locked" in changes:
                failures[attempt]["reason"] = "Database was locked"
            elif "permission_error" in changes:
                failures[attempt]["reason"] = "Permission denied"
        if changes.get("pending_seconds", 0) > pending_threshold_seconds:
            changes["pending_exceedances"] = 1
            slow[attempt] = _incident(measurement, "slow_pending", "Pending")
        if path_hash := measurement.path_hash:
            deliveries[path_hash] += 1
            if deliveries[path_hash] == 2:
                changes["duplicate_source_paths"] = 1
            if deliveries[path_hash] > 1:
                changes["extra_deliveries"] = 1
        if measurement.paired_variables is not None:
            points.append({"proposal": key[0], "run": key[1], "variables": measurement.paired_variables,
                           "database_seconds": changes["database_seconds"]})
        totals.add(changes)
        by_time[bucket_index].add(changes)
        if key is not None:
            by_run[key].add(changes)

    series = []
    for bucket in range(first // width, (last - 1) // width + 1):
        time = datetime.fromtimestamp(bucket * bucket_seconds, timezone.utc).isoformat()
        series.append({"time": time, **by_time[bucket].result(operations=True, trends=True),
                       "service_operations": {name: global_by_time[bucket][name] for name in sorted(SYSTEM_EVENTS)}})
    slow = sorted((row for attempt, row in slow.items() if attempt not in failures),
                  key=lambda row: (row["seconds"], row["time"]), reverse=True)
    failures = sorted(failures.values(), key=lambda row: row["time"], reverse=True)
    unobserved.sort(key=lambda row: row["time"], reverse=True)
    return {
        "start": start.astimezone(timezone.utc).isoformat(),
        "end": end.astimezone(timezone.utc).isoformat(),
        "bucket_seconds": bucket_seconds,
        "pending_threshold_seconds": pending_threshold_seconds,
        "incidents": {"failures": failures[:INCIDENT_LIMIT], "failure_count": len(failures),
                      "slow": slow[:INCIDENT_LIMIT], "slow_count": len(slow),
                      "unobserved": unobserved[:INCIDENT_LIMIT], "unobserved_count": len(unobserved)},
        "summary": {
            **totals.result(operations=True),
            "service_operations": {name: service_operations[name] for name in sorted(SYSTEM_EVENTS)},
            "proposal_count": len({p for p, r in by_run}),
            "run_count": len(by_run),
        },
        "series": series,
        "runs": [
            {"proposal": p, "run": r, **counts.result()}
            for (p, r), counts in sorted(by_run.items())
        ],
        "distributions": totals.distributions(),
        "scaling": _scaling(points),
    }


def summarize(records, start, end, proposal=None, run=None, *, service=None, pending_threshold_seconds=60):
    events = (event for record in records if (event := parse_record(record)) is not None)
    return summarize_events(events, start, end, proposal, run, service=service,
                            pending_threshold_seconds=pending_threshold_seconds)
