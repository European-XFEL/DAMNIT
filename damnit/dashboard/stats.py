"""Statistics from independent journal observations in chronological order."""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .events import parse_record


COUNTS = (
    "submissions", "raw_submissions", "proc_submissions", "unclassified_submissions",
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


@dataclass
class Fragment:
    key: tuple | None = None
    awaiting_file_since: int | None = None
    variables: int | None = None
    failure_key: tuple | None = None

    def incomplete(self):
        if self.awaiting_file_since is not None:
            yield Measurement(self.awaiting_file_since, self.key, {"incomplete": 1})
            self.awaiting_file_since = None

    def observe(self, timestamp, event, key):
        kind = event["kind"]
        if kind == "fragment":
            yield from self.incomplete()
            self.key, self.awaiting_file_since = key, timestamp
            self.variables = self.failure_key = None
            changes = {"fragments": 1}
            if event["pending"] is not None:
                changes["pending_seconds"] = event["pending"]
            yield Measurement(timestamp, key, changes, path_hash=event["path_hash"])
        elif kind in ("merge", "rename"):
            if key is not None and key != self.key:
                yield from self.incomplete()
            if self.awaiting_file_since is not None:
                key = self.key
            self.awaiting_file_since = None
            changes = {"renames": 1} if kind == "rename" else {"file_seconds": event["seconds"]}
            yield Measurement(timestamp, key, changes)
        elif kind == "skip":
            yield Measurement(timestamp, key or self.key, {"skipped": 1, "missing_notified_file": 1})
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
                matched = self.variables if key == self.key else None
                yield Measurement(timestamp, key, {"database_seconds": event["seconds"]},
                                  paired_variables=matched)
                self.key = self.failure_key = None
            else:
                self.failure_key = key
                yield Measurement(timestamp, key, {"database_rollback": 1})
            self.variables = None
        elif kind in ("processing_error", "reset"):
            yield from self.incomplete()
            if kind == "processing_error":
                self.failure_key = self.key or self.failure_key
                yield Measurement(timestamp, self.failure_key, {"unexpected_processing_error": 1})
            else:
                self.failure_key = None
            self.key = self.variables = None

        if details := event.get("details"):
            yield Measurement(timestamp, self.failure_key or self.key, dict.fromkeys(details, 1))


def _measurements(events):
    listeners = {}
    fragments = defaultdict(Fragment)
    for timestamp, event in events:
        source = tuple(event["source"])
        key = tuple(event["key"]) if event.get("key") is not None else None
        kind = event["kind"]
        if kind in SYSTEM_EVENTS:
            # Kafka and systemd records have no proposal/run identity.
            yield Measurement(timestamp, None, {kind: 1})
        elif event["service"] == "combiner":
            yield from fragments[source].observe(timestamp, event, key)
        elif kind == "added":
            listeners[source] = key, event["data"]
        elif kind == "launch":
            key, data = listeners.pop(source, (None, "unclassified"))
            yield Measurement(timestamp, key, {"submissions": 1, f"{data}_submissions": 1})
        elif kind in ("failure", "local_failure"):
            # Local extraction threads may fail during another submission.
            if kind == "failure":
                listeners.pop(source, None)
            yield Measurement(timestamp, key, {"listener_failures": 1})
        elif kind == "reset":
            listeners.pop(source, None)
    for fragment in fragments.values():
        yield from fragment.incomplete()


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


def summarize_events(events, start, end, proposal=None, run=None, *, service=None):
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

    eligible = (
        (ts, event) for ts, event in events
        if ts < last and (service is None or event["service"] == service)
    )
    for measurement in _measurements(eligible):
        key = measurement.key
        if measurement.timestamp < first:
            continue
        if proposal is not None and (key is None or key[0] != proposal):
            continue
        if run is not None and (key is None or key[1] != run):
            continue
        changes = measurement.counts
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
        by_time[measurement.timestamp // width].add(changes)
        if key is not None:
            by_run[key].add(changes)

    series = []
    for bucket in range(first // width, (last - 1) // width + 1):
        time = datetime.fromtimestamp(bucket * bucket_seconds, timezone.utc).isoformat()
        series.append({"time": time, **by_time[bucket].result(operations=True, trends=True)})
    return {
        "start": start.astimezone(timezone.utc).isoformat(),
        "end": end.astimezone(timezone.utc).isoformat(),
        "bucket_seconds": bucket_seconds,
        "summary": {
            **totals.result(operations=True),
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


def summarize(records, start, end, proposal=None, run=None, *, service=None):
    events = (event for record in records if (event := parse_record(record)) is not None)
    return summarize_events(events, start, end, proposal, run, service=service)
