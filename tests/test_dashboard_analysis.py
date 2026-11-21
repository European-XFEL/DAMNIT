import json
from datetime import datetime, timedelta, timezone

import pytest

from damnit.dashboard.stats import MAX_SCATTER_POINTS, parse_record, summarize, summarize_events


START = datetime(2026, 7, 13, tzinfo=timezone.utc)
END = START + timedelta(minutes=2)


def entry(seconds, message, pid="123", **fields):
    return {
        "__REALTIME_TIMESTAMP": str(int(START.timestamp()) * 1_000_000 + seconds * 1_000_000),
        "_SYSTEMD_USER_UNIT": "damnit-combiner.service", "_PID": pid, "MESSAGE": message, **fields,
    }


def fragment(seconds, filename="fragment-a", run=1, pending=None):
    line = f"INFO:__main__:Combining PosixPath('/private/path/{filename}') into PosixPath('/db/p12_r{run}.h5')"
    if pending is not None:
        line += f"; event pending for {pending} s before processing"
    return entry(seconds, line)


def update(seconds, variables, run=1):
    return entry(seconds, f"Updating database in /db with {variables} variables for p12 r{run} from context.py")


def commit(seconds, milliseconds, run=1):
    return entry(seconds, f"DB change (set_variables p12 r{run}): {milliseconds} ms in transaction, 2 ms commit")


def test_historical_starts_and_redeliveries_in_selection():
    records = [
        fragment(-1),
        fragment(0),
        entry(1, "Finished combining in 2 s"),
        fragment(2),
        entry(3, "File /private/path/fragment-a not present 0 s after notification, skipping"),
        fragment(61, pending=0.2),
        fragment(62, "other", run=2),
        entry(63, "Created PosixPath('/db/p12_r2.h5') by renaming"),
    ]
    cached = json.dumps([parse_record(record) for record in records])
    assert "/private/path" not in cached and "PosixPath" not in cached
    result = summarize_events(json.loads(cached), START, END)
    assert result == summarize(records, START, END)
    summary = result["summary"]
    assert (summary["fragments"], summary["file_samples"], summary["incomplete"]) == (4, 1, 2)
    assert (summary["renames"], summary["pending_samples"], summary["skipped"]) == (1, 1, 1)
    assert summary["operations"]["duplicate_source_paths"] == 1
    assert summary["operations"]["extra_deliveries"] == 2
    assert [b["operations"]["extra_deliveries"] for b in result["series"]] == [1, 1]
    assert summarize(records, START, END, run=2)["summary"]["operations"]["extra_deliveries"] == 0


@pytest.mark.parametrize("multiline", [False, True])
def test_processing_issues_and_manager_records(multiline):
    header = "ERROR:__main__:Unexpected error processing file submission message"
    traceback = "Traceback (most recent call last):\n  File \"combine.py\", line 25\nPermissionError: [Errno 13] denied"
    failure = [entry(1, header + "\n" + traceback)] if multiline else [entry(1, header), entry(1, traceback)]
    manager = dict(SYSLOG_IDENTIFIER="systemd", USER_UNIT="damnit-combiner.service",
                   _SYSTEMD_USER_UNIT="init.scope")
    records = [fragment(0), *failure, fragment(2, "other", run=2),
               entry(3, "DB change (set_variables p12 r2): 10.0 ms in transaction, 1.2 ms rollback"),
               entry(4, header), entry(4, "sqlite3.OperationalError: database is locked"),
               entry(5, "WARNING:kafka.coordinator.heartbeat:Heartbeat session expired, marking coordinator dead"),
               entry(6, "ERROR:kafka.coordinator.consumer:Offset commit failed: This is likely to cause duplicate message delivery"),
               entry(7, "Starting DAMNIT output file combiner service...", **manager),
               entry(8, "Started DAMNIT output file combiner service.", **manager),
               entry(9, "damnit-combiner.service: Main process exited, code=dumped, status=7/BUS", **manager),
               entry(9, "damnit-combiner.service: Failed with result 'core-dump'.", **manager)]
    ops = summarize(records, START, END)["summary"]["operations"]
    assert {k: v for k, v in ops.items() if v} == {
        "unexpected_processing_error": 2, "permission_error": 1,
        "database_rollback": 1, "database_locked": 1, "heartbeat_expired": 1,
        "offset_commit_failure": 1, "service_start": 1, "service_crash": 1,
    }
    filtered = summarize(records, START, END, run=1)["summary"]["operations"]
    assert {k: v for k, v in filtered.items() if v} == {"unexpected_processing_error": 1, "permission_error": 1}


def test_distributions_and_same_update_pairing():
    records = [
        update(-1, 2), commit(0, 8),  # Pre-window metadata can identify this commit.
        update(1, 10), commit(2, 98),
        update(3, 9), commit(4, 998, run=2),  # Different run: timing only.
        update(5, 99), fragment(6, pending=0), commit(7, 1998),  # Intervening start.
        update(8, 100), entry(9, "DB change (set_variables p12 r1): 1 ms in transaction, 0 ms rollback"),
        commit(10, 2998),  # No pending update after rollback.
        update(11, 300), entry(12, "DB change (set_variables p12 r1): 1 ms in transaction, 0 ms commit", pid="other"),
    ]
    data = summarize(records, START, END)
    dist = data["distributions"]["database"]
    assert (dist["count"], dist["p50"], dist["p95"], dist["p99"], dist["max"]) == pytest.approx(
        (6, 0.55, 2.75, 2.95, 3))
    assert data["distributions"]["file"]["p50"] is None
    assert data["distributions"]["pending"]["p50"] == 0
    assert data["series"][0]["database_p95_seconds"] == pytest.approx(2.75)
    assert data["scaling"]["count"] == 2
    assert data["scaling"]["spearman"] == pytest.approx(1)
    assert [p["variables"] for p in data["scaling"]["points"]] == [2, 10]


def test_scatter_samples_preserve_outliers():
    records = []
    for index in range(MAX_SCATTER_POINTS + 10):
        records.extend((update(0, index), commit(0, 10000 if index == 56 else index)))
    result = summarize(records, START, END)["scaling"]
    assert len(result["points"]) <= MAX_SCATTER_POINTS < result["count"]
    assert max(p["database_seconds"] for p in result["points"]) == pytest.approx(10.002)
    assert max(p["variables"] for p in result["points"]) == MAX_SCATTER_POINTS + 9
