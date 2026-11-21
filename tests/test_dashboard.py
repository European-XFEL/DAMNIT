from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from damnit.dashboard import app as dashboard
from damnit.dashboard import cache
from damnit.dashboard.stats import summarize


START = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
END = START + timedelta(minutes=5)


def entry(unit, second, message, invocation="a"):
    return {
        "__REALTIME_TIMESTAMP": str(int((START.timestamp() + second) * 1_000_000)),
        "_SYSTEMD_USER_UNIT": f"damnit-{unit}.service",
        "_SYSTEMD_INVOCATION_ID": invocation,
        "MESSAGE": message,
    }


def fragment(second, proposal, run, pending):
    return entry("combiner", second, f"INFO:__main__:Combining PosixPath('/a.ready.h5') "
                 f"into PosixPath('/p{proposal}_r{run}.h5'); event pending for {pending} s before processing")


@pytest.fixture
def combiner_records():
    return [
        fragment(1, 17, 2, 0.2),
        entry("combiner", 2, "DEBUG:__main__:Finished combining in 2.100 s"),
        entry("combiner", 3, "INFO:__main__:Updating database in /db with 8 variables for p17 r2 from context.py"),
        entry("combiner", 4, "INFO:damnit.backend.extract_data:Adding p17 r2 to database, with 8 columns"),
        entry("combiner", 5, "DEBUG:root:DB change (ensure_run): 1.0 ms in transaction, 1.1 ms commit"),
        entry("combiner", 6, "DEBUG:root:DB change (set_variables p17 r2): 30.0 ms in transaction, 2.0 ms commit"),
        fragment(65, 17, 2, 0.4),
        entry("combiner", 66, "DEBUG:__main__:Finished combining in 0.300 s"),
        entry("combiner", 67, "INFO:__main__:Updating database in /db with 4 variables for p17 r2 from context.py"),
        entry("combiner", 68, "DEBUG:root:DB change (set_variables p17 r2): 2.0 ms in transaction, 0.1 ms rollback"),
        fragment(70, 18, 2, 0.9),
        entry("combiner", 71, "DEBUG:__main__:Created PosixPath('/p18_r2.h5') by renaming"),
        fragment(75, 18, 3, 0.1),
        entry("combiner", 76, "WARNING:__main__:File /a.ready.h5 not present 30 s after notification, skipping"),
        fragment(300, 99, 100, 100),  # Exclusive upper bound
    ]


def test_combiner_counts_timings_and_buckets(combiner_records):
    result = summarize(iter(combiner_records), START, END)
    totals = result["summary"]
    assert (totals["proposal_count"], totals["run_count"], totals["fragments"]) == (2, 3, 4)
    assert (totals["variables"], totals["variable_samples"], totals["renames"], totals["skipped"]) == (12, 2, 1, 1)
    assert (totals["pending_samples"], totals["file_samples"], totals["database_samples"]) == (4, 2, 1)
    assert totals["pending_seconds"] == pytest.approx(0.4)
    assert totals["file_seconds"] == pytest.approx(1.2)
    assert totals["database_seconds"] == pytest.approx(0.032)
    assert [(row["proposal"], row["run"]) for row in result["runs"]] == [(17, 2), (18, 2), (18, 3)]
    assert result["runs"][1]["file_seconds"] is None
    assert result["bucket_seconds"] == 60
    assert [bucket["fragments"] for bucket in result["series"]] == [1, 3, 0, 0, 0]


def test_listener_submissions_and_restarts():
    def listener(second, message, invocation="a"):
        return entry("listener", second, "2026-10-01 09:00:00 INFO     damnit.backend.listener    " + message, invocation)

    launch = "Launched Slurm (maxwell) job 123 to run context file"
    records = [
        listener(0, launch),  # Run context precedes the journal query.
        listener(1, "Added p7 r3 (raw data) to database"),
        listener(2, launch),
        listener(3, "Added p7 r3 (proc data) to database"),
        listener(4, launch),
        listener(5, "Added p7 r4 (raw data) to database"),
        listener(6, "Processing p7, r4 for /some/db failed:"),
        listener(7, "Added p7 r5 (raw data) to database"),
        listener(8, launch, invocation="b"),
    ]
    summary = summarize(records, START, END)["summary"]
    assert (summary["submissions"], summary["raw_submissions"], summary["proc_submissions"],
            summary["unclassified_submissions"], summary["listener_failures"]) == (4, 1, 1, 2, 1)
    filtered = summarize(records, START, END, proposal=7, run=3)
    assert filtered["summary"]["submissions"] == 2
    assert filtered["summary"]["listener_failures"] == 0
    assert filtered["series"][0]["submissions"] == 2


def test_uncorrelated_merge_does_not_inherit_run():
    records = [
        fragment(0, 1, 2, 0.1),
        entry("combiner", 1, "DEBUG:__main__:Finished combining in 5.000 s", invocation="b"),
        entry("combiner", 2, "ERROR:__main__:Unexpected error processing file submission message"),
        entry("combiner", 3, "DEBUG:__main__:Finished combining in 6.000 s"),
    ]
    summary = summarize(records, START, END)["summary"]
    assert summary["file_samples"] == 2
    assert summarize(records, START, END, run=2)["summary"]["file_samples"] == 0


def test_stats_api(monkeypatch, tmp_path, combiner_records):
    def read_journal(start, end):
        assert (start, end) == (START - cache.LOOKBACK, END)
        yield from combiner_records

    monkeypatch.setenv("DAMNIT_DASHBOARD_CACHE", str(tmp_path / "stats.sqlite3"))
    monkeypatch.setattr(cache, "read_journal", read_journal)
    params = {"start": START.isoformat(), "end": END.isoformat(), "proposal": 17, "run": 2}
    with TestClient(dashboard.app) as client:
        response = client.get("/api/stats", params=params)
        assert response.status_code == 200
        assert response.json()["summary"]["fragments"] == 2
        assert len(response.json()["runs"]) == 1
        empty = client.get("/api/stats", params={**params, "run": 100}).json()
        assert empty["runs"] == []
        assert empty["summary"]["pending_seconds"] is None
        assert client.get("/").status_code == 200
        for asset in ["/static/dashboard.js", "/static/dashboard.css", "/assets/plotly.min.js"]:
            assert client.get(asset).is_success


def test_stats_interval_and_journal_failure(monkeypatch, tmp_path):
    def broken_journal(start, end):
        assert end - start == timedelta(days=1) + cache.LOOKBACK
        raise dashboard.JournalError("Unavailable")

    monkeypatch.setenv("DAMNIT_DASHBOARD_CACHE", str(tmp_path / "stats.sqlite3"))
    monkeypatch.setattr(cache, "read_journal", broken_journal)
    with TestClient(dashboard.app) as client:
        assert client.get("/api/stats", params={"end": END.isoformat()}).status_code == 503
        for params in [
            {"start": "2026-10-01T00:00:00"},
            {"start": END.isoformat(), "end": START.isoformat()},
            {"start": START.isoformat(), "end": (START + timedelta(days=32)).isoformat()},
            {"run": "abc"},
        ]:
            assert client.get("/api/stats", params=params).status_code == 422
