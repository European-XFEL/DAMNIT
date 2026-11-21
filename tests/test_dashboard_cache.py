import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event

import pytest

from damnit.dashboard import cache
from damnit.dashboard.journal import JournalError


START = datetime(2024, 10, 1, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
SECOND = timedelta(seconds=1)
LAUNCH = "Launched Slurm (maxwell) job 123 to run context file"


def listener(time, message):
    return {
        "__REALTIME_TIMESTAMP": str((time - cache.EPOCH) // cache.MICROSECOND),
        "_SYSTEMD_USER_UNIT": "damnit-listener.service",
        "_SYSTEMD_INVOCATION_ID": "listener1",
        "MESSAGE": message,
    }


@pytest.fixture
def journal(monkeypatch, tmp_path):
    records = []
    reads = []

    def read(start, end):
        reads.append((start, end))
        for time, message in records:
            if start <= time < end:
                yield listener(time, message)

    monkeypatch.setenv("DAMNIT_DASHBOARD_CACHE", str(tmp_path / "cache.sqlite3"))
    monkeypatch.setattr(cache, "read_journal", read)
    return records, reads


def test_cached_stats_survive_restart_and_share_filters(journal):
    records, reads = journal
    records.extend([(START - SECOND, "Added p42 r1 (raw data) to database"), (START, LAUNCH)])
    end = START + HOUR
    assert cache.cached_stats(START, end, 42)["summary"]["raw_submissions"] == 1
    assert cache.cached_stats(START, end, 17)["summary"]["submissions"] == 0
    assert reads == [(START - cache.LOOKBACK, end)]
    # A fresh interpreter sees the same parsed data without journal access.
    script = """
import sys
from datetime import datetime
from unittest.mock import patch
from damnit.dashboard.cache import cached_stats

start, end = map(datetime.fromisoformat, sys.argv[1:])
with patch("damnit.dashboard.cache.read_journal", side_effect=AssertionError):
    assert cached_stats(start, end)["summary"]["submissions"] == 1
"""
    subprocess.run([sys.executable, "-c", script, START.isoformat(), end.isoformat()], check=True)


def test_fills_only_missing_intervals_including_empty_history(journal):
    records, reads = journal
    records.extend((START + i * HOUR, LAUNCH) for i in range(1, 7))
    for begin, stop in [(2, 3), (5, 6), (1, 7), (0, 1), (0, 7)]:
        result = cache.cached_stats(START + begin * HOUR, START + stop * HOUR)
        assert result["summary"]["submissions"] == sum(begin <= i < stop for i in range(1, 7))
    lookback = cache.LOOKBACK
    assert reads == [(START + low, START + high) for low, high in [
        (2 * HOUR - lookback, 3 * HOUR), (5 * HOUR - lookback, 6 * HOUR),
        (HOUR - lookback, 2 * HOUR - lookback), (3 * HOUR, 5 * HOUR - lookback),
        (6 * HOUR, 7 * HOUR), (-lookback, HOUR - lookback),
    ]]


def test_attribution_uses_preceding_context_across_cache_boundary(journal):
    records, _ = journal
    records.extend([
        (START - SECOND, LAUNCH),
        (START - SECOND, "Added p42 r1 (raw data) to database"),
        (START, LAUNCH),
        (START, "Added p42 r2 (proc data) to database"),
        (START, LAUNCH),
    ])
    # The query ending at START picks up context; the second query fills forward.
    cache.cached_stats(START - HOUR, START)
    result = cache.cached_stats(START, START + SECOND)
    assert result["summary"]["submissions"] == 2
    assert [(row["run"], row["raw_submissions"], row["proc_submissions"])
            for row in result["runs"]] == [(1, 1, 0), (2, 0, 1)]


def test_failed_scan_rolls_back_observations_and_coverage(journal, monkeypatch):
    records, reads = journal
    records.append((START, LAUNCH))

    def broken(start, end):
        yield listener(START, LAUNCH)
        raise JournalError("journal unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(cache, "read_journal", broken)
        with pytest.raises(JournalError):
            cache.cached_stats(START, START + HOUR)
    assert cache.cached_stats(START, START + HOUR)["summary"]["submissions"] == 1
    assert reads == [(START - cache.LOOKBACK, START + HOUR)]


def test_concurrent_requests_scan_once(journal, monkeypatch):
    records, reads = journal
    records.append((START, LAUNCH))
    original = cache.read_journal
    scanning, contending = Event(), Event()
    connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        db = connect(*args, **kwargs)
        if scanning.is_set():
            def trace(sql):
                if sql == "BEGIN IMMEDIATE":
                    contending.set()
            db.set_trace_callback(trace)
        return db

    def slow(start, end):
        scanning.set()
        assert contending.wait(timeout=5)
        yield from original(start, end)

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    monkeypatch.setattr(cache, "read_journal", slow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.cached_stats, START, START + HOUR)
        assert scanning.wait(timeout=5)
        second = pool.submit(cache.cached_stats, START, START + HOUR)
        assert first.result() == second.result()
    assert len(reads) == 1


def test_live_refresh_advances_watermark_and_leaves_future_unscanned(journal):
    records, reads = journal
    now = START + HOUR
    end = now + HOUR
    records.extend([(now - 3 * SECOND, LAUNCH), (now - SECOND, LAUNCH)])
    assert cache.cached_stats(now + SECOND, end, now=now)["summary"]["submissions"] == 0
    assert not reads
    result = cache.cached_stats(START, end, now=now)
    assert result["data_through"] == (now - cache.JOURNAL_LAG).isoformat()
    assert result["summary"]["submissions"] == 1
    assert cache.cached_stats(START, end, now=now) == result
    later = cache.cached_stats(START, end, now=now + 5 * SECOND)
    assert later["summary"]["submissions"] == 2
    assert reads == [(START - cache.LOOKBACK, now - cache.JOURNAL_LAG),
                     (now - cache.JOURNAL_LAG, now + 5 * SECOND - cache.JOURNAL_LAG)]


def test_cache_version_and_path(journal, tmp_path, monkeypatch):
    _, reads = journal
    monkeypatch.delenv("DAMNIT_DASHBOARD_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(cache.socket, "gethostname", lambda: "services-host")
    cache.cached_stats(START, START + HOUR)
    assert cache.cache_path() == tmp_path / "damnit/services-host/dashboard.sqlite3"
    # A parser upgrade invalidates both cached observations and known-empty intervals.
    monkeypatch.setattr(cache, "CACHE_VERSION", cache.CACHE_VERSION + 1)
    cache.cached_stats(START, START + HOUR)
    assert reads == [(START - cache.LOOKBACK, START + HOUR)] * 2
    with sqlite3.connect(cache.cache_path()) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == cache.CACHE_VERSION
    monkeypatch.setattr(cache.socket, "gethostname", lambda: "another-host")
    cache.cached_stats(START, START + HOUR)
    assert len(reads) == 3
