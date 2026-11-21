"""Cache parsed user-journal observations and the intervals successfully scanned."""
import json
import logging
import os
import socket
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .journal import JOURNAL_TIMEOUT_SECONDS, read_journal
from .stats import parse_record, summarize_events


log = logging.getLogger(__name__)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MICROSECOND = timedelta(microseconds=1)
LOOKBACK = timedelta(minutes=1)
JOURNAL_LAG = timedelta(seconds=2)
APPLICATION_ID = 0x44414D4E  # "DAMN"
CACHE_VERSION = 1  # Bump when the schema or the parsed observations change.


class CacheError(Exception):
    """The dashboard cache could not be accessed."""


def cache_path():
    override = os.environ.get("DAMNIT_DASHBOARD_CACHE")
    if override:
        return Path(override).expanduser()
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    # Service accounts can have a shared home directory on multiple hosts.
    return Path(root) / "damnit" / socket.gethostname() / "dashboard.sqlite3"


def _initialize(db):
    app_id = db.execute("PRAGMA application_id").fetchone()[0]
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if app_id == APPLICATION_ID and version == CACHE_VERSION:
        return
    if app_id == 0 and not db.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone():
        db.execute(f"PRAGMA application_id={APPLICATION_ID}")
    elif app_id != APPLICATION_ID:
        raise CacheError("The configured cache file contains another application's database.")
    elif version > CACHE_VERSION:
        raise CacheError("The cache was written by a newer dashboard version.")
    else:
        db.execute("DROP TABLE IF EXISTS events")
        db.execute("DROP TABLE IF EXISTS coverage")
    db.execute("CREATE TABLE events (ts INTEGER NOT NULL, payload TEXT NOT NULL)")
    db.execute("CREATE INDEX events_by_time ON events (ts)")
    db.execute("CREATE TABLE coverage (start INTEGER PRIMARY KEY, end INTEGER NOT NULL)")
    db.execute(f"PRAGMA user_version={CACHE_VERSION}")


def _fill_gap(db, first, last):
    with closing(read_journal(EPOCH + first * MICROSECOND, EPOCH + last * MICROSECOND)) as records:
        def parsed():
            for record in records:
                event = parse_record(record)
                if event is not None:
                    stamp, payload = event
                    if first <= stamp < last:
                        yield stamp, json.dumps(payload, separators=(",", ":"))

        db.executemany("INSERT INTO events (ts, payload) VALUES (?, ?)", parsed())


def _cover(db, first, last):
    # A hit can proceed while another request is filling a different interval.
    if (db.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
            and db.execute("PRAGMA user_version").fetchone()[0] == CACHE_VERSION
            and db.execute("SELECT 1 FROM coverage WHERE start <= ? AND end >= ?",
                           (first, last)).fetchone()):
        return
    with db:
        db.execute("BEGIN IMMEDIATE")
        _initialize(db)
        # Recheck after acquiring the writer lock. Include adjoining intervals
        # so that incremental refreshes coalesce into a single coverage row.
        overlap = "WHERE end >= ? AND start <= ?"
        rows = db.execute(f"SELECT start, end FROM coverage {overlap} ORDER BY start",
                          (first, last)).fetchall()
        cursor = first
        for left, right in rows:
            if left <= first and right >= last:
                return
            if cursor < left:
                _fill_gap(db, cursor, left)
            cursor = max(cursor, right)
        if cursor < last:
            _fill_gap(db, cursor, last)
        db.execute(f"DELETE FROM coverage {overlap}", (first, last))
        db.execute("INSERT INTO coverage VALUES (?, ?)", (
            min(first, rows[0][0]) if rows else first,
            max(last, rows[-1][1]) if rows else last,
        ))


def cached_stats(start, end, proposal=None, run=None, *, now=None):
    """Summarize cached observations, fetching only missing journal intervals."""
    now = now if now is not None else datetime.now(timezone.utc)
    through = min(end, now - JOURNAL_LAG)
    first = max(0, (start - EPOCH - LOOKBACK) // MICROSECOND)
    last = (through - EPOCH) // MICROSECOND
    if start >= through:
        result = summarize_events((), start, end, proposal, run)
    else:
        path = cache_path()
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with closing(sqlite3.connect(path, timeout=JOURNAL_TIMEOUT_SECONDS + 10)) as db:
                _cover(db, first, last)
                rows = db.execute(
                    "SELECT ts, payload FROM events WHERE ts >= ? AND ts < ? ORDER BY ts, rowid",
                    (first, last),
                )
                events = ((stamp, json.loads(payload)) for stamp, payload in rows)
                result = summarize_events(events, start, end, proposal, run)
        except (sqlite3.Error, OSError) as exc:
            log.exception("Could not use dashboard cache %s", path)
            raise CacheError("Could not access the dashboard statistics cache.") from exc
    result["data_through"] = through.isoformat()
    return result
