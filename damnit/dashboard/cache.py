"""Cache parsed user-journal observations and the intervals successfully scanned."""
import json
import logging
import os
import socket
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .journal import JOURNAL_TIMEOUT_SECONDS, preceding_combiner_records, read_journal
from .stats import parse_record, summarize_events


log = logging.getLogger(__name__)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MICROSECOND = timedelta(microseconds=1)
LOOKBACK = timedelta(minutes=1)
JOURNAL_LAG = timedelta(seconds=2)
APPLICATION_ID = 0x44414D4E  # "DAMN"
CACHE_VERSION = 2  # Bump when the schema or the parsed observations change.


class CacheError(Exception):
    """The dashboard cache could not be accessed."""


def cache_path():
    override = os.environ.get("DAMNIT_DASHBOARD_CACHE")
    if override:
        path = Path(override).expanduser()
    else:
        root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        # Service accounts can have a shared home directory on multiple hosts.
        path = Path(root) / "damnit" / socket.gethostname() / "dashboard.sqlite3"
    # Fresh installations use the documented path. If it already holds an
    # older dashboard cache, build alongside it so rollback retains history.
    if path.is_file():
        with closing(sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)) as db:
            if (db.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
                    and 0 < db.execute("PRAGMA user_version").fetchone()[0] < CACHE_VERSION):
                return path.with_name(f"{path.stem}.v{CACHE_VERSION}{path.suffix}")
    return path


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
        raise CacheError("The configured versioned cache has an incompatible schema.")
    db.execute("CREATE TABLE events (ts INTEGER NOT NULL, payload TEXT NOT NULL)")
    db.execute("CREATE INDEX events_by_time ON events (ts)")
    db.execute("CREATE TABLE coverage (start INTEGER PRIMARY KEY, end INTEGER NOT NULL)")
    db.execute(f"PRAGMA user_version={CACHE_VERSION}")


# Only a fragment start can own the following unkeyed legacy file completion.
# Interleaved Kafka warnings and journal messages do not end the attempt.
FILE_BOUNDARIES = {"fragment", "merge", "rename", "skip", "update", "commit", "rollback", "processing_error", "reset"}


def _resolve_predecessors(db, before, unresolved):
    """Resolve only the first unkeyed completion per worker at a scan boundary."""
    def resolve(event):
        source = tuple(event["source"])
        if source in unresolved and event["service"] == "combiner" and event["kind"] in FILE_BOUNDARIES:
            rowid, completion = unresolved.pop(source)
            if event["kind"] == "fragment":
                completion["key"] = event["key"]
                db.execute("UPDATE events SET payload = ? WHERE rowid = ?",
                           (json.dumps(completion, separators=(",", ":")), rowid))

    # Reuse only contiguous observations. Across a gap, an older start may
    # belong to an unrelated attempt. One reverse pass serves all workers.
    coverage = db.execute("SELECT start FROM coverage WHERE start < ? AND end >= ?", (before, before)).fetchone()
    if coverage:
        left, = coverage
        rows = db.execute("SELECT payload FROM events WHERE ts >= ? AND ts < ? ORDER BY ts DESC, rowid DESC",
                          (left, before))
        for (payload,) in rows:
            resolve(json.loads(payload))
            if not unresolved:
                return
        before = left
    with closing(preceding_combiner_records(EPOCH + before * MICROSECOND, unresolved)) as records:
        for record in records:
            if parsed := parse_record(record):
                resolve(parsed[1])
            if not unresolved:
                return


def _fill_gap(db, first, last):
    active = {}
    unresolved = {}
    with closing(read_journal(EPOCH + first * MICROSECOND, EPOCH + last * MICROSECOND)) as records:
        for record in records:
            parsed = parse_record(record)
            if parsed is None:
                continue
            stamp, event = parsed
            if not first <= stamp < last:
                continue
            orphan = False
            if event["service"] == "combiner" and event["kind"] in FILE_BOUNDARIES:
                source = tuple(event["source"])
                if event["kind"] == "merge" and event.get("key") is None:
                    orphan = source not in active and any(source[1:])
                    if any(source[1:]):
                        event["key"] = active.get(source)
                active[source] = event.get("key") if event["kind"] == "fragment" else None
            row = db.execute("INSERT INTO events (ts, payload) VALUES (?, ?)",
                             (stamp, json.dumps(event, separators=(",", ":"))))
            if orphan:
                unresolved[source] = (row.lastrowid, event)
    if unresolved:
        _resolve_predecessors(db, first, unresolved)


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


def cached_stats(start, end, proposal=None, run=None, *, service=None, now=None, pending_threshold_seconds=60):
    """Summarize cached observations, fetching only missing journal intervals."""
    now = now if now is not None else datetime.now(timezone.utc)
    through = min(end, now - JOURNAL_LAG)
    first = max(0, (start - EPOCH - LOOKBACK) // MICROSECOND)
    last = (through - EPOCH) // MICROSECOND
    if start >= through:
        result = summarize_events((), start, end, proposal, run, service=service,
                                  pending_threshold_seconds=pending_threshold_seconds)
    else:
        try:
            path = cache_path()
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with closing(sqlite3.connect(path, timeout=JOURNAL_TIMEOUT_SECONDS + 10)) as db:
                _cover(db, first, last)
                rows = db.execute(
                    "SELECT ts, payload FROM events WHERE ts >= ? AND ts < ? ORDER BY ts, rowid",
                    (first, last),
                )
                events = ((stamp, json.loads(payload)) for stamp, payload in rows)
                result = summarize_events(events, start, end, proposal, run, service=service,
                                          pending_threshold_seconds=pending_threshold_seconds)
        except (sqlite3.Error, OSError) as exc:
            log.exception("Could not use dashboard cache")
            raise CacheError("Could not access the dashboard statistics cache.") from exc
    result["data_through"] = through.isoformat()
    return result
