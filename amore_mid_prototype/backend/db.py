import os
import logging
import sqlite3
from secrets import token_hex
from typing import Any

DB_NAME = 'runs.sqlite'

log = logging.getLogger(__name__)

def open_db(path=DB_NAME) -> sqlite3.Connection:
    """ Initialize the sqlite run database

    A new database is created if no pre-existing one is present. A single
    table is created: runs, which has columns:

        proposal, run, start_time, added_at, comment

    More columns may be added later (by executing ALTER TABLE runs ADD COLUMN)
    """
    log.info("Opening database at %s", path)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runs(proposal, runnr, start_time, added_at, comment)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS proposal_run ON runs (proposal, runnr)"
    )
    conn.execute(  # data about metadata - metametadata?
        "CREATE TABLE IF NOT EXISTS metameta(key PRIMARY KEY NOT NULL, value)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS time_comments(timestamp, comment)"
    )
    conn.row_factory = sqlite3.Row
    get_meta(  # A random ID for the update topic
        conn, 'db_id', set_default=True,
        # The ID is not a secret and doesn't need to be cryptographically
        # secure, but the secrets module is convenient to get a random string.
        default=lambda: token_hex(20)
    )

    # Ensure the database is writable by everyone
    os.chmod(path, 0o666)

    return conn

def get_meta(conn, key, default: Any =KeyError, set_default=False):
    with conn:
        row = conn.execute(
            "SELECT value FROM metameta WHERE key=?", (key,)
        ).fetchone()
        if row is not None:
            return row[0]
        elif default is KeyError:
            raise KeyError(f"No key {key} in database metadata")
        else:
            if callable(default):
                default = default()
            if set_default:
                conn.execute(
                    "INSERT INTO metameta VALUES (:key, :value)",
                    {'key': key, 'value': default}
                )
            return default

def set_meta(conn, key, value):
    with conn:
        conn.execute(
            "INSERT INTO metameta VALUES (:key, :value)"
            "ON CONFLICT (key) DO UPDATE SET value=:value",
            {'key': key, 'value': value}
        )

if __name__ == '__main__':
    open_db()
