import os
import logging
import sqlite3
from collections.abc import MutableMapping, ValuesView, ItemsView
from pathlib import Path
from secrets import token_hex

from ..context import Variable

DB_NAME = 'runs.sqlite'

log = logging.getLogger(__name__)

# More columns can be added to runs() table depending on the data
BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(proposal, runnr, start_time, added_at, comment);
CREATE UNIQUE INDEX IF NOT EXISTS proposal_run ON runs (proposal, runnr);
CREATE TABLE IF NOT EXISTS metameta(key PRIMARY KEY NOT NULL, value);
CREATE TABLE IF NOT EXISTS variables(name TEXT PRIMARY KEY, type TEXT, title TEXT, description TEXT, attributes TEXT);
CREATE TABLE IF NOT EXISTS time_comments(timestamp, comment);
"""


def db_path(root_path: Path):
    return root_path / DB_NAME

class DamnitDB:
    def __init__(self, path=DB_NAME):
        log.debug("Opening database at %s", path)
        self.conn = sqlite3.connect(path, timeout=30)
        # Ensure the database is writable by everyone
        os.chmod(path, 0o666)

        self.conn.executescript(BASE_SCHEMA)
        self.conn.row_factory = sqlite3.Row
        self.metameta = MetametaMapping(self.conn)

        # A random ID for the update topic
        if 'db_id' not in self.metameta:
            # The ID is not a secret and doesn't need to be cryptographically
            # secure, but the secrets module is convenient to get a random string.
            self.metameta.setdefault('db_id', token_hex(20))

    def close(self):
        self.conn.close()

    def add_standalone_comment(self, ts: float, comment: str):
        """Add a comment not associated with a specific run, return its ID."""
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO time_comments VALUES (?, ?)", (ts, comment)
            )
        return cur.lastrowid

    def change_standalone_comment(self, comment_id: int, comment: str):
        with self.conn:
            self.conn.execute(
                """UPDATE time_comments set comment=? WHERE rowid=?""",
                (comment, comment_id),
            )

    def ensure_run(self, proposal: int, run: int, added_at: float):
        with self.conn:
            self.conn.execute("""
                INSERT INTO runs (proposal, runnr, added_at) VALUES (?, ?, ?)
                ON CONFLICT (proposal, runnr) DO NOTHING
            """, (proposal, run, added_at))

    def change_run_comment(self, proposal: int, run: int, comment: str):
        with self.conn:
            self.conn.execute(
                "UPDATE runs set comment=? WHERE proposal=? AND runnr=?",
                (comment, proposal, run),
            )


def add_user_variable(conn, variable: Variable):

    conn.execute(
        "INSERT INTO variables (name, type, title, description) VALUES(?, ?, ?, ?)", 
        (
            variable.name,
            variable.variable_type,
            variable.title,
            variable.description
        )
    )

def create_user_column(conn, variable: Variable):

    num_cols = conn.execute(
        "SELECT COUNT(*) FROM PRAGMA_TABLE_INFO('runs') WHERE name=?", (variable.name,)
    ).fetchone()[0]

    if num_cols == 0:
        conn.execute(f"ALTER TABLE runs ADD COLUMN {variable.name}")


class MetametaMapping(MutableMapping):
    def __init__(self, conn):
        self.conn = conn

    def __getitem__(self, key):
        row = self.conn.execute(
            "SELECT value FROM metameta WHERE key=?", (key,)
        ).fetchone()
        if row is not None:
            return row[0]
        raise KeyError

    def __setitem__(self, key, value):
        with self.conn:
            self.conn.execute(
                "INSERT INTO metameta VALUES (:key, :value)"
                "ON CONFLICT (key) DO UPDATE SET value=:value",
                {'key': key, 'value': value}
            )

    def update(self, other=(), **kwargs):
        # Override to do the update in one transaction
        d = {}
        d.update(other, **kwargs)

        with self.conn:
            self.conn.executemany(
                "INSERT INTO metameta VALUES (:key, :value)"
                "ON CONFLICT (key) DO UPDATE SET value=:value",
                [{'key': k, 'value': v} for (k, v) in d.items()]
            )

    def __delitem__(self, key):
        with self.conn:
            c = self.conn.execute("DELETE FROM metameta WHERE key=?", (key,))
            if c.rowcount == 0:
                raise KeyError(key)

    def __iter__(self):
        return (r[0] for r in self.conn.execute("SELECT key FROM metameta"))

    def __len__(self):
        return self.conn.execute("SELECT count(*) FROM metameta").fetchone()[0]

    def setdefault(self, key, default=None):
        with self.conn:
            try:
                self.conn.execute(
                    "INSERT INTO metameta VALUES (:key, :value)",
                    {'key': key, 'value': default}
                )
                value = default
            except sqlite3.IntegrityError:
                # The key is already present
                value = self[key]

        return value

    def to_dict(self):
        return dict(self.conn.execute("SELECT * FROM metameta"))

    # Reimplement .values() and .items() to use just one query.
    def values(self):
        return ValuesView(self.to_dict())

    def items(self):
        return ItemsView(self.to_dict())


if __name__ == '__main__':
    DamnitDB()
