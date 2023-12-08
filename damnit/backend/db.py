import os
import logging
import sqlite3
from collections.abc import MutableMapping, ValuesView, ItemsView
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from secrets import token_hex
from typing import Any

from ..context import UserEditableVariable

DB_NAME = Path('runs.sqlite')

log = logging.getLogger(__name__)

V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_info(proposal, run, start_time, added_at);
CREATE UNIQUE INDEX IF NOT EXISTS proposal_run ON run_info (proposal, run);

CREATE TABLE IF NOT EXISTS run_variables(proposal, run, name, version, value, timestamp, max_diff, provenance);
CREATE UNIQUE INDEX IF NOT EXISTS variable_version ON run_variables (proposal, run, name, version);

-- These are dummy views that will be overwritten later, but they should at least
-- exist on startup.
CREATE VIEW IF NOT EXISTS runs      AS SELECT * FROM run_info;
CREATE VIEW IF NOT EXISTS max_diffs AS SELECT proposal, run FROM run_info;

CREATE TABLE IF NOT EXISTS metameta(key PRIMARY KEY NOT NULL, value);
CREATE TABLE IF NOT EXISTS variables(name TEXT PRIMARY KEY NOT NULL, type TEXT, title TEXT, description TEXT, attributes TEXT);
CREATE TABLE IF NOT EXISTS time_comments(timestamp, comment);
"""


@dataclass
class ReducedData:
    """
    Helper class for holding summaries and variable metdata.
    """
    value: Any
    max_diff: float = None


class BlobTypes(Enum):
    png = 'png'
    numpy = 'numpy'
    pickle = 'pickle'  # We try to avoid pickle, but have used it in the past
    unknown = 'unknown'

    @classmethod
    def identify(cls, blob: bytes):
        if blob.startswith(b'\x89PNG\r\n\x1a\x0a'):
            return cls.png
        elif blob.startswith(b'\x93NUMPY'):
            return cls.numpy
        elif blob.startswith(b'\x80'):
            # Since pickle v2 (Python 2.3), all pickles start with a protocol
            # version opcode (0x80).
            return cls.pickle

        return cls.unknown


def db_path(root_path: Path):
    return root_path / DB_NAME

DATA_FORMAT_VERSION = 1

class DamnitDB:
    def __init__(self, path=DB_NAME):
        db_existed = path.exists()
        log.debug("Opening database at %s", path)
        self.conn = sqlite3.connect(path, timeout=30)
        # Ensure the database is writable by everyone
        if os.stat(path).st_uid == os.getuid():
            os.chmod(path, 0o666)

        self.conn.row_factory = sqlite3.Row
        self.metameta = MetametaMapping(self.conn)

        # Only execute the schema if we wouldn't overwrite a previous version
        can_apply_schema = True
        if db_existed:
            data_format_version = self.metameta.get("data_format_version", 0)
            if data_format_version < DATA_FORMAT_VERSION:
                can_apply_schema = False

        if can_apply_schema:
            self.conn.executescript(V1_SCHEMA)

        # A random ID for the update topic
        if 'db_id' not in self.metameta:
            # The ID is not a secret and doesn't need to be cryptographically
            # secure, but the secrets module is convenient to get a random string.
            self.metameta.setdefault('db_id', token_hex(20))

        if not db_existed:
            # If this is a new database, set the latest current version
            self.metameta["data_format_version"] = DATA_FORMAT_VERSION
        elif db_existed and "data_format_version" not in self.metameta:
            # Otherwise this is a legacy database
            self.metameta["data_format_version"] = 0

    @classmethod
    def from_dir(cls, path):
        return cls(Path(path, DB_NAME))

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

    def ensure_run(self, proposal: int, run: int, added_at: float=None, start_time: float=None):
        if added_at is None:
            added_at = datetime.now(tz=timezone.utc).timestamp()

        with self.conn:
            self.conn.execute("""
                INSERT INTO run_info (proposal, run, start_time, added_at) VALUES (?, ?, ?, ?)
                ON CONFLICT (proposal, run) DO NOTHING
            """, (proposal, run, start_time, added_at))

            # We handle the start_time specially because it may be set after the
            # run has been created in the database.
            if start_time is not None:
                self.conn.execute("""
                UPDATE run_info
                SET start_time = ?
                WHERE proposal = ? AND run = ?
                """, (start_time, proposal, run))

    def change_run_comment(self, proposal: int, run: int, comment: str):
        self.set_variable(proposal, run, "comment", ReducedData(comment))

    def add_user_variable(self, variable: UserEditableVariable, exist_ok=False):
        v = variable
        with self.conn:
            or_replace = ' OR REPLACE' if exist_ok else ''
            self.conn.execute(
                f"INSERT{or_replace} INTO variables (name, type, title, description) VALUES(?, ?, ?, ?)",
                (v.name, v.variable_type, v.title, v.description)
            )

        self.update_views()

    def variable_names(self):
        names = { record[0] for record in
                  self.conn.execute("SELECT DISTINCT name FROM run_variables").fetchall() }

        # It could be that a user-editable variable was created but hasn't been
        # assigned yet, which means an entry for it won't have been created in
        # the run_variables table. Hence we look in the variables table as well.
        names |= { record[0] for record in
                   self.conn.execute("SELECT name FROM variables").fetchall() }

        return list(names)

    def update_views(self):
        variables = self.variable_names()

        col_select_sql = "max(CASE WHEN name='{var}' THEN {col} END) AS {var}"
        runs_cols = ", ".join([col_select_sql.format(var=var, col="value")
                               for var in variables])
        max_diff_cols = ", ".join([col_select_sql.format(var=var, col="max_diff")
                               for var in variables])

        self.conn.executescript(f"""
            DROP VIEW IF EXISTS runs;
            CREATE VIEW runs
            AS SELECT run_info.proposal, run_info.run, start_time, added_at, {runs_cols}
               FROM run_variables INNER JOIN run_info ON run_variables.proposal = run_info.proposal AND run_variables.run = run_info.run
               GROUP BY run_info.run;

            DROP VIEW IF EXISTS max_diffs;
            CREATE VIEW max_diffs
            AS SELECT proposal, run, {max_diff_cols}
               FROM run_variables
               GROUP BY run;
        """)

    def set_variable(self, proposal: int, run: int, name: str, reduced):
        timestamp = datetime.now(tz=timezone.utc).timestamp()

        variable = asdict(reduced)

        # If the value is None that implies that the variable should be
        # 'deleted', in which case we don't actually delete the row, but rather
        # set the value and all metadata fields in the database to NULL.
        if variable["value"] is None:
            for key in variable:
                variable[key] = None

        variable["proposal"] = proposal
        variable["run"] = run
        variable["name"] = name
        variable["timestamp"] = timestamp
        variable["provenance"] = "context.py"

        # TODO: enable these code snippets when we add support for versioning
        # latest_version = self.conn.execute("""
        #     SELECT max(version) FROM run_variables
        #     WHERE proposal=? AND run=? AND name=?
        # """, (proposal, run, name)).fetchone()[0]
        variable["version"] = 1 # if latest_version is None else latest_version + 1

        # These columns should match those in the run_variables table
        cols = ["proposal", "run", "name", "version", "value", "timestamp", "max_diff", "provenance"]
        col_list = ", ".join(cols)
        col_values = ", ".join([f":{col}" for col in cols])
        col_updates = ", ".join([f"{col} = :{col}" for col in cols])

        with self.conn:
            existing_variables = self.variable_names()
            is_new = name not in existing_variables

            self.conn.execute(f"""
                INSERT INTO run_variables ({col_list})
                VALUES ({col_values})
                ON CONFLICT (proposal, run, name, version) DO UPDATE SET {col_updates}
            """, variable)

            if is_new:
                self.update_views()

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


# Old schemas for reference and migration

V0_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(proposal, runnr, start_time, added_at, comment);
CREATE UNIQUE INDEX IF NOT EXISTS proposal_run ON runs (proposal, runnr);
CREATE TABLE IF NOT EXISTS metameta(key PRIMARY KEY NOT NULL, value);
CREATE TABLE IF NOT EXISTS variables(name TEXT PRIMARY KEY NOT NULL, type TEXT, title TEXT, description TEXT, attributes TEXT);
CREATE TABLE IF NOT EXISTS time_comments(timestamp, comment);
"""

if __name__ == '__main__':
    DamnitDB()
