import json
import logging
import os
import shutil
import sqlite3
import sys
import struct
from collections.abc import ItemsView, MutableMapping, ValuesView
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from secrets import token_hex
from textwrap import dedent
from typing import Any, Optional

from ..definitions import UPDATE_TOPIC
from .user_variables import UserEditableVariable

DB_NAME = Path('runs.sqlite')

log = logging.getLogger(__name__)

V4_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_info(proposal, run, start_time, added_at);
CREATE UNIQUE INDEX IF NOT EXISTS proposal_run ON run_info (proposal, run);

-- attributes column is new in v2
CREATE TABLE IF NOT EXISTS run_variables(proposal, run, name, version, value, timestamp, max_diff, provenance, summary_type, summary_method, attributes);
CREATE UNIQUE INDEX IF NOT EXISTS variable_version ON run_variables (proposal, run, name, version);

-- These are dummy views that will be overwritten later, but they should at least
-- exist on startup.
CREATE VIEW IF NOT EXISTS runs      AS SELECT * FROM run_info;
CREATE VIEW IF NOT EXISTS max_diffs AS SELECT proposal, run FROM run_info;

CREATE TABLE IF NOT EXISTS metameta(key PRIMARY KEY NOT NULL, value);
CREATE TABLE IF NOT EXISTS variables(name TEXT PRIMARY KEY NOT NULL, type TEXT, title TEXT, description TEXT, attributes TEXT);
CREATE TABLE IF NOT EXISTS time_comments(timestamp, comment);

-- Tags related tables
CREATE TABLE IF NOT EXISTS tags(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS variable_tags(
    variable_name TEXT NOT NULL,
    tag_id INTEGER NOT NULL,
    FOREIGN KEY (variable_name) REFERENCES variables(name) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (variable_name, tag_id)
);

-- Trigger to remove an orphaned tag after its last reference is deleted from variable_tags
CREATE TRIGGER IF NOT EXISTS delete_orphan_tags_after_variable_tag_delete
AFTER DELETE ON variable_tags
FOR EACH ROW
BEGIN
    -- Check if the tag_id from the deleted row (OLD.tag_id)
    -- no longer exists in any other row in variable_tags
    DELETE FROM tags
    WHERE id = OLD.tag_id
    AND NOT EXISTS (
        SELECT 1 FROM variable_tags
        WHERE tag_id = OLD.tag_id
    );
END;
"""


class SummaryType(Enum):
    # We record summary object types only where it's not clear from the value:
    # numbers, strings, thumbnails (PNG) don't need a marker.

    # from datetime, stored as seconds since epoch, displayed in local time
    timestamp = "timestamp"


@dataclass
class ReducedData:
    """
    Helper class for holding summaries and variable metdata.
    """
    value: Any
    max_diff: float = None
    summary_method: str = ''
    summary_type: Optional[str] = None
    attributes: Optional[dict] = None


class BlobTypes(Enum):
    png = 'png'
    numpy = 'numpy'
    unknown = 'unknown'

    @classmethod
    def identify(cls, blob: bytes):
        if blob.startswith(b'\x89PNG\r\n\x1a\n'):
            return cls.png
        elif blob.startswith(b'\x93NUMPY'):
            return cls.numpy

        return cls.unknown


def complex2blob(data: complex) -> bytes:
    # convert complex to bytes
    return struct.pack('<dd', data.real, data.imag)


def blob2complex(data: bytes) -> complex:
    # convert bytes to complex
    real, imag = struct.unpack('<dd', data)
    return complex(real, imag)


def db_path(root_path: Path):
    return root_path / DB_NAME

DATA_FORMAT_VERSION = 4
MIN_OPENABLE_VERSION = 1  # DBs from this version will be upgraded on opening

class DamnitDB:
    def __init__(self, path=DB_NAME, allow_old=False):
        self._path = path.absolute()

        db_existed = path.exists()
        log.debug("Opening database at %s", path)
        self.conn = sqlite3.connect(path, timeout=30)
        # Ensure the database is writable by everyone
        if os.stat(path).st_uid == os.getuid():
            os.chmod(path, 0o666)
        # Enable foreign key enforcement for this connection
        self.conn.execute("PRAGMA foreign_keys = ON;") 

        self.conn.row_factory = sqlite3.Row
        self.metameta = KeyValueMapping(self.conn, "metameta")

        # Only execute the schema if we wouldn't overwrite a previous version
        can_apply_schema = True
        if db_existed:
            data_format_version = self.metameta.get("data_format_version", 0)
            if data_format_version < DATA_FORMAT_VERSION:
                can_apply_schema = False
        else:
            data_format_version = DATA_FORMAT_VERSION

        if can_apply_schema:
            self.conn.executescript(V4_SCHEMA)

        # A random ID for the update topic
        if 'db_id' not in self.metameta:
            # The ID is not a secret and doesn't need to be cryptographically
            # secure, but the secrets module is convenient to get a random string.
            self.metameta.setdefault('db_id', token_hex(20))

        if not db_existed:
            # If this is a new database, set the latest current version
            self.metameta["data_format_version"] = DATA_FORMAT_VERSION
            # Use the Python environment the database was created under by default
            self.metameta["damnit_python"] = sys.executable
            self.metameta["concurrent_jobs"] = 15

        if not allow_old:
            if data_format_version < MIN_OPENABLE_VERSION:
                raise RuntimeError(
                    f"Cannot open older (v{data_format_version}) database, please contact DA "
                    "for help migrating"
                )
            elif data_format_version < DATA_FORMAT_VERSION:
                self.upgrade_schema(data_format_version)

    @classmethod
    def from_dir(cls, path):
        return cls(Path(path, DB_NAME))

    def close(self):
        self.conn.close()

    @property
    def kafka_topic(self):
        return UPDATE_TOPIC.format(self.metameta['db_id'])

    @property
    def path(self):
        return self._path

    def upgrade_schema(self, from_version):
        log.info("Upgrading database format from v%d to v%d",
                 from_version, DATA_FORMAT_VERSION)
        with self.conn:
            if from_version < 2:
                self.conn.execute("ALTER TABLE run_variables ADD COLUMN attributes")
                self.conn.execute("UPDATE metameta SET value=? WHERE key='data_format_version'", (2,))
                self.conn.commit()

            if from_version < 3:
                self.conn.executescript(dedent("""\
                    -- Tags related tables
                    CREATE TABLE IF NOT EXISTS tags(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS variable_tags(
                        variable_name TEXT NOT NULL,
                        tag_id INTEGER NOT NULL,
                        FOREIGN KEY (variable_name) REFERENCES variables(name) ON DELETE CASCADE,
                        FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                        PRIMARY KEY (variable_name, tag_id)
                    );"""))
                self.conn.execute("UPDATE metameta SET value=? WHERE key='data_format_version'", (3,))
                self.conn.commit()

            if from_version < 4:
                self.conn.execute(dedent("""\
                    -- Trigger to remove an orphaned tag after its last reference is deleted from variable_tags
                    CREATE TRIGGER IF NOT EXISTS delete_orphan_tags_after_variable_tag_delete
                    AFTER DELETE ON variable_tags
                    FOR EACH ROW
                    BEGIN
                        -- Check if the tag_id from the deleted row (OLD.tag_id)
                        -- no longer exists in any other row in variable_tags
                        DELETE FROM tags
                        WHERE id = OLD.tag_id
                        AND NOT EXISTS (
                            SELECT 1 FROM variable_tags
                            WHERE tag_id = OLD.tag_id
                        );
                    END;"""))
                self.conn.execute("UPDATE metameta SET value=? WHERE key='data_format_version'", (4,))
                self.conn.commit()

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

    def get_user_variables(self):
        user_variables = {}
        rows = self.conn.execute("""
            SELECT name, title, type, description, attributes FROM variables
            WHERE type IS NOT NULL
        """)
        for rr in rows:
            var_name = rr["name"]
            new_var = UserEditableVariable(
                var_name,
                title=rr["title"],
                variable_type=rr["type"],
                description=rr["description"],
                attributes=rr["attributes"],
            )
            user_variables[var_name] = new_var
        log.debug("Loaded %d user variables", len(user_variables))
        return user_variables

    def update_computed_variables(self, vars: dict):
        vars_in_db = {}
        with self.conn:
            # We want to read & write in the same transaction. This gets the
            # write lock up front, to prevent deadlocks where two processes are
            # both holding a read lock & waiting for a write lock.
            self.conn.execute("BEGIN IMMEDIATE")
            for row in self.conn.execute("""
                SELECT name, title, type, description, attributes FROM variables
                WHERE type IS NULL
            """):
                var = dict(row)
                vars_in_db[var.pop("name")] = var

            updates = {n: v for (n, v) in vars.items()
                       if v != vars_in_db.get(n, None)}
            log.debug("Updating stored metadata for %d computed variables",
                      len(updates))

            # Write new & changed variables
            # TODO: what if a new computed variable name matches an existing user var?
            self.conn.executemany("""
                INSERT INTO variables VALUES (?, NULL, ?, ?, ?)
                ON CONFLICT (name) DO UPDATE SET
                    type=excluded.type,
                    title=excluded.title,
                    description=excluded.description,
                    attributes=excluded.attributes
            """, [
                (n, v['title'], v['description'], v['attributes'])
                for (n, v) in updates.items()
            ])

            for name, var in updates.items():
                existing_tags = set(self.get_variable_tags(name))
                new_tags = set(var.get('tags', None) or [])

                tags_to_add = new_tags - existing_tags
                tags_to_remove = existing_tags - new_tags

                for tag in tags_to_add:
                    self.tag_variable(name, tag)

                for tag in tags_to_remove:
                    self.untag_variable(name, tag)

            if not set(vars) <= set(vars_in_db):
                # At least 1 variable was new, so remake the views with the new columns
                self.update_views()

        return updates

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

        with self.conn:
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
        elif isinstance(variable["value"], complex):
            variable["value"] = complex2blob(variable["value"])
            variable["summary_type"] = "complex"

        variable["proposal"] = proposal
        variable["run"] = run
        variable["name"] = name
        variable["timestamp"] = timestamp
        variable["provenance"] = "context.py"
        variable["attributes"] = None
        if reduced.attributes:
            variable["attributes"] = json.dumps(reduced.attributes)

        # TODO: enable these code snippets when we add support for versioning
        # latest_version = self.conn.execute("""
        #     SELECT max(version) FROM run_variables
        #     WHERE proposal=? AND run=? AND name=?
        # """, (proposal, run, name)).fetchone()[0]
        variable["version"] = 1 # if latest_version is None else latest_version + 1

        # These columns should match those in the run_variables table
        cols = ["proposal", "run", "name", "version", "value", "timestamp", "max_diff", "provenance", "summary_method", "summary_type", "attributes"]
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

    def delete_variable(self, name: str):
        with self.conn:
            # First delete from the `variables` table
            self.conn.execute("""
            DELETE FROM variables
            WHERE name = ?
            """, (name,))

            # And then `run_variables`
            self.conn.execute("""
            DELETE FROM run_variables
            WHERE name = ?
            """, (name, ))

            self.update_views()

    def add_tag(self, tag_name: str) -> int:
        """Add a new tag to the database if it doesn't exist.
        Returns the tag ID."""
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
            cursor = self.conn.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
            return cursor.fetchone()[0]

    def get_tag_id(self, tag_name: str) -> Optional[int]:
        """Get the ID of a tag by its name."""
        with self.conn:
            cursor = self.conn.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
            result = cursor.fetchone()
            return result[0] if result else None

    def tag_variable(self, variable_name: str, tag_name: str):
        """Associate a tag with a variable."""
        tag_id = self.add_tag(tag_name)
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO variable_tags (variable_name, tag_id) VALUES (?, ?)",
                (variable_name, tag_id)
            )

    def untag_variable(self, variable_name: str, tag_name: str):
        """Remove a tag association from a variable."""
        tag_id = self.get_tag_id(tag_name)
        if tag_id is not None:
            with self.conn:
                self.conn.execute(
                    "DELETE FROM variable_tags WHERE variable_name = ? AND tag_id = ?",
                    (variable_name, tag_id)
                )

    def get_variable_tags(self, variable_name: str) -> list[str]:
        """Get all tags associated with a variable."""
        with self.conn:
            cursor = self.conn.execute("""
                SELECT t.name
                FROM tags t
                JOIN variable_tags vt ON t.id = vt.tag_id
                WHERE vt.variable_name = ?
            """, (variable_name,))
            return [row[0] for row in cursor.fetchall()]

    def get_variables_by_tag(self, tag_name: str) -> list[str]:
        """Get all variables that have a specific tag."""
        with self.conn:
            cursor = self.conn.execute("""
                SELECT vt.variable_name
                FROM variable_tags vt
                JOIN tags t ON vt.tag_id = t.id
                WHERE t.name = ?
            """, (tag_name,))
            return [row[0] for row in cursor.fetchall()]

    def get_all_tags(self) -> list[str]:
        """Get all existing tags."""
        with self.conn:
            cursor = self.conn.execute("SELECT name FROM tags ORDER BY name")
            return [row[0] for row in cursor.fetchall()]


class KeyValueMapping(MutableMapping):
    """
    Simple class that represents a dictionary backed by a SQLite table.

    Note that the `table` argument is assumed to come from a trusted source, it
    isn't quoted into the internal SQL expressions.
    """

    def __init__(self, conn, table):
        self.conn = conn
        self.table = table

    def __getitem__(self, key):
        row = self.conn.execute(
            f"SELECT value FROM {self.table} WHERE key=?", (key,)
        ).fetchone()
        if row is not None:
            return row[0]
        raise KeyError

    def __setitem__(self, key, value):
        with self.conn:
            self.conn.execute(
                f"INSERT INTO {self.table} VALUES (:key, :value)"
                "ON CONFLICT (key) DO UPDATE SET value=:value",
                {'key': key, 'value': value}
            )

    def update(self, other=(), **kwargs):
        # Override to do the update in one transaction
        d = {}
        d.update(other, **kwargs)

        with self.conn:
            self.conn.executemany(
                f"INSERT INTO {self.table} VALUES (:key, :value)"
                "ON CONFLICT (key) DO UPDATE SET value=:value",
                [{'key': k, 'value': v} for (k, v) in d.items()]
            )

    def __delitem__(self, key):
        with self.conn:
            c = self.conn.execute(f"DELETE FROM {self.table} WHERE key=?", (key,))
            if c.rowcount == 0:
                raise KeyError(key)

    def __iter__(self):
        return (r[0] for r in self.conn.execute(f"SELECT key FROM {self.table}"))

    def __len__(self):
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def setdefault(self, key, default=None):
        with self.conn:
            try:
                self.conn.execute(
                    f"INSERT INTO {self.table} VALUES (:key, :value)",
                    {'key': key, 'value': default}
                )
                value = default
            except sqlite3.IntegrityError:
                # The key is already present
                value = self[key]

        return value

    def to_dict(self):
        return dict(self.conn.execute(f"SELECT * FROM {self.table}"))

    # Reimplement .values() and .items() to use just one query.
    def values(self):
        return ValuesView(self.to_dict())

    def items(self):
        return ItemsView(self.to_dict())


# Messages to notify clients about database changes

class MsgKind(Enum):
    # We don't distinguish added vs. changed, because we have unique IDs for the
    # objects, so recipients can easily tell if an object is new to them.
    # This also means messages are idempotent.
    variable_set = 'variable_set'
    #variable_deleted = 'variable_deleted'
    run_values_updated = 'run_values_updated'
    #run_deleted = 'run_deleted'
    #standalone_comment_set = 'standalone_comment_set'
    #standalone_comment_deleted = 'standalone_comment_deleted'
    processing_state_set = 'processing_state_set'   # Supports status indicators
    processing_finished = 'processing_finished'
    # Commented out options are not implemented yet

def msg_dict(kind: MsgKind, data: dict):
    return {'msg_kind': kind.value, 'data': data}

def initialize_proposal(root_path, proposal=None, context_file_src=None, user_vars_src=None):
    # Ensure the directory exists
    root_path.mkdir(parents=True, exist_ok=True)
    if root_path.stat().st_uid == os.getuid():
        os.chmod(root_path, 0o777)

    # If the database doesn't exist, create it
    if new_db := not db_path(root_path).is_file():
        if proposal is None:
            raise ValueError("Must pass a proposal number to `initialize_proposal()` if the database doesn't exist yet.")

        # Initialize database
        db = DamnitDB.from_dir(root_path)
        db.metameta["proposal"] = proposal
        db.metameta["context_python"] = "/gpfs/exfel/sw/software/euxfel-environment-management/environments/202502/.pixi/envs/default/bin/python"
    else:
        # Otherwise, load the proposal number
        db = DamnitDB.from_dir(root_path)
        proposal = db.metameta["proposal"]

    context_path = root_path / "context.py"
    # Copy initial context file if necessary
    if not context_path.is_file():
        if context_file_src is not None:
            shutil.copyfile(context_file_src, context_path)
        else:
            context_path.touch()
        os.chmod(context_path, 0o666)

    # Copy user editable variables if requested
    if new_db and (user_vars_src is not None):
        prev_db = DamnitDB(user_vars_src)
        for var in prev_db.get_user_variables().values():
            db.add_user_variable(var)

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
