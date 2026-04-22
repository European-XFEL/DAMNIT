import io
import json
import logging
import os
import shutil
import struct
import sys
from collections.abc import ItemsView, MutableMapping, ValuesView
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from secrets import token_hex
from typing import Any, Optional

import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ..definitions import UPDATE_TOPIC, DEFAULT_CONTEXT_PYTHON
from .db_migrations import POSTGRES_BASELINE, POSTGRES_BASELINE_VERSION
from .user_variables import UserEditableVariable

PGPASS_NAME = Path('runs.pgpass')

log = logging.getLogger(__name__)

MIN_OPENABLE_VERSION = POSTGRES_BASELINE_VERSION  # anything older is a legacy sqlite DB; use the one-shot migrator


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


def numpy2blob(arr: np.ndarray) -> bytes:
    """Serialize a numpy array into .npy bytes for database storage."""
    buff = io.BytesIO()
    np.save(buff, arr, allow_pickle=False)
    return buff.getvalue()


def blob2numpy(data: bytes) -> np.ndarray:
    """Deserialize .npy bytes from database storage into a numpy array."""
    buff = io.BytesIO(data)
    return np.load(buff, allow_pickle=False)

def pgpass_path(root_path: Path) -> Path:
    return root_path / PGPASS_NAME


def read_pgpass(root_path: Path) -> str:
    """Parse a `runs.pgpass` file into a psycopg connection string.

    File format (one line): `host:port:dbname:user:password`. Matches the
    libpq pgpass convention so it can also be consumed by `psql` / `pg_dump`
    with PGPASSFILE pointing at the same file.
    """
    line = pgpass_path(root_path).read_text().strip().splitlines()[0]
    host, port, dbname, user, password = line.split(':', 4)
    return (
        f"host={host} port={port} dbname={dbname} user={user} password={password}"
    )


def initialize_database(conn: psycopg.Connection, proposal: int) -> None:
    """Apply POSTGRES_BASELINE and seed metameta on a freshly-created DB.

    Called by the listener's provisioning handler. Not used by DamnitDB
    itself — by the time DamnitDB connects, the DB is already set up.
    """
    with conn.transaction():
        conn.execute(POSTGRES_BASELINE)
        # Use the Python environment the database was created under by default.
        # The db_id is not a secret and doesn't need to be cryptographically
        # secure, but the secrets module is convenient to get a random string.
        seed = {
            "proposal": str(proposal),
            "data_format_version": str(POSTGRES_BASELINE_VERSION),
            "damnit_python": sys.executable,
            "context_python": DEFAULT_CONTEXT_PYTHON,
            "concurrent_jobs": "15",
            "db_id": token_hex(20),
        }
        conn.cursor().executemany(
            "INSERT INTO metameta (key, value) VALUES (%s, %s)",
            list(seed.items()),
        )


class DamnitDB:
    def __init__(self, path=PGPASS_NAME):
        """Open a connection to a per-proposal Postgres database.

        Accepts a path to a runs.pgpass file.
        """
        self._path = path.absolute()

        log.debug("Connecting to Postgres database")
        dsn = read_pgpass(Path(path))
        self.conn = psycopg.connect(dsn, row_factory=dict_row)

        self.metameta = KeyValueMapping(self.conn, "metameta")

        data_format_version = int(self.metameta["data_format_version"])
        if data_format_version < MIN_OPENABLE_VERSION:
            raise RuntimeError(
                f"Cannot open older (v{data_format_version}) database; "
                "run the sqlite→postgres migrator on the source DB first."
            )

        # A random ID for the update topic
        self._db_id = self.metameta["db_id"]

    @classmethod
    def from_dir(cls, path):
        return cls(Path(path))

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def kafka_topic(self):
        return UPDATE_TOPIC.format(self._db_id)

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def add_standalone_comment(self, ts: float, comment: str) -> int:
        """Add a comment not associated with a specific run, return its ID."""
        with self.conn.transaction():
            row = self.conn.execute(
                "INSERT INTO time_comments (timestamp, comment) VALUES (%s, %s) "
                "RETURNING id",
                (ts, comment),
            ).fetchone()
        return row["id"]

    def change_standalone_comment(self, comment_id: int, comment: str):
        with self.conn.transaction():
            self.conn.execute(
                """UPDATE time_comments SET comment = %s WHERE id = %s""",
                (comment, comment_id),
            )

    def ensure_run(self, proposal: int, run: int, added_at: float=None, start_time: float=None):
        if added_at is None:
            added_at = datetime.now(tz=timezone.utc).timestamp()

        with self.conn.transaction():
            self.conn.execute("""
                INSERT INTO run_info (proposal, run, start_time, added_at) VALUES (%s, %s, %s, %s)
                ON CONFLICT (proposal, run) DO NOTHING
            """, (proposal, run, start_time, added_at))

            # We handle the start_time specially because it may be set after the
            # run has been created in the database.
            if start_time is not None:
                self.conn.execute("""
                UPDATE run_info
                SET start_time = %s
                WHERE proposal = %s AND run = %s
                """, (start_time, proposal, run))

    def change_run_comment(self, proposal: int, run: int, comment: str):
        self.set_variable(proposal, run, "comment", ReducedData(comment), provenance="")

    def add_user_variable(self, variable: UserEditableVariable, exist_ok=False):
        v = variable
        with self.conn.transaction():
            on_conflict = (
                " ON CONFLICT (name) DO UPDATE SET "
                "type = EXCLUDED.type, title = EXCLUDED.title, "
                "description = EXCLUDED.description"
                if exist_ok else ""
            )
            self.conn.execute(
                f"INSERT INTO variables (name, type, title, description) "
                f"VALUES (%s, %s, %s, %s){on_conflict}",
                (v.name, v.variable_type, v.title, v.description),
            )

        self.update_views()

    def get_user_variables(self):
        user_variables = {}
        rows = self.conn.execute("""
            SELECT name, title, type, description, attributes FROM variables
            WHERE type IS NOT NULL
        """).fetchall()
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
        with self.conn.transaction():
            # We want to read & write in the same transaction. Locking the
            # existing rows with FOR UPDATE up front prevents the deadlock
            # where two processes are both holding a read snapshot & waiting
            # to upgrade to a write (the same concern the sqlite
            # BEGIN IMMEDIATE was protecting against).
            rows = self.conn.execute("""
                SELECT name, title, type, description, attributes FROM variables
                WHERE type IS NULL
                FOR UPDATE
            """).fetchall()
            for row in rows:
                var = dict(row)
                vars_in_db[var.pop("name")] = var

            updates = {n: v for (n, v) in vars.items()
                       if v != vars_in_db.get(n, None)}
            log.debug("Updating stored metadata for %d computed variables",
                      len(updates))

            # Write new & changed variables
            # TODO: what if a new computed variable name matches an existing user var?
            if len(updates) > 0:
                self.conn.cursor().executemany("""
                    INSERT INTO variables (name, type, title, description, attributes)
                    VALUES (%s, NULL, %s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET
                        type = EXCLUDED.type,
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        attributes = EXCLUDED.attributes
                """, [
                    (n, v['title'], v['description'], _as_jsonb(v['attributes']))
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
        names = { r["name"] for r in
                  self.conn.execute("SELECT DISTINCT name FROM run_variables").fetchall() }

        # It could be that a user-editable variable was created but hasn't been
        # assigned yet, which means an entry for it won't have been created in
        # the run_variables table. Hence we look in the variables table as well.
        names |= { r["name"] for r in
                   self.conn.execute("SELECT name FROM variables").fetchall() }

        return list(names)

    def update_views(self):
        variables = self.variable_names()

        def quote_ident(ident: str) -> str:
            return '"' + ident.replace('"', '""') + '"'

        def quote_lit(lit: str) -> str:
            return "'" + lit.replace("'", "''") + "'"

        # `run_value(kind, scalar, blob)` (defined in POSTGRES_BASELINE)
        # returns JSONB: scalars as their JSON, blobs as a small
        # `{"kind": ..., "size": ...}` metadata object.
        col_select_sql = (
            "max(CASE WHEN name={lit} "
            "THEN run_value(value_kind, value_scalar, value_blob) END) AS {ident}"
        )
        runs_cols = ", ".join(
            col_select_sql.format(lit=quote_lit(v), ident=quote_ident(v))
            for v in variables
        )
        max_diff_cols = ", ".join(
            f"max(CASE WHEN name={quote_lit(v)} THEN max_diff END) AS {quote_ident(v)}"
            for v in variables
        )
        # Postgres requires at least one column in the SELECT list; when there
        # are no variables yet, emit a placeholder so CREATE VIEW succeeds.
        if not variables:
            runs_cols = "NULL::JSONB AS _placeholder"
            max_diff_cols = "NULL::DOUBLE PRECISION AS _placeholder"

        with self.conn.transaction():
            self.conn.execute("DROP VIEW IF EXISTS runs")
            self.conn.execute("DROP VIEW IF EXISTS max_diffs")
            self.conn.execute(f"""
                CREATE VIEW runs
                AS SELECT run_info.proposal, run_info.run, start_time, added_at, {runs_cols}
                   FROM run_variables INNER JOIN run_info ON run_variables.proposal = run_info.proposal AND run_variables.run = run_info.run
                   GROUP BY run_info.proposal, run_info.run, start_time, added_at
            """)
            self.conn.execute(f"""
                CREATE VIEW max_diffs
                AS SELECT proposal, run, {max_diff_cols}
                   FROM run_variables
                   GROUP BY proposal, run
            """)

    def set_variable(
            self, proposal: int, run: int, name: str, reduced, provenance: str
    ):
        timestamp = datetime.now(tz=timezone.utc).timestamp()

        variable = asdict(reduced)
        kind, scalar, blob, summary_type = _encode_value(
            variable["value"], variable["summary_type"]
        )
        variable["summary_type"] = summary_type
        variable["value_kind"] = kind
        variable["value_scalar"] = scalar
        variable["value_blob"] = blob
        del variable["value"]

        # If the value is None that implies that the variable should be
        # 'deleted', in which case we don't actually delete the row, but rather
        # set the value and all metadata fields in the database to NULL.
        if kind == "null":
            for key in ("max_diff", "summary_method", "summary_type", "attributes"):
                variable[key] = None
        elif variable["attributes"] is not None:
            variable["attributes"] = _as_jsonb(variable["attributes"])

        variable["proposal"] = proposal
        variable["run"] = run
        variable["name"] = name
        variable["timestamp"] = timestamp
        variable["provenance"] = provenance

        # TODO: enable these code snippets when we add support for versioning
        # latest_version = self.conn.execute("""
        #     SELECT max(version) FROM run_variables
        #     WHERE proposal=%s AND run=%s AND name=%s
        # """, (proposal, run, name)).fetchone()["max"]
        variable["version"] = 1 # if latest_version is None else latest_version + 1

        # These columns should match those in the run_variables table
        cols = [
            "proposal", "run", "name", "version",
            "value_kind", "value_scalar", "value_blob",
            "timestamp", "max_diff", "provenance",
            "summary_method", "summary_type", "attributes",
        ]
        col_list = ", ".join(cols)
        col_values = ", ".join([f"%({col})s" for col in cols])
        col_updates = ", ".join([f"{col} = EXCLUDED.{col}" for col in cols])

        with self.conn.transaction():
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
        with self.conn.transaction():
            # First delete from the `variables` table
            self.conn.execute("""
            DELETE FROM variables
            WHERE name = %s
            """, (name,))

            # And then `run_variables`
            self.conn.execute("""
            DELETE FROM run_variables
            WHERE name = %s
            """, (name, ))

            self.update_views()

    def add_tag(self, tag_name: str) -> int:
        """Add a new tag to the database if it doesn't exist.
        Returns the tag ID."""
        with self.conn.transaction():
            row = self.conn.execute("""
                INSERT INTO tags (name) VALUES (%s)
                ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
            """, (tag_name,)).fetchone()
            return row["id"]

    def get_tag_id(self, tag_name: str) -> Optional[int]:
        """Get the ID of a tag by its name."""
        with self.conn.transaction():
            row = self.conn.execute(
                "SELECT id FROM tags WHERE name = %s", (tag_name,)
            ).fetchone()
            return row["id"] if row else None

    def tag_variable(self, variable_name: str, tag_name: str):
        """Associate a tag with a variable."""
        tag_id = self.add_tag(tag_name)
        with self.conn.transaction():
            self.conn.execute(
                "INSERT INTO variable_tags (variable_name, tag_id) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING",
                (variable_name, tag_id)
            )

    def untag_variable(self, variable_name: str, tag_name: str):
        """Remove a tag association from a variable."""
        tag_id = self.get_tag_id(tag_name)
        if tag_id is not None:
            with self.conn.transaction():
                self.conn.execute(
                    "DELETE FROM variable_tags WHERE variable_name = %s AND tag_id = %s",
                    (variable_name, tag_id)
                )

    def get_variable_tags(self, variable_name: str) -> list[str]:
        """Get all tags associated with a variable."""
        rows = self.conn.execute("""
            SELECT t.name
            FROM tags t
            JOIN variable_tags vt ON t.id = vt.tag_id
            WHERE vt.variable_name = %s
        """, (variable_name,)).fetchall()
        return [r["name"] for r in rows]

    def get_variables_by_tag(self, tag_name: str) -> list[str]:
        """Get all variables that have a specific tag."""
        rows = self.conn.execute("""
            SELECT vt.variable_name
            FROM variable_tags vt
            JOIN tags t ON vt.tag_id = t.id
            WHERE t.name = %s
        """, (tag_name,)).fetchall()
        return [r["variable_name"] for r in rows]

    def get_all_tags(self) -> list[str]:
        """Get all existing tags."""
        rows = self.conn.execute("SELECT name FROM tags ORDER BY name").fetchall()
        return [r["name"] for r in rows]


def _encode_value(value: Any, summary_type: Optional[str]):
    """Map a Python value to (value_kind, value_scalar, value_blob, summary_type).

    `value_scalar` is wrapped in Jsonb(...) so psycopg3 adapts it to JSONB.
    `value_blob` is raw bytes (psycopg3 adapts to BYTEA). `summary_type` is
    returned so complex/numpy values can inherit the default tag, matching
    the pre-port behaviour.
    """
    if value is None:
        return "null", None, None, summary_type
    if isinstance(value, bool):
        return "int", Jsonb(int(value)), None, summary_type
    if isinstance(value, int):
        return "int", Jsonb(value), None, summary_type
    if isinstance(value, float):
        return "float", Jsonb(value), None, summary_type
    if isinstance(value, str):
        return "text", Jsonb(value), None, summary_type
    if isinstance(value, complex):
        return "complex", None, complex2blob(value), summary_type or "complex"
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Unsupported array dtype for database storage")
        return "numpy", None, numpy2blob(value), summary_type or "numpy"
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return BlobTypes.identify(raw).value, None, raw, summary_type
    raise TypeError(f"Unsupported value type: {type(value)!r}")


def _as_jsonb(value):
    """Wrap a value for a JSONB column; pass through None. Back-compat path
    handles pre-serialised JSON strings left over from the sqlite era."""
    if value is None:
        return None
    if isinstance(value, str):
        return Jsonb(json.loads(value))
    return Jsonb(value)


class KeyValueMapping(MutableMapping):
    """
    Simple class that represents a dictionary backed by a two-column table.

    Values are coerced to `str` on write (the schema stores `value TEXT`);
    callers that need typed values cast explicitly on read, e.g.
    `int(db.metameta["proposal"])`.

    Note that the `table` argument is assumed to come from a trusted source, it
    isn't quoted into the internal SQL expressions.
    """

    def __init__(self, conn, table):
        self.conn = conn
        self.table = table

    def __getitem__(self, key):
        row = self.conn.execute(
            f"SELECT value FROM {self.table} WHERE key=%s", (key,)
        ).fetchone()
        if row is not None:
            return row["value"]
        raise KeyError

    def __setitem__(self, key, value):
        with self.conn.transaction():
            self.conn.execute(
                f"INSERT INTO {self.table} (key, value) VALUES (%(key)s, %(value)s) "
                "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                {'key': key, 'value': str(value)}
            )

    def update(self, other=(), **kwargs):
        # Override to do the update in one transaction
        d = {}
        d.update(other, **kwargs)
        if not d:
            return

        with self.conn.transaction():
            self.conn.cursor().executemany(
                f"INSERT INTO {self.table} (key, value) VALUES (%(key)s, %(value)s) "
                "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                [{'key': k, 'value': str(v)} for (k, v) in d.items()]
            )

    def __delitem__(self, key):
        with self.conn.transaction():
            c = self.conn.execute(f"DELETE FROM {self.table} WHERE key=%s", (key,))
            if c.rowcount == 0:
                raise KeyError(key)

    def __iter__(self):
        rows = self.conn.execute(f"SELECT key FROM {self.table}").fetchall()
        return iter(r["key"] for r in rows)

    def __len__(self):
        row = self.conn.execute(f"SELECT count(*) AS n FROM {self.table}").fetchone()
        return row["n"]

    def setdefault(self, key, default=None):
        try:
            with self.conn.transaction():
                self.conn.execute(
                    f"INSERT INTO {self.table} (key, value) VALUES (%s, %s)",
                    (key, str(default))
                )
            return default
        except psycopg.IntegrityError:
            # The key is already present
            return self[key]

    def to_dict(self):
        rows = self.conn.execute(
            f"SELECT key, value FROM {self.table}"
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

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

    # These messages are sent on the file submission topic & received by the
    # combiner service, unlike the other messages above.
    file_submission = 'file_submission'

    # Sent on the admin topic; handled by the listener's provisioning path.
    create_proposal_db = 'create_proposal_db'

def msg_dict(kind: MsgKind, data: dict):
    return {'msg_kind': kind.value, 'data': data}

def initialize_proposal(root_path, proposal=None, context_file_src=None, user_vars_src=None):
    # Ensure the directory exists
    root_path.mkdir(parents=True, exist_ok=True)
    if root_path.stat().st_uid == os.getuid():
        os.chmod(root_path, 0o777)

    # With Postgres, the DB itself is provisioned by the listener via Kafka.
    # This function assumes `runs.pgpass` is already present; if not, the
    # caller should request provisioning first.
    if not pgpass_path(root_path).is_file():
        raise RuntimeError(
            f"No runs.pgpass in {root_path}. Request a Postgres database via "
            f"`damnit proposal --request-pg {proposal}` first."
        )

    db = DamnitDB.from_dir(root_path)

    context_path = root_path / "context.py"
    # Copy initial context file if necessary
    if not context_path.is_file():
        if context_file_src is not None:
            shutil.copyfile(context_file_src, context_path)
        else:
            context_path.touch()
        os.chmod(context_path, 0o666)

    # Copy user editable variables if requested
    if user_vars_src is not None:
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
