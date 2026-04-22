import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Callable, Iterable

import sqlite3


@dataclass(frozen=True)
class Migration:
    """
    A single schema migration step applied to go to `to_version`.

    - `to_version`: schema version after this migration completes
    - `description`: short human description for docs/logs
    - `apply`: a callable that takes a sqlite3.Connection and applies changes
    """

    to_version: int
    description: str
    apply: Callable[[sqlite3.Connection], None]


MIGRATIONS: list[Migration] = []


def migration_step(
    to_version: int, *, schema: str | None = None, description: str = ""
):
    """Register a migration step.

    Two usage patterns are supported:

    - As a decorator for callable migrations:
        @migration_step(to_version=5)
        def _to_v5(conn):
            ...  # arbitrary Python using the connection

    - As a direct call for schema-only migrations (shortcut):
        migration_step(
            to_version=6,
            description="Add new table",
            schema='''
            CREATE TABLE IF NOT EXISTS example(
                id INTEGER PRIMARY KEY
            );
            ''',
        )

    When used with `schema`, the SQL is dedented and applied via
    `conn.executescript(...)`. When used as a decorator, the function's
    docstring is used as the description (if present).
    """

    # Shortcut: register a schema-only migration without a callable
    if schema is not None:
        sql = dedent(schema)

        def _apply(conn: sqlite3.Connection) -> None:
            conn.executescript(sql)

        migration = Migration(
            to_version=to_version,
            description=description,
            apply=_apply,
        )
        MIGRATIONS.append(migration)
        return migration

    # Decorator form for callable migrations
    def decorator(func: Callable[[sqlite3.Connection], None]):
        migration = Migration(
            to_version=to_version,
            description=func.__doc__ or description or "",
            apply=func,
        )
        MIGRATIONS.append(migration)
        return migration

    return decorator


migration_step(
    to_version=1,
    description="Initial baseline schema with core tables and views.",
    schema="""
-- Core schema (baseline)
CREATE TABLE IF NOT EXISTS run_info(
    proposal,
    run,
    start_time,
    added_at
);
CREATE UNIQUE INDEX IF NOT EXISTS proposal_run ON run_info (proposal, run);

-- Long/narrow variables store (without attributes yet)
CREATE TABLE IF NOT EXISTS run_variables(
    proposal,
    run,
    name,
    version,
    value,
    timestamp,
    max_diff,
    provenance,
    summary_type,
    summary_method
);
CREATE UNIQUE INDEX IF NOT EXISTS variable_version
    ON run_variables (proposal, run, name, version);

-- Dummy views are replaced by update_views() later
CREATE VIEW IF NOT EXISTS runs      AS SELECT * FROM run_info;
CREATE VIEW IF NOT EXISTS max_diffs AS SELECT proposal, run FROM run_info;

-- Metadata tables
CREATE TABLE IF NOT EXISTS metameta(
    key PRIMARY KEY NOT NULL,
    value
);
CREATE TABLE IF NOT EXISTS variables(
    name TEXT PRIMARY KEY NOT NULL,
    type TEXT,
    title TEXT,
    description TEXT,
    attributes TEXT
);
CREATE TABLE IF NOT EXISTS time_comments(
    timestamp,
    comment
);
""",
)


migration_step(
    to_version=2,
    description="Add attributes column to run_variables table.",
    schema="""ALTER TABLE run_variables ADD COLUMN attributes""",
)


migration_step(
    to_version=3,
    description="Add tags and variable_tags tables.",
    schema="""\
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
""",
)


migration_step(
    to_version=4,
    description="Add trigger to cleanup orphaned tags after variable_tags deletions.",
    schema="""\
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
""",
)


# Postgres baseline schema for a freshly provisioned per-proposal DB.
#
# Not a migration — this is the starting point. Postgres DBs never replay the
# sqlite v1..v4 ladder; legacy data reaches Postgres via the one-shot migrator,
# which inserts rows directly into this schema.
#
# `data_format_version` continues the sqlite counter: sqlite ended at v4, so
# this baseline is v5. Future Postgres-side changes go through migration_step
# as v6, v7, ... on top of this starting point.
POSTGRES_BASELINE = """\
SET client_min_messages = WARNING;

CREATE TABLE run_info (
    proposal    INTEGER NOT NULL,
    run         INTEGER NOT NULL,
    start_time  DOUBLE PRECISION,
    added_at    DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (proposal, run)
);

CREATE TABLE variables (
    name         TEXT PRIMARY KEY,
    type         TEXT,
    title        TEXT,
    description  TEXT,
    attributes   JSONB
);

CREATE TABLE run_variables (
    proposal        INTEGER NOT NULL,
    run             INTEGER NOT NULL,
    name            TEXT    NOT NULL,
    version         INTEGER NOT NULL,

    -- Value storage: the kind tag disambiguates which of scalar/blob is used.
    -- Scalar kinds (int, float, text, timestamp) live in value_scalar as JSONB.
    -- Blob kinds (png, numpy, complex) live in value_blob as raw bytes.
    -- The 'null' kind means the variable was logically deleted for this run.
    value_kind      TEXT    NOT NULL,
    value_scalar    JSONB,
    value_blob      BYTEA,

    timestamp       DOUBLE PRECISION,
    max_diff        DOUBLE PRECISION,
    provenance      TEXT,
    summary_method  TEXT,
    summary_type    TEXT,
    attributes      JSONB,

    PRIMARY KEY (proposal, run, name, version),
    FOREIGN KEY (proposal, run) REFERENCES run_info (proposal, run),

    CONSTRAINT value_shape CHECK (
        (value_kind = 'null'
            AND value_scalar IS NULL     AND value_blob IS NULL) OR
        (value_kind IN ('int', 'float', 'text')
            AND value_scalar IS NOT NULL AND value_blob IS NULL) OR
        (value_kind IN ('png', 'numpy', 'complex', 'unknown')
            AND value_scalar IS NULL     AND value_blob IS NOT NULL)
    )
);
CREATE INDEX run_variables_name ON run_variables (name);

CREATE TABLE tags (
    id    BIGSERIAL PRIMARY KEY,
    name  TEXT UNIQUE NOT NULL
);

CREATE TABLE variable_tags (
    variable_name  TEXT   NOT NULL REFERENCES variables (name) ON DELETE CASCADE,
    tag_id         BIGINT NOT NULL REFERENCES tags (id)        ON DELETE CASCADE,
    PRIMARY KEY (variable_name, tag_id)
);

CREATE OR REPLACE FUNCTION delete_orphan_tags() RETURNS TRIGGER AS $$
BEGIN
    DELETE FROM tags t
    WHERE t.id = OLD.tag_id
      AND NOT EXISTS (SELECT 1 FROM variable_tags WHERE tag_id = OLD.tag_id);
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER delete_orphan_tags_after_variable_tag_delete
AFTER DELETE ON variable_tags
FOR EACH ROW EXECUTE FUNCTION delete_orphan_tags();

-- Unified accessor used by the dynamic `runs` view. Returns the scalar JSONB
-- directly for scalar kinds; for blob kinds, returns a small metadata object
-- describing the blob without materialising it. Consumers that actually need
-- the bytes query run_variables directly.
CREATE OR REPLACE FUNCTION run_value(kind TEXT, scalar JSONB, blob BYTEA)
RETURNS JSONB AS $$
    SELECT CASE
        WHEN kind = 'null'       THEN NULL
        WHEN scalar IS NOT NULL  THEN scalar
        ELSE jsonb_build_object('kind', kind, 'size', octet_length(blob))
    END
$$ LANGUAGE sql IMMUTABLE;

CREATE TABLE metameta (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

CREATE TABLE time_comments (
    id         BIGSERIAL PRIMARY KEY,
    timestamp  DOUBLE PRECISION NOT NULL,
    comment    TEXT NOT NULL
);
"""

POSTGRES_BASELINE_VERSION = 5


def pending_migrations(from_version: int, to_version: int) -> Iterable[Migration]:
    _assert_valid_migrations()
    for m in sorted(MIGRATIONS, key=lambda x: x.to_version):
        if from_version < m.to_version <= to_version:
            yield m


def apply_migrations(
    conn: sqlite3.Connection,
    from_version: int,
    to_version: int,
    set_version: Callable[[int], None],
) -> list[Migration]:
    """
    Apply all migration steps to go from `from_version` up to `to_version`.
    Sets the version after each successful step using `set_version`.
    Returns the list of applied migrations.
    """
    applied: list[Migration] = []
    for step in pending_migrations(from_version, to_version):
        # Each step in its own transaction for clarity and resilience
        with conn:
            step.apply(conn)
            set_version(step.to_version)
        applied.append(step)
    return applied


def latest_version() -> int:
    """Return the highest schema version known to the application."""
    _assert_valid_migrations()
    return max((m.to_version for m in MIGRATIONS), default=0)


def _assert_valid_migrations() -> None:
    """Ensure versions are exactly [1, 2, ..., len(MIGRATIONS)]."""
    versions = sorted(m.to_version for m in MIGRATIONS)
    expected = list(range(1, len(versions) + 1))
    if versions != expected:
        raise ValueError(f"Invalid migration versions: {versions} (expected {expected})")
