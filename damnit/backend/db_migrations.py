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


def _to_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        dedent(
            """
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
            """
        )
    )


def _to_v2(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE run_variables ADD COLUMN attributes")


def _to_v3(conn: sqlite3.Connection) -> None:
    conn.executescript(
        dedent(
            """
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
            """
        )
    )


def _to_v4(conn: sqlite3.Connection) -> None:
    conn.execute(
        dedent(
            """
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
        )
    )


MIGRATIONS: list[Migration] = [
    Migration(
        to_version=1,
        description="Baseline schema (core tables, views)",
        apply=_to_v1,
    ),
    Migration(
        to_version=2,
        description="Add attributes column to run_variables",
        apply=_to_v2,
    ),
    Migration(
        to_version=3,
        description="Add tags and variable_tags tables",
        apply=_to_v3,
    ),
    Migration(
        to_version=4,
        description="Add trigger to cleanup orphaned tags",
        apply=_to_v4,
    ),
]


def create_backup(db_path: Path) -> Path:
    """
    Create a timestamped backup copy of the SQLite database file next to it.
    Returns the backup path.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_suffix(db_path.suffix + f".bak.{ts}")
    shutil.copy2(db_path, backup_path)
    return backup_path


def pending_migrations(from_version: int, to_version: int) -> Iterable[Migration]:
    for m in MIGRATIONS:
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
    return max((m.to_version for m in MIGRATIONS), default=0)
