"""
Simplified database migration system for damnit.

This refactored implementation reduces complexity while maintaining all functionality:
- Embeds migrations directly in code instead of file discovery
- Simplifies bootstrap process
- Reduces classes and abstractions
- Improves error handling and transaction management
"""

import hashlib
import logging
import sqlite3
from collections.abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

log = logging.getLogger(__name__)


class MigrationError(Exception):
    """Base exception for migration errors."""
    pass


@dataclass 
class MigrationRecord:
    """Record of an applied migration."""
    version: int
    description: str
    applied_at: datetime
    checksum: str


class Migration(ABC):
    """A database migration definition."""

    def __init__(self):
        if self.version < 0:
            raise ValueError("Migration version must be a positive integer.")

    @abstractproperty
    def version(self) -> int:
        pass

    @abstractproperty
    def description(self) -> str:
        pass

    @abstractmethod
    def up(self, conn: sqlite3.Connection) -> None:
        pass

    @abstractmethod
    def down(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError(f"Migration {self.version} cannot be rolled back")

    @abstractmethod
    def validate(self) -> bool:
        return True


class V1(Migration):
    def up(self, conn):
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS run_info(proposal, run, start_time, added_at);
            CREATE UNIQUE INDEX IF NOT EXISTS proposal_run ON run_info (proposal, run);

            CREATE TABLE IF NOT EXISTS run_variables(
                proposal, run, name, version, value, timestamp, max_diff, 
                provenance, summary_type, summary_method
            );
            CREATE UNIQUE INDEX IF NOT EXISTS variable_version ON run_variables (proposal, run, name, version);

            CREATE VIEW IF NOT EXISTS runs AS SELECT * FROM run_info;
            CREATE VIEW IF NOT EXISTS max_diffs AS SELECT proposal, run FROM run_info;

            CREATE TABLE IF NOT EXISTS metameta(key PRIMARY KEY NOT NULL, value);
            CREATE TABLE IF NOT EXISTS variables(name TEXT PRIMARY KEY NOT NULL, type TEXT, title TEXT, description TEXT, attributes TEXT);
            CREATE TABLE IF NOT EXISTS time_comments(timestamp, comment);
        """)

    def down(self, conn):
        conn.executescript("""
            DROP TABLE IF EXISTS time_comments;
            DROP TABLE IF EXISTS variables;
            DROP TABLE IF EXISTS metameta;
            DROP VIEW IF EXISTS max_diffs;
            DROP VIEW IF EXISTS runs;
            DROP TABLE IF EXISTS run_variables;
            DROP TABLE IF EXISTS run_info;
        """)


def v2_up(conn):
    conn.execute("ALTER TABLE run_variables ADD COLUMN attributes")


def v3_up(conn):
    conn.executescript("""
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
    """)

def v3_down(conn):
    conn.execute("DROP TABLE IF EXISTS variable_tags")
    conn.execute("DROP TABLE IF EXISTS tags")

def v4_up(conn):
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS delete_orphan_tags_after_variable_tag_delete
        AFTER DELETE ON variable_tags
        FOR EACH ROW
        BEGIN
            DELETE FROM tags
            WHERE id = OLD.tag_id
            AND NOT EXISTS (
                SELECT 1 FROM variable_tags WHERE tag_id = OLD.tag_id
            );
        END
    """)

def v4_down(conn):
    conn.execute("DROP TRIGGER IF EXISTS delete_orphan_tags_after_variable_tag_delete")

class MigrationManager:
    """Manages database schema migrations with simplified design."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._migrations = self._define_migrations()

    def _define_migrations(self) -> dict[int, Migration]:
        """Define all migrations inline - no file discovery needed."""
        migrations = {}

        migrations[1] = Migration(1, "Create initial database schema", v1_up, v1_down)
        migrations[2] = Migration(2, "Add attributes column to run_variables", v2_up)
        migrations[3] = Migration(3, "Add tags and variable_tags tables", v3_up, v3_down)
        migrations[4] = Migration(4, "Add trigger to clean up orphaned tags", v4_up, v4_down)

        return migrations

    def _ensure_tracking_table(self):
        """Create migration tracking table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at REAL NOT NULL,
                checksum TEXT NOT NULL
            )
        """)

    def _tracking_table_exists(self) -> bool:
        """Check if migration tracking table exists."""
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='schema_migrations'
        """)
        return cursor.fetchone() is not None

    def _detect_legacy_version(self) -> int:
        """Detect version of legacy database by examining structure."""
        try:
            # Check if run_info table exists - if not, it's a new database
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='run_info'
            """)
            if not cursor.fetchone():
                return 0  # New database

            # Check for v2 feature (attributes column)
            cursor = self.conn.execute("PRAGMA table_info(run_variables)")
            columns = {row[1] for row in cursor.fetchall()}
            if 'attributes' not in columns:
                return 1

            # Check for v3 feature (tags table)
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='tags'
            """)
            if not cursor.fetchone():
                return 2

            # Check for v4 feature (trigger)
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='trigger' AND name='delete_orphan_tags_after_variable_tag_delete'
            """)
            if not cursor.fetchone():
                return 3

            return 4  # Current legacy version

        except Exception as e:
            log.warning(f"Could not detect legacy version: {e}")
            return 0

    def _get_applied_migrations(self) -> set[int]:
        """Get set of applied migration versions."""
        if not self._tracking_table_exists():
            return set()

        cursor = self.conn.execute("SELECT version FROM schema_migrations")
        return {row[0] for row in cursor.fetchall()}

    def _calculate_checksum(self, migration: Migration) -> str:
        """Calculate checksum for a migration."""
        # Use migration description and version as content for checksum
        content = f"{migration.version}:{migration.description}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _record_migration(self, migration: Migration):
        """Record that a migration was applied."""
        self.conn.execute("""
            INSERT OR REPLACE INTO schema_migrations 
            (version, description, applied_at, checksum) 
            VALUES (?, ?, ?, ?)
        """, (
            migration.version,
            migration.description, 
            datetime.now(tz=timezone.utc).timestamp(),
            self._calculate_checksum(migration)
        ))

    def get_current_version(self) -> int:
        """Get current database version."""
        if not self._tracking_table_exists():
            return 0

        cursor = self.conn.execute("SELECT MAX(version) FROM schema_migrations")
        result = cursor.fetchone()[0]
        return result or 0

    def get_applied_migrations(self) -> list[MigrationRecord]:
        """Get list of applied migrations."""
        if not self._tracking_table_exists():
            return []

        cursor = self.conn.execute("""
            SELECT version, description, applied_at, checksum 
            FROM schema_migrations 
            ORDER BY version
        """)

        records = []
        for row in cursor.fetchall():
            records.append(MigrationRecord(
                version=row[0],
                description=row[1],
                applied_at=datetime.fromtimestamp(row[2], tz=timezone.utc),
                checksum=row[3]
            ))
        return records

    def get_pending_migrations(self) -> list[Migration]:
        """Get list of pending migrations."""
        applied = self._get_applied_migrations()
        pending = []

        for version in sorted(self._migrations.keys()):
            if version not in applied:
                pending.append(self._migrations[version])

        return pending

    def has_pending_migrations(self) -> bool:
        """Check if there are pending migrations."""
        return len(self.get_pending_migrations()) > 0

    def apply_migrations(self, target_version: int | None = None):
        """Apply all pending migrations up to target version."""
        # Ensure tracking table exists
        self._ensure_tracking_table()

        # Handle legacy database bootstrap
        if not self._tracking_table_exists():
            # This is a legacy database - record existing migrations
            legacy_version = self._detect_legacy_version()
            for version in range(1, legacy_version + 1):
                if version in self._migrations:
                    migration = self._migrations[version]
                    self._record_migration(migration)
                    log.info(f"Marked legacy migration {version} as applied")

        # Get pending migrations
        pending = self.get_pending_migrations()

        if target_version is not None:
            pending = [m for m in pending if m.version <= target_version]

        if not pending:
            log.info("No pending migrations to apply")
            return

        log.info(f"Applying {len(pending)} migrations")

        for migration in pending:
            self._apply_migration(migration)

        log.info(f"Successfully applied {len(pending)} migrations")

    def _apply_migration(self, migration: Migration):
        """Apply a single migration within a transaction."""
        log.info(f"Applying migration {migration.version}: {migration.description}")

        try:
            with self.conn:
                migration.up(self.conn)
                self._record_migration(migration)

            log.info(f"Successfully applied migration {migration.version}")

        except Exception as e:
            log.error(f"Migration {migration.version} failed: {e}")
            raise MigrationError(f"Migration {migration.version} failed: {e}") from e

    def rollback_migration(self, version: int):
        """Rollback a specific migration."""
        if version not in self._migrations:
            raise MigrationError(f"Migration {version} not found")

        migration = self._migrations[version]

        if migration.down is None:
            raise MigrationError(f"Migration {version} does not support rollback")

        applied = self._get_applied_migrations()
        if version not in applied:
            raise MigrationError(f"Migration {version} is not applied")

        log.info(f"Rolling back migration {version}: {migration.description}")

        try:
            with self.conn:
                migration.down(self.conn)
                self.conn.execute("DELETE FROM schema_migrations WHERE version = ?", (version,))

            log.info(f"Successfully rolled back migration {version}")

        except Exception as e:
            log.error(f"Rollback of migration {version} failed: {e}")
            raise MigrationError(f"Rollback failed: {e}") from e


# # Integration with existing DamnitDB class
# class DamnitDB:
#     """Updated DamnitDB class with simplified migration system."""

#     def __init__(self, path, allow_old=False):
#         self.path = path.absolute()

#         db_existed = path.exists()
#         self.conn = sqlite3.connect(path, timeout=30)
#         self.conn.execute("PRAGMA foreign_keys = ON")
#         self.conn.row_factory = sqlite3.Row

#         # Initialize migration manager
#         self.migration_manager = MigrationManager(self.conn)

#         if not db_existed:
#             # New database - apply all migrations
#             log.debug("Creating new database with latest schema")
#             self.migration_manager.apply_migrations()
#         elif not allow_old:
#             # Existing database - apply any pending migrations
#             current_version = self.migration_manager.get_current_version()
#             if self.migration_manager.has_pending_migrations():
#                 log.info("Applying pending database migrations")
#                 self.migration_manager.apply_migrations()

#         # Initialize other components (metameta, etc.)
#         self._init_metadata()

#     def _init_metadata(self):
#         """Initialize metadata systems."""
#         # This would contain the existing metameta initialization
#         # and other setup code from the original DamnitDB.__init__
#         pass


# # Example usage and testing
# if __name__ == "__main__":
#     import tempfile
#     import os

#     # Test the simplified migration system
#     with tempfile.NamedTemporaryFile(delete=False) as f:
#         db_path = f.name

#     try:
#         # Test new database creation
#         conn = sqlite3.connect(db_path)
#         manager = MigrationManager(conn)

#         print(f"Current version: {manager.get_current_version()}")
#         print(f"Pending migrations: {len(manager.get_pending_migrations())}")

#         # Apply all migrations
#         manager.apply_migrations()

#         print(f"After migration - version: {manager.get_current_version()}")
#         print(f"Applied migrations: {[m.version for m in manager.get_applied_migrations()]}")

#         conn.close()

#     finally:
#         os.unlink(db_path)
