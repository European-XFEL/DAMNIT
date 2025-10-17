Schema Migrations (SQLite)
=========================

This project evolves frequently. To keep database updates simple and safe, we
use a minimal migration runner that automatically upgrades existing databases
on first access and makes a local backup before applying changes.

How it works
------------

- The current schema version is stored in the `metameta` table under the key
  `data_format_version`.
- New databases are bootstrapped with a sentinel version and then the baseline
  migration (→ v1) plus subsequent steps are applied. The migration list in
  `damnit/backend/db_migrations.py` is the single source of truth.
- When opening an existing database, the code compares the stored version with
  the app’s required version. If upgrades are needed, it:
  - Creates a timestamped backup file next to `runs.sqlite`.
  - Applies each migration step in order, updating the version after each step.

Where to add a migration
------------------------

1. Bump the target version in `damnit/backend/db.py`:
   - Update `DATA_FORMAT_VERSION` to the next integer.
   - No monolithic schema dump is used; instead, add a migration.

2. Add a step in `damnit/backend/db_migrations.py`:
   - Create a function that applies the schema change using SQL which is safe
     to re-run (e.g. `ALTER TABLE ... ADD COLUMN` or `CREATE ... IF NOT EXISTS`).
   - Add a new `Migration` entry to `MIGRATIONS` with the `to_version` value
     matching `DATA_FORMAT_VERSION` and a concise `description`.

3. Document it:
   - Add a short note to `docs/internals.md` under the versioned list, or edit
     this file to keep a brief changelog.

Rollback and safety
-------------------

- A backup file `runs.sqlite.bak.YYYYMMDD-HHMMSS` is created just before any
  migration is applied. If you encounter issues, stop the application and
  replace `runs.sqlite` with the backup.
- Migrations are run inside transactions. If any step fails, previous changes
  are not committed.

Advanced notes
--------------

- Opening legacy/unsupported databases can be allowed by passing
  `allow_old=True` to `DamnitDB` (see `MIN_OPENABLE_VERSION` in
  `damnit/backend/db.py`). Prefer upgrading when possible.
- Keep migrations pragmatic and focused: prefer additive, idempotent changes;
  avoid risky table rewrites unless absolutely necessary.
