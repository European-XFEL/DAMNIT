Schema Migrations (SQLite)
=========================

This project evolves frequently. To keep database updates simple and safe, we
use a minimal migration runner that automatically upgrades existing databases
on first access and makes a local backup before applying changes.

How it works
------------

- The current schema version is stored in the `metameta` table under the key
  `data_format_version`.
- New databases are created with the latest schema (see `V4_SCHEMA` in
  `damnit/backend/db.py`) and the `data_format_version` is set accordingly.
- When opening an existing database, the code compares the stored version with
  the app’s required version. If upgrades are needed, it:
  - Creates a timestamped backup file next to `runs.sqlite`.
  - Applies each migration step in order, updating the version after each step.

Where to add a migration
------------------------

1. Bump the target version in `damnit/backend/db.py`:
   - Update `DATA_FORMAT_VERSION` to the next integer.
   - Update the base schema (`V4_SCHEMA`) if new installs should include your
     changes out of the box. Keep it idempotent (use `IF NOT EXISTS` where
     possible) and extend it rather than removing existing parts.

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

