import logging
import sqlite3

DB = 'runs.sqlite'

log = logging.getLogger(__name__)

def open_db() -> sqlite3.Connection:
    """ Initialize the sqlite run database

    A new database is created if no pre-existing one is present. A single
    table is created: runs, which has columns:

        proposal, run, migrated_at

    More columns may be added later (by executing ALTER TABLE runs ADD COLUMN)
    """
    log.info("Opening database at %s", DB)
    conn = sqlite3.connect(str(DB))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runs(proposal, runnr, migrated_at, comment)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS proposal_run ON runs (proposal, runnr)"
    )
    conn.row_factory = sqlite3.Row
    return conn


if __name__ == '__main__':
    open_db()
