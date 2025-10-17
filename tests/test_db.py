import pytest

from damnit.backend.db import complex2blob, blob2complex, DamnitDB
from damnit.backend.db_migrations import latest_version
import sqlite3
from pathlib import Path


def test_metameta(mock_db):
    _, db = mock_db

    # Test various parts of the mutable mapping API
    assert set(db.metameta.keys()) == {'db_id', 'data_format_version', 'concurrent_jobs', 'damnit_python'}
    del db.metameta['db_id']
    del db.metameta['data_format_version']
    del db.metameta['concurrent_jobs']
    del db.metameta['damnit_python']
    assert len(db.metameta) == 0

    db.metameta['a'] = 12
    assert set(db.metameta) == {'a'}
    db.metameta.setdefault('a', 34)
    db.metameta.setdefault('b', 34)
    assert set(db.metameta.items()) == {('a', 12), ('b', 34)}

    db.metameta.update({'b': 56, 'c': 78})
    assert set(db.metameta) == {'a', 'b', 'c'}
    assert set(db.metameta.values()) == {12, 56, 78}


def test_run_comment(mock_db):
    _, db = mock_db

    db.ensure_run(1234, 5, added_at=1670498578.)
    db.change_run_comment(1234, 5, 'Test comment')
    runs = [tuple(r) for r in db.conn.execute(
        "SELECT proposal, run, comment FROM runs"
    )]
    assert runs == [(1234, 5, 'Test comment')]


def test_standalone_comment(mock_db):
    _, db = mock_db

    ts = 1670498578.
    cid = db.add_standalone_comment(ts, 'Comment without run')
    db.change_standalone_comment(cid, 'Revised comment')
    res = [tuple(r) for r in db.conn.execute("SELECT * FROM time_comments")]
    assert res == [(ts, 'Revised comment')]


def test_tags(mock_db_with_data):
    _, db = mock_db_with_data

    # Test adding tags and getting tag IDs
    tag_id1 = db.add_tag("SPB")
    tag_id2 = db.add_tag("SFX")
    assert tag_id1 != tag_id2
    assert db.get_tag_id("SPB") == tag_id1
    assert db.get_tag_id("SFX") == tag_id2
    assert db.get_tag_id("nonexistent") is None

    # Test adding duplicate tag (should return same ID)
    assert db.add_tag("SPB") == tag_id1

    # Test getting tags for a variable
    var1_tags = db.get_variable_tags("scalar1")
    assert set(var1_tags) == {"scalar", "integer"}
    var2_tags = db.get_variable_tags("scalar2")
    assert set(var2_tags) == {"scalar", "float"}
    empty_tags = db.get_variable_tags("nonexistent_var")
    assert empty_tags == []

    # Test getting variables by tag
    scalar_vars = db.get_variables_by_tag("scalar")
    assert set(scalar_vars) == {"scalar1", "scalar2"}
    text_vars = db.get_variables_by_tag("text")
    assert set(text_vars) == {"empty_string"}
    nonexistent_vars = db.get_variables_by_tag("nonexistent")
    assert nonexistent_vars == []

    # Test getting all tags
    all_tags = db.get_all_tags()
    assert set(all_tags) == {"scalar", "integer", "float", "text", "SPB", "SFX"}

    # Test untagging variables
    db.untag_variable("scalar1", "scalar")
    assert set(db.get_variable_tags("scalar1")) == {"integer"}

    # Test untagging with nonexistent tag (should not raise error)
    db.untag_variable("scalar1", "nonexistent")
    assert set(db.get_variable_tags("scalar1")) == {"integer"}

    # Test untagging with nonexistent variable (should not raise error)
    db.untag_variable("nonexistent_var", "important")


def test_tag_cleanup(tmp_path):
    """
    Tests that the SQLite trigger correctly cleans up orphaned tags
    when variable-tag associations are removed or variables are deleted.
    """
    db = DamnitDB.from_dir(tmp_path)

    # helper function
    def _var_def(title, tags=None):
        return {
            "title": title,
            "description": f"Desc for {title}",
            "tags": tags,
            "attributes": None,
            "type": None,
        }

    # 1. remove tags by updating variables
    db.update_computed_variables({
        "var1": _var_def("Var1 Title", ["tagA", "tagB"]),
        "var2": _var_def("Var2 Title", ["tagB"]),
    })
    assert set(db.get_all_tags()) == {"tagA", "tagB"}
    assert set(db.get_variable_tags("var1")) == {"tagA", "tagB"}
    assert set(db.get_variable_tags("var2")) == {"tagB"}

    # Update var1 to remove tagA. 'tagA' is removed by trigger.
    db.update_computed_variables({
        "var1": _var_def("Var1 Title", ["tagB"]),
        "var2": _var_def("Var2 Title", ["tagB"]),
    })
    assert set(db.get_all_tags()) == {"tagB"}
    assert set(db.get_variable_tags("var1")) == {"tagB"}
    assert set(db.get_variable_tags("var2")) == {"tagB"}

    # Update var1 to remove tagB. 'tagB' is still used by var2, so it remains.
    db.update_computed_variables({
        "var1": _var_def("Var1 Title"),
        "var2": _var_def("Var2 Title", ["tagB"]),
    })
    assert set(db.get_all_tags()) == {"tagB"}
    assert set(db.get_variable_tags("var1")) == set()
    assert set(db.get_variable_tags("var2")) == {"tagB"}

    # Update var2 to remove tagB. 'tagB' now becomes orphaned and trigger removes it.
    db.update_computed_variables({
        "var1": _var_def("Var1 Title"),
        "var2": _var_def("Var2 Title")
    })
    assert set(db.get_variable_tags("var2")) == set()
    assert set(db.get_all_tags()) == set()

    # 2. remove tags by delete_variable
    db.update_computed_variables({
        "var3": _var_def("Var3 Title", ["tagC", "tagD"]),
        "var4": _var_def("Var4 Title", ["tagD"]),
    })
    assert set(db.get_all_tags()) == {"tagC", "tagD"}

    # Delete var3.
    db.delete_variable("var3")
    assert set(db.get_all_tags()) == {"tagD"}
    assert "var3" not in {row[0] for row in db.conn.execute("SELECT name FROM variables")}
    assert set(db.get_variable_tags("var4")) == {"tagD"}

    # Delete var4.
    db.delete_variable("var4")
    assert set(db.get_all_tags()) == set()
    assert "var4" not in {row[0] for row in db.conn.execute("SELECT name FROM variables")}


@pytest.mark.parametrize("value", [
    1+2j,
    0+0j,
    -1.5-3.7j,
    2.5+0j,
    0+3.1j,
    float('inf')+0j,
    complex(float('inf'), -float('inf')),
])
def test_complex_blob_conversion(value):
    # Test that converting complex -> blob -> complex preserves the value
    blob = complex2blob(value)
    result = blob2complex(blob)
    assert result == value


def test_new_db_schema_is_latest(tmp_path):
    db = DamnitDB.from_dir(tmp_path)

    # Schema version should match the latest known migration
    assert db.metameta["data_format_version"] == latest_version()

    # run_variables should include the v2 column 'attributes'
    cols = [row[1] for row in db.conn.execute("PRAGMA table_info('run_variables')").fetchall()]
    assert "attributes" in cols

    # v3 tables should exist
    tables = {row[0] for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "tags" in tables and "variable_tags" in tables

    # v4 trigger should exist
    trig = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND name='delete_orphan_tags_after_variable_tag_delete'"
    ).fetchone()
    assert trig is not None


def test_upgrade_from_v2_creates_backup_and_applies_missing(tmp_path):
    # Start with a fresh (latest) DB, then downgrade state to v2 (no tags tables, no trigger)
    db = DamnitDB.from_dir(tmp_path)
    with db.conn:
        db.conn.execute("DROP TRIGGER IF EXISTS delete_orphan_tags_after_variable_tag_delete")
        db.conn.execute("DROP TABLE IF EXISTS variable_tags")
        db.conn.execute("DROP TABLE IF EXISTS tags")
    db.metameta["data_format_version"] = 2
    db.close()

    # No backups yet
    before = list(Path(tmp_path).glob("runs.sqlite.bak.*"))

    # Reopen to trigger upgrade
    db2 = DamnitDB.from_dir(tmp_path)

    # Backup created
    after = list(Path(tmp_path).glob("runs.sqlite.bak.*"))
    assert len(after) == len(before) + 1

    # Upgraded to latest
    assert db2.metameta["data_format_version"] == latest_version()

    # v3 tables and v4 trigger restored
    tables = {row[0] for row in db2.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "tags" in tables and "variable_tags" in tables
    trig = db2.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND name='delete_orphan_tags_after_variable_tag_delete'"
    ).fetchone()
    assert trig is not None


def test_min_openable_version_guard(tmp_path):
    # Create a legacy DB with version 0 (unsupported)
    db_path = Path(tmp_path) / "runs.sqlite"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE metameta(key PRIMARY KEY NOT NULL, value)")
    con.execute("INSERT INTO metameta VALUES('data_format_version', 0)")
    con.commit()
    con.close()

    # By default, opening should raise
    with pytest.raises(RuntimeError):
        DamnitDB.from_dir(tmp_path)

    # With allow_old=True, it should open and upgrade to latest
    db = DamnitDB(Path(tmp_path, "runs.sqlite"), allow_old=True)
    assert db.metameta["data_format_version"] == latest_version()
