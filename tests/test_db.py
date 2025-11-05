import os

import pytest

from damnit.backend.db import complex2blob, blob2complex, DamnitDB


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


def test_open_readonly(tmp_path):
    db = DamnitDB.from_dir(tmp_path)
    # Delete a known recent addition to the schema that old databases will not have
    del db.metameta["damnit_python"]
    db.close()

    os.chmod(tmp_path, 0o500)

    assert "damnit_python" not in DamnitDB.from_dir(tmp_path).metameta
