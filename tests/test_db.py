import pytest

from damnit.backend.db import complex2blob, blob2complex


def test_metameta(mock_db):
    _, db = mock_db

    # Test various parts of the mutable mapping API
    assert set(db.metameta.keys()) == {'db_id', 'data_format_version', 'concurrent_jobs'}
    del db.metameta['db_id']
    del db.metameta['data_format_version']
    del db.metameta['concurrent_jobs']
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
