def test_metameta(mock_db):
    _, db = mock_db

    # Test various parts of the mutable mapping API
    assert set(db.metameta.keys()) == {'db_id', 'data_format_version'}
    del db.metameta['db_id']
    del db.metameta['data_format_version']
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


def test_tags(mock_db):
    _, db = mock_db

    # Test adding tags and getting tag IDs
    tag_id1 = db.add_tag("important")
    tag_id2 = db.add_tag("needs_review")
    assert tag_id1 != tag_id2
    assert db.get_tag_id("important") == tag_id1
    assert db.get_tag_id("needs_review") == tag_id2
    assert db.get_tag_id("nonexistent") is None

    # Test adding duplicate tag (should return same ID)
    assert db.add_tag("important") == tag_id1

    # Test tagging variables
    db.tag_variable("var1", "important")
    db.tag_variable("var1", "needs_review")
    db.tag_variable("var2", "important")

    # Test getting tags for a variable
    var1_tags = db.get_variable_tags("var1")
    assert set(var1_tags) == {"important", "needs_review"}
    var2_tags = db.get_variable_tags("var2")
    assert set(var2_tags) == {"important"}
    empty_tags = db.get_variable_tags("nonexistent_var")
    assert empty_tags == []

    # Test getting variables by tag
    important_vars = db.get_variables_by_tag("important")
    assert set(important_vars) == {"var1", "var2"}
    review_vars = db.get_variables_by_tag("needs_review")
    assert set(review_vars) == {"var1"}
    nonexistent_vars = db.get_variables_by_tag("nonexistent")
    assert nonexistent_vars == []

    # Test getting all tags
    all_tags = db.get_all_tags()
    assert set(all_tags) == {"important", "needs_review"}

    # Test untagging variables
    db.untag_variable("var1", "important")
    assert set(db.get_variable_tags("var1")) == {"needs_review"}
    
    # Test untagging with nonexistent tag (should not raise error)
    db.untag_variable("var1", "nonexistent")
    assert set(db.get_variable_tags("var1")) == {"needs_review"}

    # Test untagging with nonexistent variable (should not raise error)
    db.untag_variable("nonexistent_var", "important")
