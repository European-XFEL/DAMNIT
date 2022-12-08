
def test_metameta(mock_db):
    _, db = mock_db

    # Test various parts of the mutable mapping API
    assert set(db.metameta.keys()) == {'db_id'}
    del db.metameta['db_id']
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
        "SELECT proposal, runnr, comment FROM runs"
    )]
    assert runs == [(1234, 5, 'Test comment')]


def test_independent_comment(mock_db):
    _, db = mock_db

    ts = 1670498578.
    cid = db.add_independent_comment(ts, 'Comment without run')
    db.change_independent_comment(cid, 'Revised comment')
    res = [tuple(r) for r in db.conn.execute("SELECT * FROM time_comments")]
    assert res == [(ts, 'Revised comment')]
