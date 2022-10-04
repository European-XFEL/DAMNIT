from unittest.mock import patch

from amore_mid_prototype.cli import main
from amore_mid_prototype.backend.db import get_meta

def test_new_id(mock_db, monkeypatch):
    db_dir, db = mock_db

    old_id = get_meta(db, "db_id")

    # Test setting the ID with an explicit path
    with patch("sys.argv", ["amore-proto", "new-id", str(db_dir)]):
        main()
    assert old_id != get_meta(db, "db_id")

    # Test with the default path (PWD)
    monkeypatch.chdir(db_dir)
    old_id = get_meta(db, "db_id")
    with patch("sys.argv", ["amore-proto", "new-id"]):
        main()
    assert old_id != get_meta(db, "db_id")
