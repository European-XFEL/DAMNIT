import pickle

import h5py
import numpy as np

from damnit.migrations import migrate_images
from damnit.backend.extract_data import add_to_db


def test_migrate_images(mock_db, monkeypatch):
    db_dir, db = mock_db
    monkeypatch.chdir(db_dir)

    proposal = 1234
    db.metameta["proposal"] = proposal

    extracted_data_dir = db_dir / "extracted_data"
    extracted_data_dir.mkdir()

    # Create fake HDF5 file and SQLite rows with old 2D image summaries
    run = 42
    mock_h5_path = extracted_data_dir / f"p{proposal}_r{run}.h5"
    foo = np.random.rand(100, 100)
    bar = 42
    with h5py.File(mock_h5_path, "w") as f:
        f.create_dataset(".reduced/foo", data=foo)
        f.create_dataset(".reduced/bar", data=np.array(bar))
    add_to_db({"foo": foo, "bar": bar}, db.conn, proposal, run)

    migrate_images(db, db_dir)

    # Now the summary should be a 3D RGBA image, and the non-image summary
    # should be unchanged.
    with h5py.File(mock_h5_path) as f:
        assert f[".reduced/foo"].shape == (*foo.shape, 4)
        assert f[".reduced/foo"].dtype == np.uint8
        assert f[".reduced/bar"][()] == bar

    cursor = db.conn.execute("SELECT * FROM runs")
    row = cursor.fetchone()
    thumbnail = pickle.loads(row["foo"])
    assert isinstance(thumbnail, np.ndarray)
    assert thumbnail.shape == (*foo.shape, 4)
    assert thumbnail.dtype == np.uint8
    assert row["bar"] == bar
