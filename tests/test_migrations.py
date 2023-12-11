import pickle

import h5py
import numpy as np

from damnit.migrations import migrate_images
from damnit.backend.extract_data import add_to_db
from damnit.backend.db import BlobTypes

from .helpers import reduced_data_from_dict


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
    add_to_db(reduced_data_from_dict({"foo": foo, "bar": bar}),
              db, proposal, run)

    migrate_images(db, db_dir, dry_run=False)

    # Now the summary should be a PNG image, and the non-image summary
    # should be unchanged.
    with h5py.File(mock_h5_path) as f:
        assert f[".reduced/foo"].ndim == 1
        assert f[".reduced/foo"].dtype == np.uint8
        assert f[".reduced/foo"].attrs['damnit_png'] == 1
        assert f[".reduced/bar"][()] == bar

    cursor = db.conn.execute("SELECT * FROM runs")
    row = cursor.fetchone()
    thumbnail = row["foo"]
    assert isinstance(thumbnail, bytes)
    assert BlobTypes.identify(thumbnail) is BlobTypes.png
    assert row["bar"] == bar
