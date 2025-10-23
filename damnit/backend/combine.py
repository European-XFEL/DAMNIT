"""Read temporary HDF5 files, combine them into a single file per run, updating DB"""
import re
import sys
from pathlib import Path

import h5py

from .db import DamnitDB
from .extract_data import load_reduced_data, add_to_db

FRAGMENT_PATTERN = re.compile(r"p(\d+)_r(\d+).(.+).ready.h5$")

def combine(src: Path, dst: Path):
    """Combine the the contents of src (an HDF5 file) into dst"""
    # Shortcut: if the destination file doesn't exist, rename src
    try:
        dst.hardlink_to(src)
    except FileExistsError:
        pass
    else:
        src.unlink()
        return

    with h5py.File(src) as fsrc, h5py.File(dst, 'r+') as fdst:
        for grp in fsrc:
            if not grp.startswith("."):
                fdst.pop(grp, None)
                fsrc.copy(grp, fdst, grp)

        for special_grp in [".reduced", ".preview", ".errors"]:
            for k in fsrc[special_grp]:
                path = f"{special_grp}/{k}"
                fdst.pop(path, None)
                fsrc.copy(path, fdst, path)

    src.unlink()

def update_db(db: DamnitDB, proposal: int, run: int, src: Path):
    new_data = load_reduced_data(src)
    add_to_db(new_data, db, proposal, run)


def gather_all_fragments(damnit_dir: Path):
    db = DamnitDB.from_dir(damnit_dir)
    h5_dir = damnit_dir / "extracted_data"

    for p in h5_dir.iterdir():
        if not (m := FRAGMENT_PATTERN.match(p.name)):
            continue

        proposal = int(m[1])
        run = int(m[2])
        dst = h5_dir / f"p{proposal}_r{run}.h5"
        update_db(db, proposal, run, p)
        combine(p, dst)

if __name__ == "__main__":
    gather_all_fragments(Path(sys.argv[1]))
