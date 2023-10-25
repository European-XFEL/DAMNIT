import shutil

from .db import DamnitDB


def copy_and_reduce(db_dir, dest_dir, runs):
    if dest_dir.exists():
        raise RuntimeError(f"Destination {dest_dir.absolute()} already exists, please delete it before continuing")

    # Create the reduced directory
    dest_dir.mkdir(parents=True)

    # Copy everything but the HDF5 files into it
    print("Copying files...")
    for path in db_dir.iterdir():
        if path.absolute() != dest_dir.absolute() and path.name != "extracted_data":
            dest_path = dest_dir / path.relative_to(db_dir)

            try:
                if path.is_dir():
                    shutil.copytree(path, dest_path)
                else:
                    shutil.copy2(path, dest_path)
            except OSError as e:
                print(f"Couldn't copy {path}: {e}")

    # Remove runs from the database
    db = DamnitDB.from_dir(dest_dir)
    version = db.metameta.get("data_format_version", 0)
    proposal = db.metameta["proposal"]
    all_runs = set(db.get_runs())
    unselected_runs = all_runs - set(runs)

    print(f"Deleting {len(unselected_runs)} runs from the database...")
    for run in unselected_runs:
        with db.conn:
            if version == 0:
                db.conn.execute("DELETE FROM runs WHERE proposal=? AND runnr=?",
                                (proposal, run))
            else:
                db.conn.execute("DELETE FROM run_info WHERE proposal=? AND run=?",
                                (proposal, run))
                db.conn.execute("DELETE FROM run_variables WHERE proposal=? AND run=?",
                                (proposal, run))

    # And copy the selected HDF5 files
    data_dir = db_dir / "extracted_data"
    dest_data_dir = dest_dir / "extracted_data"
    dest_data_dir.mkdir(exist_ok=True)

    print(f"Copying up to {len(runs)} HDF5 files...")
    for run in runs:
        original_h5_path = data_dir / f"p{proposal}_r{run}.h5"
        if original_h5_path.is_file():
            shutil.copy2(original_h5_path, dest_data_dir / original_h5_path.relative_to(data_dir))
