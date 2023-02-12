from collections import defaultdict

from .backend.db import get_meta
from .backend.extract_data import add_to_db
from .ctxsupport.ctxrunner import generate_thumbnail, add_to_h5_file

def migrate_images(db, db_dir):
    proposal = get_meta(db, "proposal", None)
    if proposal is None:
        raise RuntimeError("Database must have a proposal configured for it to be migrated.")

    reduced_data = defaultdict(dict)
    files = list((db_dir / "extracted_data").glob("*.h5"))
    n_files = len(files)
    suffix = "s" if n_files > 1 or n_files == 0 else ""

    print(f"Looking through {n_files} HDF5 file{suffix}...")

    # First we modify all of the image summaries in the HDF5 files
    for h5_path in files:
        with add_to_h5_file(h5_path) as f:
            if ".reduced" in f:
                reduced = f[".reduced"]
                run = int(h5_path.stem.split("_")[1][1:])

                for ds_name in reduced:
                    if reduced[ds_name].ndim == 2:
                        # Generate a new thumbnail
                        image = reduced[ds_name][()]
                        image = generate_thumbnail(image)
                        reduced_data[run][ds_name] = image

                        # Overwrite the dataset
                        del reduced[ds_name]
                        reduced.create_dataset(ds_name, data=image)

    # And then update the summaries in the database
    for run, run_reduced_data in reduced_data.items():
        add_to_db(run_reduced_data, db, proposal, run)

    print(f"Migration completed successfully, updated {len(reduced_data)} run{suffix}.")
