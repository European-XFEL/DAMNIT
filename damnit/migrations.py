from collections import defaultdict

import h5py
import numpy as np
import xarray as xr

from .backend.db import DamnitDB, DB_NAME
from .backend.extract_data import add_to_db, ReducedData
from .ctxsupport.ctxrunner import generate_thumbnail, add_to_h5_file, DataType


def migrate_images(db, db_dir, dry_run):
    proposal = db.metameta.get("proposal")
    if proposal is None:
        raise RuntimeError("Database must have a proposal configured for it to be migrated.")

    reduced_data = defaultdict(dict)
    files = list((db_dir / "extracted_data").glob("*.h5"))
    n_files = len(files)
    files_modified = set()
    suffix = "s" if n_files > 1 or n_files == 0 else ""

    print(f"Looking through {n_files} HDF5 file{suffix}...")

    # First we modify all of the image summaries in the HDF5 files
    for h5_path in files:
        with add_to_h5_file(h5_path) as f:
            if ".reduced" in f:
                reduced = f[".reduced"]
                run = int(h5_path.stem.split("_")[1][1:])

                for ds_name, dset in reduced.items():
                    if dset.ndim == 2 or (dset.ndim == 3 and dset.shape[2] == 4):
                        # Generate a new thumbnail
                        image = reduced[ds_name][()]
                        image = generate_thumbnail(image)
                        reduced_data[run][ds_name] = ReducedData(image)
                        files_modified.add(h5_path)

                        if not dry_run:
                            # Overwrite the dataset
                            del reduced[ds_name]
                            dset = reduced.create_dataset(
                                ds_name, data=np.frombuffer(image.data, dtype=np.uint8)
                            )
                            dset.attrs['damnit_png'] = 1

    # And then update the summaries in the database
    for run, run_reduced_data in reduced_data.items():
        if not dry_run:
            add_to_db(run_reduced_data, db, proposal, run)

    info = f"updated {len(reduced_data)} variables in {len(files_modified)} files"
    if dry_run:
        print(f"Dry run: would have {info}.")
    else:
        print(f"Migration completed successfully, {info}")

def dataarray_from_group(group):
    data = group["data"][()]
    coords = { ds_name: group[ds_name][()] for ds_name in group.keys()
               if ds_name != "data" }

    # Attempt to map the coords to the right dimensions. This
    # will fail if there are two coordinates/dimensions with the
    # same length.
    coord_sizes = { len(coord_data): coord for coord, coord_data in coords.items() }
    if len(set(coord_sizes.keys())) != len(coord_sizes.keys()):
        return None

    dims = [coord_sizes[dim] if dim in coord_sizes else f"dim_{i}"
            for i, dim in enumerate(data.shape)]
    return xr.DataArray(data,
                        dims=dims,
                        coords={ dim: coords[dim] for dim in dims
                                 if dim in coords })


def migrate_dataarrays(db, db_dir, dry_run):
    files = list((db_dir / "extracted_data").glob("*.h5"))
    n_files = len(files)
    suffix = "s" if n_files > 1 or n_files == 0 else ""

    would = "(would) " if dry_run else ""

    print(f"Looking through {n_files} HDF5 file{suffix}...")

    files_modified = set()
    total_groups = 0
    for h5_path in files:
        groups_to_replace = []
        with h5py.File(h5_path, "a") as f:
            for name, grp in f.items():
                if name == '.reduced':
                    continue

                if isinstance(grp, h5py.Group) and 'data' in grp and len(grp) > 1:
                    dataarray = dataarray_from_group(grp)
                    if dataarray is None:
                        raise RuntimeError(
                            f"Error: could not convert v0 array for '{name}' to a DataArray automatically"
                        )
                    groups_to_replace.append((name, dataarray))

            for name, _ in groups_to_replace:
                print(f"{would}Delete {name} in {h5_path.relative_to(db_dir)}")
                if not dry_run:
                    f[name].clear()
                    f[name].attrs.clear()
                    f[name].attrs['_damnit_objtype'] = DataType.DataArray.value

        for name, arr in groups_to_replace:
            print(f"{would}Save {name} in {h5_path.relative_to(db_dir)}")
            if not dry_run:
                arr.to_netcdf(h5_path, mode="a", format="NETCDF4", group=name, engine="h5netcdf")

            files_modified.add(h5_path)
            total_groups += 1

    print(("(would have) " if dry_run else "") +
          f"Modified {total_groups} groups in {len(files_modified)} files")
    if dry_run:
        print("Dry run - no files were changed")


def main_dataset(grp: h5py.Group):
    candidates = {name for (name, dset) in grp.items()
                  if dset.attrs.get('CLASS', b'') != b'DIMENSION_SCALE'}
    if len(candidates) == 1:
        return grp[candidates.pop()]


def migrate_v0_to_v1(db, db_dir, dry_run):
    """
    For reference, see the V0_SCHEMA variable in db.py.
    In the v1 schema, the runs table was deleted and replaced with a view of
    the new run_variables table. The run_info table also needs to be created,
    but that can be done by executing the v1 schema.
    """
    migrate_dataarrays(db, db_dir, dry_run)

    # Get all column and variable names
    column_names = [rec[0] for rec in
                    db.conn.execute("SELECT name FROM PRAGMA_TABLE_INFO('runs')").fetchall()]
    variable_names = [name for name in column_names
                      if name not in ["proposal", "runnr", "start_time", "added_at", "comment"]]
    #
    # And then read all run data. This is what we'll need to copy into the new
    # `run_variables` table.
    runs = [rec for rec in db.conn.execute("SELECT * FROM runs").fetchall()]

    print(f"Found {len(runs)} runs, with these variables:")
    print(variable_names)
    print()

    # Scan HDF5 files to get timestamps (from mtime) & max diff for 1D arrays
    timestamps = {}  # keys (proposal, run)
    max_diffs = {}  # keys (proposal, run, variable)
    for record in runs:
        proposal = record["proposal"]
        run_no = record["runnr"]
        h5_path = db_dir / f"extracted_data/p{proposal}_r{run_no}.h5"

        if not h5_path.exists():
            print(f"Skipping variables for run {run_no} because {h5_path} does not exist")
            continue

        timestamps[(proposal, run_no)] = h5_path.stat().st_mtime

        with h5py.File(h5_path, "a") as f:
            for name in variable_names:
                if name not in f:
                    continue

                if (ds := main_dataset(f[name])) is None:
                    continue

                if 'max_diff' in ds.attrs:
                    max_diffs[(proposal, run_no, name)] = ds.attrs['max_diff'].item()
                elif ds.ndim == 1 and np.issubdtype(ds.dtype, np.number):
                    data = ds[()]
                    max_diff = abs(np.nanmax(data) - np.nanmin(data)).item()
                    max_diffs[(proposal, run_no, name)] = max_diff
                    ds.attrs['max_diff'] = max_diff

    print(f"Found max difference for {len(max_diffs)} variables")

    new_db_path = db_dir / "runs.v1.sqlite"
    new_db_path.unlink(missing_ok=True)  # Clear any previous attempt
    new_db = DamnitDB(new_db_path)
    for k, v in db.metameta.items():
        if k != "data_format_version":
            new_db.metameta[k] = v

    # Load the data into the new database
    total_vars = 0
    with new_db.conn:
        for record in runs:
            run = dict(zip(column_names, record))
            proposal = run["proposal"]
            run_no = run["runnr"]

            # Add the run info to the `run_info` table
            new_db.conn.execute("""
                INSERT INTO run_info
                VALUES (:proposal, :runnr, :start_time, :added_at)
                """, run)

            for name in variable_names:
                value = record[name]
                if value is None:
                    continue

                max_diff = max_diffs.get((proposal, run_no, name))
                timestamp = timestamps.get((proposal, run_no))

                variable = {
                    "proposal": proposal,
                    "run": run_no,
                    "name": name,
                    "version": 1,
                    "value": value,
                    "timestamp": timestamp,
                    "max_diff": max_diff
                }
                new_db.conn.execute("""
                    INSERT INTO run_variables (proposal, run, name, version, value, timestamp, max_diff)
                    VALUES (:proposal, :run, :name, :version, :value, :timestamp, :max_diff)
                    """, variable)
                total_vars += 1

        # And now that we're done, we need to recreate the `runs` view
        new_db.update_views()

    new_db.close()
    db.close()

    if dry_run:
        print(f"Dry-run: new format DB created at {new_db_path.name}")
        print("If all seems OK, re-run the migration without --dry-run.")
    else:
        db_path = db_dir / DB_NAME
        backup_path = db_dir / "runs.v0-backup.sqlite"
        db_path.rename(backup_path)
        new_db_path.rename(db_path)
        print(f"New format DB created and moved to {db_path.name}")
        print(f"Old database backed up as {backup_path.name}")
