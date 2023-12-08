from collections import defaultdict

import h5py
import numpy as np
import xarray as xr

from .backend.db import V1_SCHEMA
from .backend.extract_data import add_to_db, ReducedData
from .ctxsupport.ctxrunner import generate_thumbnail, add_to_h5_file, DataType


def migrate_images(db, db_dir):
    proposal = db.metameta.get("proposal")
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
                        reduced_data[run][ds_name] = ReducedData(image)

                        # Overwrite the dataset
                        del reduced[ds_name]
                        dset = reduced.create_dataset(
                            ds_name, data=np.frombuffer(image.data, dtype=np.uint8)
                        )
                        dset.attrs['damnit_png'] = 1

    # And then update the summaries in the database
    for run, run_reduced_data in reduced_data.items():
        add_to_db(run_reduced_data, db, proposal, run)

    print(f"Migration completed successfully, updated {len(reduced_data)} run{suffix}.")

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

def migrate_v0_to_v1(db, db_dir, dry_run):
    """
    For reference, see the V0_SCHEMA variable in db.py.
    In the v1 schema, the runs table was deleted and replaced with a view of
    the new run_variables table. The run_info table also needs to be created,
    but that can be done by executing the v1 schema.
    """
    arrays_to_save = defaultdict(list)

    # Begin a transaction, anything in here should be rolled back if there's an exception
    with db.conn:
        # Get all column and variable names
        column_names = [rec[0] for rec in
                        db.conn.execute("SELECT name FROM PRAGMA_TABLE_INFO('runs')").fetchall()]
        variable_names = [name for name in column_names
                          if name not in ["proposal", "runnr", "start_time", "added_at", "comment"]]

        # And then read all run data. This is what we'll need to copy into the new
        # `run_variables` table.
        runs = [rec for rec in db.conn.execute("SELECT * FROM runs").fetchall()]

        print(f"Found {len(runs)} runs, with these variables:")
        print(variable_names)
        print()

        # Now delete the runs table
        if not dry_run:
            db.conn.execute("DROP TABLE runs")
        else:
            print("Would delete the `runs` table")

        # Execute the v1 schema to create all the new tables
        if not dry_run:
            db.conn.executescript(V1_SCHEMA)
        else:
            print("Would execute the v1 schema")

        # And re-add the data from the old `runs` table
        if not dry_run:
            for record in runs:
                run = dict(zip(column_names, record))
                proposal = run["proposal"]
                run_no = run["runnr"]
                h5_path = db_dir / f"extracted_data/p{proposal}_r{run_no}.h5"

                # Add the run info to the `run_info` table
                db.conn.execute("""
                INSERT INTO run_info
                VALUES (:proposal, :runnr, :start_time, :added_at)
                """, run)

                if not h5_path.exists():
                    print(f"Skipping variables for run {run_no} because {h5_path} does not exist")
                    continue

                h5_mtime = h5_path.stat().st_mtime

                # And then add each variable
                with h5py.File(h5_path) as f:
                    variables = { name: value for name, value in run.items()
                                  if name in variable_names }
                    for name, value in variables.items():
                        if name not in f:
                            continue

                        ds = f[name]["data"]
                        reduced_ds = f[".reduced"][name]
                        max_diff = None

                        if ds.ndim > 0:
                            if reduced_ds.ndim != 3:
                                has_coords = len(f[name].keys()) > 1

                                if ds.ndim == 1 and ds.dtype != bool:
                                    data = ds[()]
                                    max_diff = abs(np.nanmax(data) - np.nanmin(data)).item()

                                if has_coords:
                                    dataarray = dataarray_from_group(f[name])
                                    if dataarray is None:
                                        raise RuntimeError(f"Error: could not convert v0 array for '{name}' to a DataArray automatically")
                                    else:
                                        arrays_to_save[h5_path].append((dataarray, name))

                        else:
                            raise RuntimeError(f"Could not guess a type for '{name}' in {h5_path}")

                        variable = {
                            "proposal": proposal,
                            "run": run_no,
                            "name": name,
                            "version": 1,
                            "value": value,
                            "timestamp": h5_mtime,
                            "max_diff": max_diff
                        }
                        db.conn.execute("""
                        INSERT INTO run_variables (proposal, run, name, version, value, timestamp, max_diff)
                        VALUES (:proposal, :run, :name, :version, :value, :timestamp, :max_diff)
                        """, variable)

            # And now that we're done, we need to recreate the `runs` view
            db.update_views()

            # And set the new version
            db.metameta["data_format_version"] = 1
        else:
            print("Would copy data into `run_info` and `run_variables`")

    # First we go through and delete the old groups
    for h5_path, array_names in arrays_to_save.items():
        with h5py.File(h5_path, "a") as f:
            for _, name in array_names:
                del f[name]

    for h5_path, array_names in arrays_to_save.items():
        for array, name in array_names:
            array.to_netcdf(h5_path, mode="a", format="NETCDF4", group=name, engine="h5netcdf")

    print("Done")
