# Internals

This page describes some internal details of DAMNIT. It will probably not be of
interest to you unless you're a developer.

## Making a new release

To make a new release you need to:

1. Bump the version number in `damnit/__init__.py`.
1. Make a git tag for the new version on the new commit. Our CI will
   automatically pick that up and publish a new release to PyPI.

## Installation

To install both the frontend and backend:
```bash
# Get the code
git clone https://github.com/European-XFEL/DAMNIT.git
cd DAMNIT

# Make an environment for DAMNIT
conda create -n damnit python

conda activate damnit
pip install '.[gui,backend]'
```

If you only need the API there's no need to clone the repo, `pip install
damnit` should be sufficient.

## Deployment on Maxwell
DAMNIT is deployed in a stable and beta module on Maxwell:
```bash
$ module load exfel damnit  # The full module name is damnit/stable
$ module load exfel damnit/beta
```

To update a module:
```bash
$ ssh xsoft@max-exfl.desy.de

# Helper command to cd into the module directory and activate its environment
$ damnitmod mid # This will activate the stable module, use `beta` for the beta module
$ git pull # Or whatever command is necessary to update the code
```

The package is installed in a pixi environment as an editable install, so
we don't normally need to reinstall it. But if we do, e.g. to add or update
dependencies, run:

```bash
$ module load exfel pixi
$ pixi install
```

## Kafka
The GUI is updated by Kafka messages sent by the backend. Currently we use
XFEL's internal Kafka broker at `exflwgs06.desy.de:9091`, but this is only
accessible inside the DESY network.

DAMNIT can run offline, but if you want updates from the backend and you're
running DAMNIT outside the network and not using a VPN, you'll first have to
forward the broker port to your machine:
```bash
ssh -L 9091:exflwgs06.desy.de:9091 max-exfl.desy.de
```

And then add a line in your `/etc/hosts` file to resolve `exflwgs06.desy.de`
to `localhost` (remember to remove it afterwards!):
```
127.0.0.1 exflwgs06.desy.de
```

And then DAMNIT should be able to use XFELs broker. If you want to use a specific
broker you can set the `AMORE_BROKER` variable:
```bash
export AMORE_BROKER=localhost:9091
```

DAMNIT will then connect to the broker at that address.

## Data storage

There are two types of storage used, both in the `usr/Shared/amore` directory of
a proposal by default:

- HDF5 files are saved in the `extracted_data/` subdirectory for each run. These
  hold the values returned by the variable functions in the context file.
- A SQLite database named `runs.sqlite` that contains things like:
    - Data entered by the user through the GUI, such as comments or editable
      variable values.
    - Summary data for all the variables. The summaries are also stored in the
      HDF5 file but they're cached in the database too so that the GUI doesn't
      have to open a bunch of HDF5 files to display the table.
    - General DAMNIT settings for things like the slurm partition to use etc.

The DAMNIT data format details the exact structure of the data in the database
and HDF5 files.

### v4 (current)

Adds a trigger to remove orphaned tags after their last reference is deleted
from `variable_tags`.

### v3

Adds `tags` and `variables_tags` tables to the database schema which stores
tags (any string) associated to variables.

### v2

This is a minor change to the database schema, adding an `attributes` column to
the `run_variables` table which contains summary values for the table.

### v1

In v1 there were a few changes to the way we store images and `xarray` types:

- Thumbnails are stored as PNG byte arrays.
- Support for `xarray.Dataset`'s was added.
- `Dataset`'s and `DataArray`'s are stored inline in the NetCDF 4 format. This
  means we get support for saving all properties of `Dataset`'s/`DataArray`'s
  for free (e.g. attributes).

The most important change was to the database schema, which moved to a 'long
narrow' format so that we don't need to change the schema whenever a new
variable is added. It should also allow for versioning variables in the future.

### v0

v0 refers to the first format, created before we started versioning at
all. Here's an example v0 HDF5 file for a single run:
```bash
p1234_r100.h5

# Start off with a group to hold the summary values

├.reduced
│ ├scalar       [float64: scalar]
│ ├ndarray      [float64: scalar]
│ ├string       [UTF-8 string: 1]
│ ├dataarray    [float64: scalar]
│ ├2d_array     [uint8: 100 × 100 × 4] # 2D arrays are treated as images
│ └rgba_image   [uint8: 75 × 150 × 4]  # Image summaries are downscaled RGBA images

# After the .reduced group we have groups for each variable

├scalar
│ └data   [int64: scalar]
├ndarray
│ ├data   [float64: 100]
├string
│ └data   [UTF-8 string: 1]
├dataarray   # The coordinates are saved, but not the dimension-coord mapping
│ ├data      [float64: 100 × 16 × 512 × 128]
│ ├dim_0     [int64: 512]
│ ├dim_1     [int64: 128]
│ ├module    [int64: 16]
│ └pulseId   [int64: 300]
├2d_array
│ └data   [float64: 1352 × 1196]
├rgba_image
│ └data   [uint8: 600 × 1200 × 4]
```

TL;DR:

- There's a group for all the summaries, and then a group per-variable for the
  object returned from the variable function.
- The 'main value' of each variable is always stored in the `data` dataset in
  the variables group. This is only relevant for DataArray's, which can have
  multiple datasets.
- 2D arrays are treated as images, and their summary is an RGBA thumbnail.

When it comes to the SQLite database schema, the most important thing to know is
that there was one big `runs` table:

```
proposal | runnr | start_time        | added_at         | comment            | var1               | var2               | var3     | var4            |
---------|-------|-------------------|------------------|--------------------|--------------------|--------------------|----------|-----------------|
3422     | 1     | 1683091429.382994 | 1683098801.769   | agipd dark         | 300.5509948730469  | 1.666236494202166… | 608      | 0.161871538018  |
3422     | 2     | 1683091513.36196  | 1683098860.364   | agipd dark         | 300.5509948730469  | 1.672673170105554… | 607      | 0.160357959558  |
3422     | 3     | 1683091596.528593 | 1683098974.7     | agipd dark         | 300.5509948730469  | 1.676370447967201… | 605      | 0.160361199298  |
3422     | 4     | 1683096460.784313 | 1683103844.035   | agipd dark         | 300.5509948730469  | 1.72180516528897e… | 603      | 0.159418800086  |
3422     | 5     | 1683096542.744437 | 1683103901.837   | agipd dark         | 300.5509948730469  | 1.722106935631018… | 601      | 0.159602915718  |
3422     | 6     | 1683096624.703127 | 1683103959.994   | agipd dark         | 300.5509948730469  | 1.722659908409696… | 607      | 0.161150277002  |
3422     | 7     | 1683099077.949262 | 1683106416.386   | dark               | 300.5509948730469  | 1.696372237347532… | 596      | 0.167543028426  |
3422     | 8     | 1683099158.289963 | 1683106530.607   | dark               | 300.5509948730469  | 1.693211743258871… | 600      | 0.169153788618  |
3422     | 9     | 1683099239.020509 | 1683106603.581   | dark               | 300.5509948730469  | 1.694534876151010… | 596      | 0.167274568394  |
3422     | 10    | 1683109651.413693 | 1683117096.313   |                    | 300.551025390625   | 1.726133086776826… | 1578     | 0.447119904602  |
3422     | 11    | 1683109907.527204 | 1683117352.791   | Knife edge scan    | 300.551025390625   | 1.722175329632591… | 1564     | 0.459057441431  |
```

And every time a variable was added another column would be added to the table
by changing its schema. There are a few other tables in the database but they're
not so important.
