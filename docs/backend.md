# The backend and context file
The backend runs a context file like this one, which contains functions that
it will execute:
```python title="context.py" linenums="1"
import numpy as np
from damnit_ctx import Variable

from extra.components import XGM, XrayPulses

@Variable(title="XGM intensity [uJ]", summary="mean")
def xgm_intensity(run):
    """
    Mean XGM intensity per-train.
    """
    return XGM(run).pulse_energy().mean("pulseIndex")

@Variable(title="Pulses", summary="mean")
def pulses(run):
    """
    Number of pulses in the run.
    """
    return XrayPulses(run).pulse_counts().to_xarray()
```

By convention it's stored under the `usr/Shared/amore` directory of a proposal,
along with other files that DAMNIT creates like the SQLite database and the HDF5
files that are created for each run.

## `@Variable`'s
Functions in the context file can be decorated with `@Variable` to denote that
these are variables to be executed for each run. The `@Variable` decorator takes
these arguments:

- `title` (string): title displayed for the variable's column.
- `tags` (string or list of strings): tags to categorize and filter variables.
  Tags can be used to group related variables and filter the table view:
  ```python
  @Variable(title="AGIPD data", tags=["detector", "agipd", "raw"])
  def agipd_data(run):
      ...
  ```
- `summary` (string): if the function returns an array, then `summary` will be used to
  reduce it to a single number. Internally it gets mapped to `np.<summary>()`,
  so you can use e.g. `sum` or `nanmean` to compute the summary with `np.sum()`
  or `np.nanmean()` respectively.
- `data` (string): this sets the trigger for the variable. By default
  `Variable`'s have `data="raw"`, which means they will be triggered by a
  migration of raw data to the offline cluster. But if you want to process
  detector data which requires calibration, then you'll want to set
  `data="proc"` to tell DAMNIT to run that `Variable` when the calibration
  pipeline finishes processing the run:
  ```python
  @Variable(title="Detector preview", data="proc")
  def detector_preview(run):
      ...
  ```
- `cluster` (bool): whether or not to execute this variable in a Slurm job. This
  should always be used if the variable does any heavy processing.
- `transient` (bool): do not save the variable's result to the database. This
  is useful for e.g. intermediate results to be reused by other Variables. Since
  their data isn't saved, `transient` variables can return any object. By
  default Variables do save their results (`transient=False`).

Variable functions can return any of:

- Scalars
- Lists of scalars
- Multi-dimensional `numpy.ndarray`'s or `xarray.DataArray`'s (2D arrays will be
  treated as images)
- `xarray.Dataset`'s
- Matplotlib `Figure`s or `Axes` (will be saved as 2D images).
- Plotly figures (will be saved as JSON so that the GUI can display them in an
  interactive plot).
- Strings
- `None`

The functions must always take in one argument, `run`, to which is passed a
[`DataCollection`](https://extra-data.readthedocs.io/en/latest/reading_files.html#data-structure)
of the data in the run. In addition, a function can take some other special
arguments if they have the right _annotations_, currently.  
`meta` accesses internal arguments:

- `meta#run_number`: The number of the current run being processed.
- `meta#proposal`: The number of the current proposal.
- `meta#proposal_path`: The root
  [Path](https://docs.python.org/3/library/pathlib.html) to the current
  proposal.

`mymdc` requests information from the EuXFEL data management portal
[MyMDC](https://in.xfel.eu/metadata/):

- `mymdc#run_type`: The run type from myMdC.
- `mymdc#sample_name`: The sample name from myMdC.
- `mymdc#techniques`: list of
  [technique](https://expands-eu.github.io/ExPaNDS-experimental-techniques-ontology/index-en.html)
  associated with the run. Each technique listed is a `dict` containing the
  following keys: `description`, `flg_available`, `id`, `identifier`, `name`,
  `runs_techniques_id`, `url`.

You can also use annotations to express a dependency between `Variable`'s using
the `var#<name>` annotation:
```python
@Variable(title="foo")
def foo(run, run_no: "meta#run_number"):
    # Just return the run number
    return run_no
    
@Variable(title="bar")
def bar(run, value: "var#foo"):
    # Now bar() will be executed after foo(), and we can use its return value
    return value * 2
```

Dependents are not executed if a variable raises an error or returns `None`. You
can raise `Skip` to provide a reason, which will be visible as a tooltip on the
table cell in the GUI:

```python
from damnit_ctx import Variable, Skip

@Variable()
def binned_by_scan_step(run):
    scan = Scantool(run)
    if not scan.active:
        raise Skip("Run is not a scan")
    ...
```

Dependencies with default values are also allowed, the default value will be
passed to the function if the dependency did not complete execution for some
reason:
```python
@Variable(title="baz")
def baz(run, value: "var#foo"=42):
    # This will return the result of foo() if foo() succeeded, otherwise 42
    return value
```

Variable functions can use up to 4 CPU cores and 25 GB of RAM by default.
If more resources are needed, use `cluster=True` (see the [Using
Slurm](backend.md#using-slurm) section) to access all
of the cores & memory of an assigned cluster node. If required, you can also
change the limits for non-cluster variables:

```bash
# Allow 8 CPU cores
$ damnit db-config noncluster_cpus 8

# Allow 50 GB memory
$ damnit db-config noncluster_mem 50G
```

## `VariableGroup`
For more complex or reusable sets of analyses, you can group related variables
together using the `VariableGroup` class. This allows you to create
self-contained components that can be configured and instantiated multiple
times.

A `VariableGroup` is a class that inherits from `damnit_ctx.VariableGroup` and
contains methods decorated with `@Variable`.

```python title="context.py"
from damnit_ctx import Variable, VariableGroup

class XGMDiagnostics(VariableGroup):

    @Variable(title="Pulse Energy", summary="mean")
    def pulse_energy(self, run, device_name: str):
        return XGM(run, device_name).pulse_energy()

    @Variable(title="Corrected Energy", summary="mean")
    def corrected_energy(self, run, energy: "var#pulse_energy", offset: float = 0.0):
        # This has an intra-group dependency on the 'pulse_energy' variable
        return energy + offset

# Instantiate the group in your context file
xgm_sase2 = XGMDiagnostics(
    "XGM SA2",
    device_name="SA2_XTD6_XGM/XGM/DOOCS",
    offset=1.1,
    tags=["XGM", "SA2"]
)
xgm_hed = XGMDiagnostics(
    "XGM HED",
    device_name="HED_XTD9_XGM/XGM/DOOCS",
    offset=0.9,
    tags=["XGM", "HED"]
)
```

### Naming and Titles
When you create an instance of a `VariableGroup` (e.g., `xgm_sase2`), the
variables within it are automatically given prefixed names to avoid conflicts.
- The **variable name** (the Python identifier) is formed by joining the
  instance name and the method name with a double underscore:
  `xgm_sase2__pulse_energy`.
- The **variable title** (for display in the GUI) is formed by joining the
  group's title and the variable's title with a separator (default is `/`):
  `XGM SA2/Pulse Energy`.

### Dependencies
`VariableGroup` simplifies dependency management:
- **Intra-group dependencies:** To depend on another variable within the *same
  group instance*, just use its method name (e.g., `var#pulse_energy`). DAMNIT
  automatically rewrites this to the correct full name
  (`var#xgm_sase2__pulse_energy`).
- **Global dependencies:** To depend on a top-level variable outside of any
  `VariableGroup`, you must use the `_root.` prefix to avoid ambiguity. This
  explicitly tells DAMNIT to look in the global scope.

```python
@Variable(title="Global Offset")
def global_offset(run):
    return 42

class Group(VariableGroup):
    @Variable
    def local_var(self, run, offset: "var#_root.global_offset"):
        # Correctly depends on the top-level global_offset
        return 10 + offset

instance = Group("Group")
```

!!! warning
    A dependency on a variable outside the group (e.g., `var#global_offset`)
    without the `_root.` prefix will fail at load time if a local variable of
    the same name does not exist. The `_root.` prefix is required for all
    non-local dependencies.

- **Cross-group dependencies:** To use the result of one group instance in
  another, you must create a top-level "composer" variable that depends on the
  fully-qualified names of the variables from each group.

```python
# Continuing the XGMDiagnostics example...
@Variable(title="SASE2 vs HED")
def xgm_comparison(run, sa2_energy: "var#xgm_sase2__corrected_energy",
                        hed_energy: "var#xgm_hed__corrected_energy"):
    return hed_energy / sa2_energy
```

### Composition (Nesting Groups)

You can build more complex structures by nesting `VariableGroup` instances
inside other groups. This allows you to compose small, focused analysis
components into a larger, hierarchical system.

```python
class Detector(VariableGroup):
    @Variable(title="Photon Count")
    def n_photons(self, run, det_name: str):
        return run.alias[det_name]['photon-count'].xarray()

# MIDDiagnostics composes XGMDiagnostics and Detector
class MIDDiagnostics(VariableGroup):
    # Nested instances of other groups
    xgm = XGMDiagnostics(
        "XGM SA2",
        device_name="SA2_XTD6_XGM/XGM/DOOCS",
        offset=1.1,
        tags=["XGM", "SA2"]
    )
    agipd = Detector("AGIPD", det_name="AGIPD")

    # This variable can depend on children of the nested groups
    @Variable(title="Photons per µJ")
    def photons_per_microjoule(self, run,
                               photons: "var#agipd.n_photons",
                               energy: "var#xgm.corrected_energy"):
        return photons / energy

# Instantiate the top-level group
mid = MIDDiagnostics("MID Diagnostics", tags=["Diag"])
```

When groups are nested:
- **Naming and Titles:** Prefixes are applied recursively. A variable
  `n_photons` inside the `agipd` instance, which is inside the `mid` instance,
  will have the name `mid__agipd__n_photons` and the title `MID
  Diagnostics/AGIPD/Photon Count`.
- **Dependencies:** The dependency syntax remains the same. To depend on a
  variable within a nested group from a sibling, you use its path from the
  current group's perspective, joined by double underscores (e.g.,
  `var#agipd__n_photons`) or a dot (e.g. `var#agipd.n_photons`).
- **Property Inheritance:**
    - `tags`: Are inherited recursively. In the example above, the `Diag` tag
      from the `MIDDiagnostics` instance will be applied to all variables inside
      it, including those from the nested `xgm` and `agipd` groups.
    - `cluster` and `transient`: These properties are **not** inherited by
      subgroups. They only apply to the direct `@Variable`s defined within that
      specific group class. This ensures that a high-level grouping doesn't
      unintentionally mark a low-level, lightweight variable for heavy cluster
      processing.

!!! note "Dependency syntax limitation"
  
    Dot-notation currently only works inside a VariableGroup, if your Variable
    outside a VariableGroup depends on a Variable inside a VariableGroup,
    you'll have to use the double-underscore notation.

### Inheritance
`VariableGroup` also supports standard Python class inheritance. A group can
inherit from a base class and will automatically include all `@Variable`s from
its parent(s), allowing you to create common, reusable sets of analyses.

```python
class BaseAnalysis(VariableGroup):
    @Variable(title="Train Count")
    def n_trains(self, run):
        return len(run.train_ids)

class DetectorAnalysis(BaseAnalysis):  # Inherits n_trains
    @Variable(title="Photon Count", data="proc")
    def photon_count(self, run, n_trains: "var#n_trains"):
        # Depends on an inherited variable
        return 1e6 / n_trains

detector = DetectorAnalysis("Detector")
# This instance will have two variables: detector__n_trains and detector__photon_count
```

## Cell
The `Cell` object is a versatile container that allows customizing how data is
stored and displayed in the table. When writing [Variables](#variables), you can
return a `Cell` object to control both the full data storage and its summary
representation. A `Cell` takes these arguments:

- `data`: The main data to store
- `summary`: Function name (as string) from the `numpy` module to compute
  summary from data (e.g., 'mean', 'std')
- `summary_value`: Direct value to use as summary (number or string)
- `bold`: A boolean indicating whether the text should be rendered in a bold
  font in the table's cell
- `background`: Cell background color as hex string (e.g. `'#ffcc00'`)
  or RGB sequence (0-255 values)
- `preview`: What to show in a pop-up when the cell is double clicked.
  This can be a 1D or 2D array, or a Matplotlib or Plotly figure.
  If `data` is one of these types, it doesn't need to be specified again.

Example Usage:

```python
@Variable(title="Peaks")
def peaks(run):
    success, counts, data = computation(run)
    return Cell(
        data=data,
        summary_value=f"{counts} peaks detected" if success else "No peak",
        bold=True,
        background="#7cfc00" if success else "#ff0000"
    )
```

## Using Slurm
As mentioned in the previous section, variables can be marked for execution in a
Slurm job with the `cluster=True` argument to the decorator:
```python
@Variable(title="Foo", cluster=True)
def foo(run):
    # some heavy computation ...
    return 42
```

This should work out-of-the-box with no other configuration needed. By default
DAMNIT will figure out an appropriate partition that user has access to, but
that can be overridden by explicitly setting a partition or reservation:
```bash
# Set a reservation
$ damnit db-config slurm_reservation upex_001234

# Set a partition
$ damnit db-config slurm_partition allgpu
```

If both `slurm_reservation` and `slurm_partition` are set, the reservation will
be chosen. The jobs will be named something like `r42-p1234-damnit` and both
stdout and stderr will be written to the run's log file in the `process_logs/`
directory.

!!! warning

    Make sure to delete the reservation setting after the reservation has
    expired, otherwise Slurm jobs will fail to launch.

    ```bash
    $ damnit db-config slurm_reservation --delete
    ```

## Using custom environments
DAMNIT supports running the context file in a user-defined Python environment,
which is handy if there's a certain package you want that's only installed in
a certain environment. At some point the setting for this will be exposed
in the GUI, but right now you'll have to change it on the command line by
passing the path to the `python` executable of the required environment:
```bash
$ damnit db-config context_python /path/to/your/python
```

The environment *must* have these dependencies installed for DAMNIT to work:

- `extra_data`
- `pyyaml`
- `requests`

If your variables return [plotly](https://plotly.com/python/) plots, the
environment must also have the `kaleido` package.

## Managing the backend
The backend is a process running under [Supervisor](http://supervisord.org/). In
a nutshell:

- Supervisor will manage the backend using a configuration file named
  `supervisord.conf` stored in the database directory. It's configured to listen
  for commands over HTTP on a certain port with a certain
  username/password. Supervisor will save its logs to `supervisord.log`.
- It can be controlled with `supervisorctl` on any machine using the same config
  file.

So lets say you're running the GUI on FastX, and the backend is now started. If
you open a terminal and cd to the database directory you'll see:
```bash
$ cd /gpfs/path/to/proposal/usr/Shared/amore
$ ls
amore.log  context.py  extracted_data/  runs.sqlite  supervisord.conf supervisord.log
```

You could get the status of the backend with:
```bash
$ supervisorctl -c supervisord.conf status damnit
damnit                           RUNNING   pid 3793870, uptime 0:00:20
```

And you could restart it with:
```bash
$ supervisorctl -c supervisord.conf restart damnit
damnit: stopped
damnit: started

$ supervisorctl -c supervisord.conf status damnit
damnit                           RUNNING   pid 3793880, uptime 0:00:04
```

## Starting from scratch
Sometimes it's useful to delete all of the data so far and start from
scratch. As long as you have the context file this is safe, with the caveat that
comments and user-editable variables _cannot_ be restored.

The steps to delete all existing data are:

1. `rm runs.sqlite` to delete the database used by the GUI.
1. `rm -rf extracted_data/` to delete the HDF5 files created by the backend.
1. `damnit proposal 1234` to create a blank database for the given
   proposal.

And then you can reprocess runs with `damnit reprocess` to restore
their data.
