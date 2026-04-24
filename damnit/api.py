import json
import os
import os.path as osp
from contextlib import contextmanager
from enum import Enum
from glob import iglob
from pathlib import Path

import h5py
import numpy as np

from .backend.db import BlobTypes, DamnitDB, blob2complex, blob2numpy
from .util import isinstance_no_import


# This is a copy of damnit.ctxsupport.ctxrunner.DataType, purely so that we can
# avoid the dependencies of the runner in the API (namely requests and pyyaml).
class DataType(Enum):
    DataArray = "dataarray"
    Dataset = "dataset"
    Series = "series"
    DataFrame = "dataframe"
    Image = "image"
    Timestamp = "timestamp"
    PlotlyFigure = "PlotlyFigure"

# Also copied, this time from extra_data.read_machinery
def find_proposal(propno):
    root_dir = os.environ.get('XFEL_DATA_ROOT', '/gpfs/exfel/exp')
    dir_name = f"p{propno:06}"
    for d in iglob(osp.join(root_dir, '*/*/{}'.format(dir_name))):
        return Path(d)

    raise FileNotFoundError("Couldn't find proposal dir for {!r}".format(propno))


def _decode_dataframe_columns(cols):
    # pandas needs hashable labels (e.g. MultiIndex tuple columns).
    # But tuples are saved as list in JSON.
    def convert(item):
        if isinstance(item, list):
            return tuple(convert(v) for v in item)
        return item

    return [convert(c) for c in cols]


class VariableData:
    """Represents a variable for a single run.

    Don't create this object yourself, index a [Damnit][damnit.api.Damnit] or
    [RunVariables][damnit.api.RunVariables] object instead.
    """

    def __init__(self, name: str, title: str,
                 proposal: int, run: int,
                 h5_path: Path, data_format_version: int,
                 db: DamnitDB, db_only: bool):
        self._name = name
        self._title = title
        self._proposal = proposal
        self._run = run
        self._h5_path = h5_path
        self._data_format_version = data_format_version
        self._db = db
        self._db_only = db_only

    @property
    def name(self) -> str:
        """The variable name."""
        return self._name

    @property
    def title(self) -> str:
        """The variable title (defaults to the name if not set explicitly)."""
        return self._title

    @property
    def proposal(self) -> int:
        """The proposal to which the variable belongs."""
        return self._proposal

    @property
    def run(self) -> int:
        """The run to which the variable belongs."""
        return self._run

    @property
    def file(self) -> Path:
        """The path to the HDF5 file for the run.

        Note that the data for user-editable variables will not be stored in the
        HDF5 files.
        """
        return self._h5_path

    @property
    def units(self) -> str | None:
        """Physical units metadata for this variable, if available."""
        return self._summary_attributes().get("units")

    def type_hint(self):
        """Type hint for this variable data.

        one of ``DataType`` or None.
        """
        with self._open_h5_group() as group:
            return self._type_hint(group)

    @contextmanager
    def _open_h5_group(self):
        with h5py.File(self._h5_path) as f:
            yield f[self.name]

    @staticmethod
    def _type_hint(group):
        hint_s = group.attrs.get('_damnit_objtype', '')
        if hint_s:
            return DataType(hint_s)
        return None

    def _summary_attributes(self):
        row = self._db.conn.execute("""
            SELECT attributes FROM run_variables
            WHERE proposal=? AND run=? AND name=?
            ORDER BY version DESC LIMIT 1
        """, (self.proposal, self.run, self.name)).fetchone()

        if row is None or row[0] is None:
            return {}

        return json.loads(row[0])

    def _read_netcdf(self, one_array=False, strip_internal_attrs=True):
        import xarray as xr
        load = xr.load_dataarray if one_array else xr.load_dataset
        obj = load(self._h5_path, group=self.name, engine="h5netcdf")
        if strip_internal_attrs:
            # Remove internal attributes from loaded object
            obj.attrs = {k: v for (k, v) in obj.attrs.items()
                         if not k.startswith('_damnit_')}
        return obj

    def read(self, deserialize_plotly=True):
        """Read the data for the variable.

        Args:
            deserialize_plotly (bool): Whether to deserialize Plotly figures
                into `Figure` objects. If this is `False` the JSON string will be returned.
        """
        if self._db_only:
            return self.summary()

        with self._open_h5_group() as group:
            type_hint = self._type_hint(group)
            if type_hint is DataType.Dataset:
                return self._read_netcdf()
            elif type_hint is DataType.DataFrame:
                dataset = self._read_netcdf(strip_internal_attrs=False)
                if len(dataset.dims) <= 1:
                    dataframe = dataset.to_pandas()
                else:
                    dataframe = dataset.to_dataframe()

                columns_json = dataset.attrs["_damnit_dataframe_columns"]
                dataframe.columns = _decode_dataframe_columns(json.loads(columns_json))

                index_names_json = dataset.attrs["_damnit_dataframe_index_names"]
                index_names = json.loads(index_names_json)
                dataframe.index = dataframe.index.set_names(index_names)

                column_names_json = dataset.attrs["_damnit_dataframe_column_names"]
                column_names = json.loads(column_names_json)
                if len(column_names) == dataframe.columns.nlevels:
                    dataframe.columns = dataframe.columns.set_names(column_names)
                else:
                    import pandas as pd
                    dataframe.columns = pd.MultiIndex.from_tuples(
                        dataframe.columns.tolist(), names=column_names
                    )
                return dataframe
            elif type_hint is DataType.DataArray:
                return self._read_netcdf(one_array=True)
            elif type_hint is DataType.Series:
                return self._read_netcdf(one_array=True).to_pandas()

            dset = group["data"]
            if type_hint is DataType.PlotlyFigure:
                import plotly.io as pio

                # plotly figures are json serialized and saved as uint8 arrays
                # to enable compression in HDF5
                byte_array = dset[()].tobytes()
                return pio.from_json(byte_array) if deserialize_plotly else byte_array.decode()
            elif h5py.check_string_dtype(dset.dtype) is not None:
                # Strings. Scalar/non-scalar strings need to be read differently.
                if dset.ndim == 0:
                    return dset[()].decode("utf-8", "surrogateescape")
                else:
                    return dset.asstr("utf-8", "surrogateescape")[0]
            elif dset.ndim == 0:
                # Scalars
                return dset[()]
            else:
                # Otherwise, return a Numpy array
                return group["data"][()]

    def summary(self):
        """Read the summary data for a variable.

        For user-editable variables like comments, this will be the same as
        [VariableData.read()][damnit.api.VariableData.read].
        """
        result = self._db.conn.execute("""
            SELECT value, summary_type, max(version) FROM run_variables
            WHERE proposal=? AND run=? AND name=?
        """, (self.proposal, self.run, self.name)).fetchone()

        if result is None:
            # This should never be reached unless the variable is deleted
            # after creating the VariableData object.
            raise RuntimeError(f"Could not find value for '{self.name}' in p{self.proposal}, r{self.name}")
        else:
            value, summary_type, version = result
            if isinstance(value, bytes):
                if summary_type == "complex":
                    return blob2complex(value)
                if summary_type in ("numpy", "trendline"):
                    return blob2numpy(value)
                if BlobTypes.identify(value) is BlobTypes.numpy:
                    return blob2numpy(value)
            return value

    def preview_data(self, *, data_fallback=True, deserialize_plotly=True):
        """Get the preview data for the variable

        May return a 1D or 2D data array, a 3D RGB(A) arrray, a Plotly figure
        object, a str of Plotly JSON (with deserialize_plotly=False) or None
        if no preview is available.

        If no preview was specified in the context file, but the returned data
        meets the conditions for a preview, this can be returned instead.
        Pass `data_fallback=False` to prevent this.
        """
        with h5py.File(self._h5_path) as f:
            xarray_group = dset = None
            if (obj := f.get(f".preview/{self.name}")) is not None:
                # Explicit preview
                type_hint = self._type_hint(obj)
                if isinstance(obj, h5py.Group):
                    xarray_group = obj.name
                else:
                    dset = obj
            elif data_fallback:
                # Implicit: use data as preview if suitable
                grp = f[self.name]
                type_hint = self._type_hint(grp)
                if type_hint in (DataType.DataArray, DataType.Series):
                    xarray_group = self.name
                elif type_hint in (DataType.Dataset, DataType.DataFrame):
                    return None
                else:
                    dset = grp['data']
            else:
                return None

            if xarray_group is not None:
                for obj in f[xarray_group].values():
                    if isinstance(obj, h5py.Dataset) and (
                        obj.ndim > 3 or (obj.ndim == 3 and obj.shape[-1] not in (3, 4))
                    ):
                        return None  # Too many dims: bail out before loading

                import xarray as xr
                arr = xr.load_dataarray(
                    self._h5_path, group=xarray_group, engine="h5netcdf"
                )
                if arr.ndim != 0 and (np.issubdtype(arr.dtype, np.number) or arr.dtype == bool):
                    return arr

            elif (np.issubdtype(dset.dtype, np.number) or dset.dtype == bool) and (
                    dset.ndim in (1, 2) or (dset.ndim == 3 and dset.shape[-1] in (3, 4))
            ):
                value = dset[()]

                if type_hint is DataType.PlotlyFigure:
                    import plotly.io as pio
                    b = value.tobytes()
                    return pio.from_json(b) if deserialize_plotly else b.decode()
                else:
                    return value

        return None

    def preview(self):
        """Show the preview data

        This is intended for use in Jupyter notebooks
        """
        if (obj := self.preview_data()) is None:
            raise ValueError("No preview available")

        if isinstance_no_import(obj, 'plotly.graph_objs', 'Figure'):
            obj.show()
            return obj
        elif isinstance_no_import(obj, 'xarray', 'DataArray'):
            return obj.plot()  # Let Xarray decide what to plot
        else:  # ndarray
            import matplotlib.pyplot as plt
            obj = obj.squeeze()

            fig, ax = plt.subplots()

            match obj.ndim:
                case 3:  # Colour image
                    res = ax.imshow(obj, interpolation="antialiased")
                    ax.tick_params(left=False, bottom=False,
                                   labelleft=False, labelbottom=False)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                case 2:
                    res = ax.imshow(obj, interpolation="antialiased")
                    vmin = np.nanquantile(obj, 0.01, method='nearest')
                    vmax = np.nanquantile(obj, 0.99, method='nearest')
                    res.set_clim(vmin, vmax)
                    fig.colorbar(res, ax=ax)
                case 1:
                    res = ax.plot(obj, fmt='o')
                    ax.set_ylabel(self.title)
                case _:
                    return None

            if obj.ndim != 3:
                ax.set_title(f'{self.title} (run {self.run})')

            return res

    def __repr__(self):
        return f"<VariableData for '{self.name}' in p{self.proposal}, r{self.run}>"


class RunVariables:
    """Represents the variables for a single run.

    Don't create this object yourself, index a [Damnit][damnit.api.Damnit]
    object instead.

    Indexing this by either a variable name or title will return a
    [VariableData][damnit.api.VariableData] object:
    ```python
    db = Damnit(1234)
    run_vars = db[100]
    myvar = run_vars["myvar"] # Alternatively by title, `run_vars["My Variable"]`
    ```
    """

    def __init__(self, db_dir, run, proposal=None):
        self._db = DamnitDB.from_dir(db_dir)
        self._proposal = proposal or self._db.metameta["proposal"]
        self._run = run
        self._data_format_version = self._db.metameta["data_format_version"]
        self._h5_path = Path(db_dir) / f"extracted_data/p{self._proposal}_r{self._run}.h5"

    @property
    def proposal(self) -> int:
        """The proposal of the run."""
        return self._proposal

    @property
    def run(self) -> int:
        """The run number."""
        return self._run

    @property
    def file(self) -> Path:
        """The path to the HDF5 file for the run."""
        return self._h5_path

    def __getitem__(self, name):
        key_locs = self._key_locations()
        names_to_titles = self._var_titles()
        titles_to_names = { title: name for name, title in names_to_titles.items() }

        if name not in key_locs and name not in titles_to_names:
            raise KeyError(f"Variable data for {name!r} not found for p{self.proposal}, r{self.run}")

        if name in titles_to_names:
            name = titles_to_names[name]

        return VariableData(name, names_to_titles[name],
                            self.proposal, self.run,
                            self._h5_path, self._data_format_version,
                            self._db, key_locs[name])

    def _key_locations(self):
        # Read keys from the HDF5 file
        with h5py.File(self.file) as f:
            all_keys = { name: False for name in f.keys()
                         if not name.startswith('.')}

            # Add keys with summary but not data in file
            all_keys.update({ name: False for name in f['.reduced'].keys()
                              if name not in all_keys})

        # And the keys from the database
        user_vars = list(self._db.get_user_variables().keys())
        user_vars.append("comment")

        for var_name in user_vars:
            result = self._db.conn.execute("""
                SELECT name, value, max(version) FROM run_variables
                WHERE proposal=? AND run=? AND name=?
            """, (self.proposal, self.run, var_name)).fetchone()
            if result is not None and result[1] is not None:
                all_keys[var_name] = True

        return all_keys

    def keys(self) -> list:
        """The names of the available variables.

        Note that a variable will not appear in the list if there is no data for
        it.
        """
        return sorted(self._key_locations().keys())

    def _var_titles(self):
        result = self._db.conn.execute("SELECT name, title FROM variables").fetchall()
        available_vars = self.keys()
        titles = { row[0]: row[1] if row[1] is not None else row[0] for row in result
                   if row[0] in available_vars }

        # These variables are created automatically, but they aren't included in
        # the `variables` table (yet) so we need to explicitly add their titles.
        special_vars = {
            "start_time": "Timestamp",
            "comment": "Comment"
        }
        for name, title in special_vars.items():
            if name in available_vars:
                titles[name] = title

        return titles

    def titles(self) -> list:
        """The titles of available variables.

        As with [RunVariables.keys()][damnit.api.RunVariables.keys], only
        variables that have data for the run will be included.
        """
        return sorted(list(self._var_titles().values()))

    def _ipython_key_completions_(self):
        # This makes autocompleting the variable names work with ipython
        return self.keys()

    def __repr__(self):
        return f"<RunVariables for p{self.proposal}, r{self.run} with {len(self.keys())} variables>"


class Damnit:
    """Represents a DAMNIT database.

    Indexing this will return either a [RunVariables][damnit.api.RunVariables]
    or [VariableData][damnit.api.VariableData] object:
    ```python
    db = Damnit(1234)

    # Index by run number (or proposal & run number) to get a RunVariables object
    run_vars = db[100]
    run_vars = db[1234, 100]
    # You can add a variable name/title to get a VariableData object
    myvar = db[100, "myvar"]
    ```
    """

    def __init__(self, location):
        """
        This is the entrypoint for inspecting data stored by DAMNIT.

        Args:
            location (int or str or Path): This can be either a proposal number or
                a path to a database directory.
        """
        if isinstance(location, int):
            proposal_path = find_proposal(location)
            self._db_dir = proposal_path / "usr/Shared/amore"
        elif isinstance(location, (Path, str)):
            self._db_dir = Path(location)
        else:
            raise TypeError(f"Unsupported location: {location}")

        if not self._db_dir.is_dir():
            raise FileNotFoundError(f"DAMNIT directory does not exist: {self._db_dir}")

        self._db_path = self._db_dir / "runs.sqlite"
        if not self._db_path.is_file():
            raise FileNotFoundError(f"DAMNIT database does not exist: {self._db_path}")

        self._db = DamnitDB(self._db_path)

    def __getitem__(self, obj):
        proposal = variable = None
        if isinstance(obj, int):
            run = obj
        elif isinstance(obj, tuple) and len(obj) == 2:
            if isinstance(obj[1], str):
                run, variable = obj
            else:
                proposal, run = obj
        elif isinstance(obj, tuple) and len(obj) == 3:
            proposal, run, variable = obj
        else:
            raise TypeError(f"Unrecognised key type: {type(obj)}")

        if run not in self.runs():
            raise KeyError(f"Unknown run number for p{self.proposal}")

        run_vars = RunVariables(self._db_dir, run, proposal=proposal)
        return run_vars[variable] if variable is not None else run_vars

    @property
    def proposal(self) -> int:
        """The currently active proposal of the database."""
        return self._db.metameta["proposal"]

    def runs(self) -> list:
        """A list of all existing runs.

        Note that this does not include runs that were pre-created through the
        GUI but were never taken by the DAQ.
        """
        result = self._db.conn.execute("SELECT run FROM run_info WHERE start_time IS NOT NULL").fetchall()
        return [row[0] for row in result]

    def table(self, with_titles=False) -> "pd.DataFrame":
        """Retrieve the run table as a [DataFrame][pandas.DataFrame].

        There are a few differences compared to what you'll see in the table
        displayed in the GUI:

        - Images will be replaced with an `<image>` string.
        - Trendline summaries will be replaced with a `<sparkline>` string.
        - Runs that were pre-created through the GUI but never taken by the DAQ
          will not be included.

        Args:
            with_titles (bool): Whether to use variable titles instead of names
                for the columns in the dataframe.
        """
        import pandas as pd

        df = pd.read_sql_query("""
            SELECT proposal, run, start_time, added_at
            FROM run_info
            WHERE start_time IS NOT NULL
            ORDER BY proposal, run
        """, self._db.conn)

        # Get variable values from `run_variables` and pivot them into columns
        variable_data = pd.read_sql_query("""
            SELECT
                v.proposal,
                v.run,
                v.name,
                CASE
                    WHEN typeof(v.value) = 'blob' AND v.summary_type = 'complex' THEN v.value
                    WHEN typeof(v.value) = 'blob' AND v.summary_type = 'trendline' THEN '<sparkline>'
                    WHEN typeof(v.value) = 'blob' AND v.summary_type = 'numpy' THEN '<ndarray>'
                    -- If the summary type is missing, we assume the blobs are thumbnails (PNG or numpy)
                    -- TODO: should we always add summary types for blobs in the future?
                    WHEN typeof(v.value) = 'blob' AND (v.summary_type IS NULL OR v.summary_type = '') THEN '<thumbnail>'
                    WHEN typeof(v.value) = 'blob' THEN '<blob>'
                    ELSE v.value
                END AS value,
                v.summary_type
            FROM run_variables AS v
            INNER JOIN (
                SELECT proposal, run, name, max(version) AS version
                FROM run_variables
                GROUP BY proposal, run, name
            ) AS latest
            ON latest.proposal = v.proposal
            AND latest.run = v.run
            AND latest.name = v.name
            AND latest.version = v.version
            ORDER BY v.proposal, v.run
        """, self._db.conn)

        summary_types_by_run = {}
        if not variable_data.empty:
            values_wide = variable_data.pivot(index=["proposal", "run"],
                                              columns="name",
                                              values="value")
            missing_cells = variable_data.assign(_present=True).pivot(
                index=["proposal", "run"], columns="name", values="_present"
            ).isna()

            for name in missing_cells.columns[missing_cells.any()]:
                values_wide[name] = values_wide[name].astype(object)
                values_wide.loc[missing_cells[name], name] = pd.NA
            df = df.merge(values_wide.reset_index(), on=["proposal", "run"], how="left")

            for row in variable_data.itertuples(index=False):
                if row.summary_type is not None:
                    key = (row.proposal, row.run)
                    summary_types_by_run.setdefault(key, {})[row.name] = row.summary_type

        # Keep all known variables as columns, even if they have no value yet.
        for name in self._db.variable_names():
            if name not in ("proposal", "run", "start_time", "added_at") and name not in df:
                df[name] = None

        # Convert the start_time into a datetime column
        start_time = pd.to_datetime(df["start_time"], unit="s", utc=True)
        df["start_time"] = start_time.dt.tz_convert("Europe/Berlin")

        # Delete added_at, this is internal
        del df["added_at"]

        # Ensure that there's always a comment column for consistency, it may
        # not be present if no comments were made.
        if "comment" not in df:
            df.insert(3, "comment", None)

        def interpret_blobs(row):
            summary_types = summary_types_by_run.get((row["proposal"], row["run"]), {})

            for col in row.keys():
                if summary_types.get(col) == "complex":
                    row[col] = blob2complex(row[col])
            return row

        df = df.apply(interpret_blobs, axis=1)

        # Use the full variable titles
        if with_titles:
            results = self._db.conn.execute("SELECT name, title FROM variables").fetchall()
            renames = { row[0]: row[1] for row in results }
            renames["proposal"] = "Proposal"
            renames["run"] = "Run"
            renames["start_time"] = "Timestamp"

            df.rename(columns=renames, inplace=True)

        return df

    def __repr__(self):
        return f"<Damnit database for p{self.proposal}>"


def submit(proposal: int, run: int, variables, *, provenance,
        errors: dict[str, Exception] = None, damnit_dir=None,
):
    """Add some results into DAMNIT's store

    Args:
        proposal (int): Proposal number
        run (int): Run number
        variables (dict): Mapping of names to arrays or DAMNIT Cell objects.
        provenance (str): A name for what produced these results.
        errors (dict): Mapping of names to exceptions, to make error messages
            visible in the table.
        damnit_dir (Path or str, optional): The DAMNIT directory to write into.
            If not specified, it will find the default directory for the
            relevant proposal.
    """
    from .backend.extract_data import notify_new_file
    from .context import Cell, save_fragment

    variables = {k: (v if isinstance(v, Cell) else Cell(v))
                 for (k, v) in variables.items()}
    errors = errors or {}

    if damnit_dir is None:
        damnit_dir = find_proposal(proposal) / "usr/Shared/amore"
    else:
        damnit_dir = Path(damnit_dir)

    path = save_fragment(
        damnit_dir, proposal, run, variables, errors, provenance=provenance
    )
    notify_new_file(damnit_dir, proposal, run, str(path))
