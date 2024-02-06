import glob
from pathlib import Path
from contextlib import contextmanager

import h5py
import xarray as xr

from .db import DamnitDB
from ..context import DataType, add_to_h5_file


class VariableData:
    def __init__(self, name, proposal, run, h5_path, data_format_version):
        self._name = name
        self._proposal = proposal
        self._run = run
        self._h5_path = h5_path
        self._data_format_version = data_format_version

    @property
    def name(self):
        return self._name

    @property
    def file(self):
        return self._h5_path

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

    def read(self):
        with self._open_h5_group() as group:
            type_hint = self._type_hint(group)
            if type_hint is DataType.Dataset:
                return self._read_netcdf()
            elif type_hint is DataType.DataArray:
                return self._read_netcdf(one_array=True)

            dset = group["data"]
            if dset.ndim == 0:  # Scalars
                if h5py.check_string_dtype():
                    return dset[()].decode("utf-8", "surrogateescape")
                return dset[()]

            if len(group.keys()) > 1:
                # Has coordinates - old-style DataArray storage
                self._read_xarray_oldfmt(group)

            # Otherwise, return a Numpy array
            return group["data"][()]

    def _read_netcdf(self, one_array=False):
        load = xr.load_dataarray if one_array else xr.load_dataset
        obj = load(self._h5_path, group=self.name, engine="h5netcdf")
        # Remove internal attributes from loaded object
        obj.attrs = {k: v for (k, v) in obj.attrs.items()
                     if not k.startswith('_damnit_')}
        return obj

    def xarray(self):
        with self._open_h5_group() as group:
            type_hint = self._type_hint(group)
            if type_hint is DataType.DataArray:
                return self._read_netcdf(one_array=True)
            elif type_hint is DataType.Dataset:
                return self._read_netcdf()
            else:
                return self._read_xarray_oldfmt(group)

    def _read_xarray_oldfmt(self, group):
        # Old-style, coordinates stored ad-hoc rather than using NetCDF4
        data = group["data"][()]
        coords = { ds_name: group[ds_name][()] for ds_name in group.keys()
                   if ds_name != "data" }

        # Attempt to map the coords to the right dimensions. This
        # will fail if there are two coordinates/dimensions with the
        # same length.
        coord_sizes = { len(coord_data): coord for coord, coord_data in coords.items() }
        if len(set(coord_sizes.keys())) != len(coord_sizes.keys()):
            raise RuntimeError(f"Could not read DataArray for variable '{self.name}', dimensions have the same length")

        dims = [coord_sizes.get(dim, f'dim_{i}') for i, dim in enumerate(data.shape)]
        return xr.DataArray(data, dims=dims, coords=coords)

    def ndarray(self):
        with self._open_h5_group() as group:
            type_hint = self._type_hint(group)
            if type_hint is DataType.DataArray:
                return self._read_netcdf(one_array=True).data
            elif type_hint is DataType.Dataset:
                raise TypeError("Variable is an xarray Dataset, can't convert to ndarray")
            else:
                return group["data"][()]


class RunVariables:
    def __init__(self, db_dir, run):
        db = DamnitDB.from_dir(db_dir)
        self._proposal = db.metameta["proposal"]
        self._run = run
        self._data_format_version = db.metameta["data_format_version"]
        self._h5_path = Path(db_dir) / f"extracted_data/p{self._proposal}_r{self._run}.h5"

    def __getitem__(self, name):
        return VariableData(name, self._proposal, self._run,
                            self._h5_path, self._data_format_version)

    @property
    def file(self):
        return self._h5_path

def delete_variable(db, name):
    # Remove from the database
    db.delete_variable(name)

    # And the HDF5 files
    for h5_path in glob.glob(f"{db.path.parent}/extracted_data/*.h5"):
        with add_to_h5_file(h5_path) as f:
            if name in f:
                del f[f".reduced/{name}"]
                del f[name]
