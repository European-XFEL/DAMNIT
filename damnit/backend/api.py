from pathlib import Path
from contextlib import contextmanager

import h5py
import xarray as xr

from .db import DamnitDB
from ..ctxsupport.ctxrunner import DataType


class VariableData:
    def __init__(self, name, proposal, run, h5_path, data_format_version, stored_type):
        self._name = name
        self._proposal = proposal
        self._run = run
        self._h5_path = h5_path
        self._data_format_version = data_format_version
        self._stored_type = stored_type

    @property
    def name(self):
        return self._name

    @property
    def file(self):
        return self._h5_path

    @property
    def stored_type(self):
        if self._stored_type is None:
            with self._open_h5_group() as (group, stored_type):
                self._stored_type = stored_type

        return self._stored_type

    def _type(self, group: h5py.Group) -> DataType:
        # Versions from 1 onwards store the type
        if self._data_format_version > 0:
            return self._stored_type

        # Otherwise fall back to guessing from the HDF5 file

        if h5py.check_string_dtype(group["data"].dtype) is not None:
            return DataType.String

        if group["data"].ndim == 0:
            return DataType.Scalar

        if group["data"].ndim > 0:
            has_coords = len(group.keys()) > 1
            return DataType.DataArray if has_coords else DataType.NDArray

        raise RuntimeError(f"Couldn't determine type of variable '{self.name}'")

    @contextmanager
    def _open_h5_group(self):
        with h5py.File(self._h5_path) as f:
            group = f[self.name]
            stored_type = self.stored_type
            yield group, stored_type

    def read(self):
        with self._open_h5_group() as (group, stored_type):
            if stored_type == DataType.String:
                return group["data"].asstr()[0]
            elif stored_type == DataType.Scalar:
                return group["data"][()].item()
            elif stored_type == DataType.NDArray:
                return self._read_ndarray(group, stored_type)
            elif stored_type == DataType.DataArray:
                return self._read_xarray(group, stored_type)
            else:
                raise RuntimeError(f"Unsupported type: {stored_type}")

    def xarray(self):
        with self._open_h5_group() as (group, stored_type):
            return self._read_xarray(group, stored_type)

    def _read_xarray(self, group, stored_type):
            if stored_type == DataType.DataArray:
                if self._data_format_version >= 1:
                    return xr.open_dataarray(self._h5_path, group=self.name, engine="h5netcdf")
                else:
                    data = group["data"][()]
                    coords = { ds_name: group[ds_name][()] for ds_name in group.keys()
                               if ds_name != "data" }

                    # Attempt to map the coords to the right dimensions. This
                    # will fail if there are two coordinates/dimensions with the
                    # same length.
                    coord_sizes = { len(coord_data): coord for coord, coord_data in coords.items() }
                    if len(set(coord_sizes.keys())) != len(coord_sizes.keys()):
                        raise RuntimeError(f"Could not read DataArray for variable '{self.name}', dimensions have the same length")

                    dims = [coord_sizes[dim] for dim in data.shape]
                    return xr.DataArray(data,
                                        dims=dims,
                                        coords={ dim: coords[dim] for dim in dims })

            elif stored_type == DataType.NDArray:
                data = group["data"][()]
                return xr.DataArray(data)

    def ndarray(self):
        with self._open_h5_group() as (group, stored_type):
            return self._read_ndarray(group, stored_type)

    def _read_ndarray(self, group, stored_type):
        if stored_type in [DataType.NDArray, DataType.Image]:
            return group["data"][()]
        elif stored_type == DataType.DataArray:
            return self._read_xarray(group, stored_type).data
        else:
            raise RuntimeError(f"Could not read ndarray for variable '{self.name}', actual type is '{stored_type}'")


class RunVariables:
    def __init__(self, db_dir, run):
        db = DamnitDB.from_dir(db_dir)
        self._proposal = db.metameta["proposal"]
        self._run = run
        self._data_format_version = db.metameta["data_format_version"]
        self._h5_path = Path(db_dir) / f"extracted_data/p{self._proposal}_r{self._run}.h5"

        records = db.conn.execute("""
            SELECT name, stored_type FROM run_variables
            WHERE proposal=? AND run=? AND version=1
        """, (self._proposal, self._run))
        self._stored_types = { name: None if t is None else DataType(t)
                               for name, t in records }

        db.close()

    def __getitem__(self, name):
        return VariableData(name, self._proposal, self._run,
                            self._h5_path, self._data_format_version,
                            self._stored_types[name])

    @property
    def file(self):
        return self._h5_path
