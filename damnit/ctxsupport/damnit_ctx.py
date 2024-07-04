"""This module is made available by manipulating sys.path

We aim to maintain compatibility with older Python 3 versions (currently 3.9+)
than the DAMNIT code in general, to allow running context files in other Python
environments.
"""
import re
import sys
from collections.abc import Sequence
from enum import Enum

import h5py
import numpy as np
import xarray as xr

__all__ = ["RunData", "Variable", "Cell"]


THUMBNAIL_SIZE = 300 # px


def isinstance_no_import(obj, mod: str, cls: str):
    """Check if isinstance(obj, mod.cls) without loading mod"""
    m = sys.modules.get(mod)
    if m is None:
        return False

    return isinstance(obj, getattr(m, cls))


class RunData(Enum):
    RAW = "raw"
    PROC = "proc"
    ALL = "all"


class Variable:
    _name = None

    def __init__(
            self, title=None, description=None, summary=None, data=None, cluster=False,
    ):
        self.title = title
        self.description = description
        self.summary = summary
        self.cluster = cluster
        self._data = data

    # @Variable() is used as a decorator on a function that computes a value
    def __call__(self, func):
        self.func = func
        self.name = func.__name__
        return self

    def check(self):
        problems = []
        if not re.fullmatch(r'[a-zA-Z_]\w+', self.name, flags=re.A):
            problems.append(
                f"The variable name {self.name!r} is not of the form '[a-zA-Z_]\\w+'"
            )
        if self._data not in (None, "raw", "proc"):
            problems.append(
                f"data={self._data!r} for variable {self.name} (can be 'raw'/'proc')"
            )
        return problems

    @property
    def data(self):
        """
        Return the RunData of the Variable.
        """
        return RunData.RAW if self._data is None else RunData(self._data)

    def arg_dependencies(self, prefix="var#"):
        """
        Get all direct dependencies of this Variable with a certain
        type/prefix. Returns a dict of argument name to variable name.
        """
        return { arg_name: annotation.removeprefix(prefix)
                 for arg_name, annotation in self.annotations().items()
                 if annotation.startswith(prefix) }

    def annotations(self):
        """
        Get all annotated arguments of this Variable (including meta arguments,
        unlike `Variable.dependencies()`).

        Returns a dict of argument names to their annotations.
        """
        return getattr(self.func, '__annotations__', {})


class Cell:
    """Variable functions can return this"""
    def __init__(self, data, summary=None, summary_value=None, bold=None, background=None):
        # If the user returns an Axes, save the whole Figure
        if isinstance_no_import(data, 'matplotlib.axes', 'Axes'):
            data = data.get_figure()

        isfig = isinstance_no_import(data, 'matplotlib.figure', 'Figure') or \
                isinstance_no_import(data, 'plotly.graph_objs', 'Figure')

        if not (isfig or isinstance(data, (xr.Dataset, xr.DataArray, str, type(None)))):
            data = np.asarray(data)
            # Numpy will wrap any Python object, but only native arrays
            # can be saved in HDF5, not those containing Python objects.
            if data.dtype.hasobject:
                raise TypeError(f"Returned data type {type(data)} cannot be saved")
            elif not np.issubdtype(data.dtype, np.number):
                try:
                    h5py.h5t.py_create(data.dtype, logical=True)
                except TypeError:
                    raise TypeError(
                        f"Returned data type {type(data)} whose native "
                        f"array type {data.dtype} cannot be saved",
                    )

        if summary_value is not None and not isinstance(summary_value, str):
            arr = np.asarray(summary_value)
            if arr.dtype.hasobject:
                raise TypeError(f"summary_value should be number or string, not {type(summary)}")
            elif not np.issubdtype(arr.dtype, np.number):
                try:
                    h5py.h5t.py_create(arr.dtype, logical=True)
                except TypeError:
                    raise TypeError(
                        f"Summary value {type(arr)} whose native "
                        f"array type {arr.dtype} cannot be saved",
                    )
            summary_value = arr

        self.data = data
        self.summary = summary
        self.summary_value = summary_value
        self.bold = bold
        self.background = self._normalize_colour(background)

    @staticmethod
    def _normalize_colour(c):
        if isinstance(c, str):
            if not re.match(r'#[0-9A-Fa-f]{6}', c):
                raise ValueError("Colour string should be hex code (like '#ffcc00')")
            b = bytes.fromhex(c[1:])
            return np.frombuffer(b, dtype=np.uint8)
        elif isinstance(c, Sequence):
            if not len(c) == 3:
                raise TypeError(f"Wrong number of values ({len(c)}) for R,G,B")
            if not all(0 <= v <= 255 for v in c):
                raise ValueError("Colour values must be 0 - 255")
            return np.array(c, dtype=np.uint8)
        elif c is None:
            return c
        else:
            raise TypeError(f"Don't understand colour as {type(c)}")

    def get_summary(self):
        if self.summary_value is not None:
            return self.summary_value
        elif self.summary is not None:
            return np.asarray(getattr(np, self.summary)(self.data))

        return None

    def _max_diff(self):
        a = self.data
        if isinstance(a, (np.ndarray, xr.DataArray)) and a.size > 1:
            return abs(np.subtract(np.nanmax(a), np.nanmin(a), dtype=np.float64))

    def summary_attrs(self):
        d = {}
        if self.summary is not None:
            d['summary_method'] = self.summary
        if self.bold is not None:
            d['bold'] = self.bold
        if self.background is not None:
            d['background'] = self.background
        if (max_diff := self._max_diff()) is not None:
            d['max_diff'] = max_diff
        return d
