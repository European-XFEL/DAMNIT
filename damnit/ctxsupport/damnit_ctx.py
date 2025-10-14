"""This module is made available by manipulating sys.path

We aim to maintain compatibility with older Python 3 versions (currently 3.9+)
than the DAMNIT code in general, to allow running context files in other Python
environments.
"""
import inspect
import logging
import re
import sys
from collections.abc import Sequence
from copy import copy
from enum import Enum
from functools import wraps

import h5py
import numpy as np
import xarray as xr

__all__ = ["RunData", "Variable", "Cell"]

log = logging.getLogger(__name__)


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
            self, title=None, description=None, summary=None, data=None,
            cluster=False, tags=None, transient=False
    ):
        self.tags = (tags,) if isinstance(tags, str) else tags
        self.description = description
        self.summary = summary
        self.cluster = cluster
        self.transient = transient
        self._data = data

        if callable(title):
            # @Variable called without parenthesis
            func = title
            self.title = None
            self(func)
        else:
            self.title = title

    # @Variable() is used as a decorator on a function that computes a value
    def __call__(self, func):
        self.func = func
        if hasattr(func, '__annotations__'):
            for k, v in func.__annotations__.items():
                if isinstance(v, str) and v.startswith('var#'):
                    # Replace '.' syntaxic sugar with '__' in var dependencies
                    func.__annotations__[k] = v.replace('.', '__')

        self.name = func.__name__
        if self.title is None:
            self.title = self.name
        return self

    def check(self):
        problems = []
        if not self.name.isidentifier():
            problems.append(
                f"The variable name {self.name!r} is not a valid Python identifier"
            )
        if self._data not in (None, "raw", "proc"):
            problems.append(
                f"data={self._data!r} for variable {self.name} (can be 'raw'/'proc')"
            )
        if self.tags is not None:
            if not isinstance(self.tags, Sequence) or not all(
                isinstance(tag, str) and tag != "" for tag in self.tags
            ):
                problems.append(
                    f"tags={self.tags!r} for variable {self.name} "
                    "(must be a non-empty string or an iterable of strings)"
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
        return {
            arg_name: annotation.removeprefix(prefix)
            for arg_name, annotation in self.annotations().items()
            if isinstance(annotation, str) and annotation.startswith(prefix)
        }

    def annotations(self):
        """
        Get all annotated arguments of this Variable (including meta arguments,
        unlike `Variable.dependencies()`).

        Returns a dict of argument names to their annotations.
        """
        return getattr(self.func, '__annotations__', {})

    def evaluate(self, run_data, kwargs):
        cell = self.func(run_data, **kwargs)

        if self.transient:
            if not isinstance(cell, Cell):
                cell = _DummyCell(cell)
        else:
            if not isinstance(cell, Cell):
                cell = Cell(cell)

            if cell.summary is None:
                cell.summary = self.summary

        return cell

    def copy(self):
        return copy(self)


class VariableGroup:
    """Base class for creating reusable groups of Variables"""

    def __init__(
        self,
        title: str = None,
        *,
        sep: str = '/',
        tags: str | list[str] | None = None,
        cluster: bool = False,
        transient: bool = False,
        **kwargs,
    ):
        """

        Args:
            title: The title of the variable group.
            sep: The separator to use to construct variable titles. Variables'
                titles in a VariableGroup are formatted as
                <group_title><sep><variable_title>. Defaults to '/'.
            tags: A list of tags to apply to all Variables in this group.
            cluster: If True, all Variables in this group are marked as cluster.
            transient: If True, all Variables in this group are transient.
            kwargs: Additional keyword arguments that will be passed to all
                Variables in this group. These can be used to pass
                device-specific parameters or other context that should be
                available to all variables.
        """
        self.title = title
        self.sep = sep
        self.tags = (tags,) if isinstance(tags, str) else tags
        self.cluster = cluster
        self.transient = transient
        self.group_kwargs = kwargs

        self._variables = {}
        self._groups = {}

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Variable):
                self._variables[attr_name] = attr
            elif isinstance(attr, VariableGroup):
                self._groups[attr_name] = attr

    def _merge_tags(self, original_tags):
        """Merge original tags with group tags"""
        if self.tags is None:
            return original_tags
        if original_tags is None:
            return self.tags
        return set(self.tags) | set(original_tags)

    def _init_group_variable(self, prefix, original_var: Variable) -> Variable:
        """Create a new Variable instance for this group"""
        new_var = original_var.copy()

        new_var.name = f'{prefix}__{new_var.name}'
        new_var.title = f"{self.title or prefix}{self.sep}{new_var.title}"
        new_var.tags = self._merge_tags(new_var.tags)
        new_var.cluster |= self.cluster
        new_var.transient |= self.transient

        return new_var

    def _create_wrapper_function(self, var, annotations, signature):
        """Create a wrapper function that provides group context"""
        func = var.func

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Add group-specific kwargs
            args_in_func = inspect.signature(func).parameters
            group_kwargs = {k: v for k, v in self.group_kwargs.items() if k in args_in_func}
            kwargs |= group_kwargs

            # Call the original function with group context
            if next(iter(args_in_func)) == 'self':
                return func(self, *args, **kwargs)
            return func(*args, **kwargs)

        wrapper.__name__ = var.name
        wrapper.__annotations__ = annotations
        wrapper.__signature__ = signature
        wrapper.__doc__ = func.__doc__

        return wrapper

    def variables(self, prefix: str) -> dict[str, Variable]:
        """Get all variables in this group"""
        _vars = {}
        for original_var in self._variables.values():
            var = self._init_group_variable(prefix, original_var)

            # edit annotations to include the group prefix
            annotations = var.annotations().copy()
            for arg_name, annotation in annotations.items():
                if isinstance(annotation, str) and annotation.startswith('var#'):
                    dep_name = annotation.removeprefix('var#')
                    if dep_name.startswith('self__'):
                        # Dependency is in this instance -> prefix the dep_name
                        annotations[arg_name] = f'var#{prefix}__{dep_name.removeprefix("self__")}'

            # Create new signature with the modified annotations
            original_sig = inspect.signature(var.func)
            params = []
            for index, param in enumerate(original_sig.parameters.values()):
                if index == 0 and param.name == 'self':
                    # Skip the 'self' parameter for instance methods
                    continue
                # Replace the parameter with the new annotation if it exists
                if param.name in annotations:
                    param = param.replace(annotation=annotations[param.name])                
                params.append(param)

            new_sig = original_sig.replace(parameters=params)

            # wrap the original function with the new signature and instance kwargs
            var.func = self._create_wrapper_function(
                var, annotations=annotations, signature=new_sig)

            _vars[var.name] = var

        # Add variables from nested groups
        for group_name, group in self._groups.items():
            group_prefix = f"{prefix}__{group_name}"
            # update title and tags for the group
            group.title = f"{self.title or prefix}{self.sep}{group.title or group_name}"
            group.tags = self._merge_tags(group.tags)
            _vars.update(group.variables(group_prefix))

        return _vars


class _DummyCell:
    """For transient results, holds data with no type checks"""
    def __init__(self, data):
        self.data = data


class Cell:
    """A container for data with customizable table display options.

    Validates and converts input data to HDF5-compatible formats.
    Provides flexible summary generation through direct values or numpy functions.
    Supports visual customization with bold text and background colors.

    Parameters
    ----------
    data : array-like, Figure, Dataset, str, or None
        The main data to store
    summary : str, optional
        Name of numpy function to compute summary from data
    summary_value : str or number, optional
        Direct value to use as summary
    bold : bool, optional
        Whether to display cell in bold
    background : str or sequence, optional
        Cell background color as hex string ('#ffcc00') or RGB sequence (0-255)
    preview : array or figure object, optional
        A plot, 1D or 2D array to show when double-clicking the table cell.
    """
    def __init__(self, data, summary=None, summary_value=None, bold=None, background=None,
                 *, preview=None):
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

        if preview is not None:
            if isinstance(preview, (np.ndarray, xr.DataArray)):
                if preview.ndim not in (1, 2):
                    raise TypeError(
                        f"preview should be a 1D or 2D array (shape is {preview.shape})"
                    )
                elif not np.issubdtype(preview.dtype, np.number):
                    raise TypeError("preview array should be numeric")
            elif not (
                    isinstance_no_import(preview, 'matplotlib.figure', 'Figure') or
                    isinstance_no_import(preview, 'plotly.graph_objs', 'Figure')
            ):
                raise TypeError("preview must be an array or a figure object "
                                f"(got {type(preview)})")

        self.data = data
        self.summary = summary
        self.summary_value = summary_value
        self.bold = bold
        self.background = self._normalize_colour(background)
        self.preview = preview

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
        elif (self.data is not None) and (self.summary is not None):
            try:
                return np.asarray(getattr(np, self.summary)(self.data))
            except Exception:
                log.error("Failed to produce summary data", exc_info=True)

        return None

    def _max_diff(self):
        a = self.data
        if isinstance(a, (np.ndarray, xr.DataArray)) and a.size > 1:
            if np.issubdtype(a.dtype, np.bool_):
                return 1. if (True in a) and (False in a) else 0.
            return np.abs(np.subtract(np.nanmax(a), np.nanmin(a)), dtype=np.float64)

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


class Skip(Exception):
    pass
