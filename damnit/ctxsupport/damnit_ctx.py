"""Core DAMNIT context types and helpers.

This module is made available by manipulating sys.path

We aim to maintain compatibility with older Python 3 versions (currently 3.9+)
than the DAMNIT code in general, to allow running context files in other Python
environments.
"""
import logging
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import h5py
import numpy as np
import xarray as xr

__all__ = [
    "Cell",
    "Group",
    "GroupError",
    "RunData",
    "Skip",
    "Variable"
]

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
        self._annotation_overrides = None

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
        self.name = func.__name__
        if self.title is None:
            self.title = self.name
        return self

    def __get__(self, instance, owner):
        if is_group_instance(instance):
            # Return a proxy that resolves the group name lazily after context exec.
            return GroupBoundVariable(instance, self)
        return self

    def check(self):
        problems = []
        if not all(part.isidentifier() for part in self.name.split(".")):
            problems.append(
                f"The variable name {self.name!r} is not a valid Python identifier or dotted name"
            )
        if self._data not in (None, "raw", "proc"):
            problems.append(
                f"data={self._data!r} for variable {self.name} (can be 'raw'/'proc')"
            )
        if self.tags is not None:
            if not isinstance(self.tags, Iterable) or not all(
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
        return { arg_name: annotation.removeprefix(prefix)
                 for arg_name, annotation in self.annotations().items()
                 if annotation.startswith(prefix) }

    def annotations(self):
        """
        Get all annotated arguments of this Variable (including meta arguments,
        unlike `Variable.dependencies()`).

        Returns a dict of argument names to their annotations.
        """
        if self._annotation_overrides is not None:
            return self._annotation_overrides
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
                elif not (np.issubdtype(preview.dtype, np.number) or preview.dtype == bool):
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


class GroupError(Exception):
    pass


def _normalize_tags(tags) -> tuple[str]:
    if tags is None:
        return ()
    if isinstance(tags, str):
        return (tags,)
    return tuple(tags)


def _inherit_group_config(cls):
    for base in cls.mro()[1:]:
        config = base.__dict__.get("__damnit_group_config__")
        if config is not None:
            return config
    return {}


def Group(
    _cls=None,
    *,
    tags: Iterable[str] | str | None = None,
):
    """Decorate a class to define a reusable group of Variables.

    Group instances are dataclasses, therefore a Group subclass must also be
    decorated with @Group. Instantiate them in the context file and access group
    variables as "<group>.<var>" names. Use "self#..." in Variable annotations
    to link to other group values or nested groups.

    A group has 4 default parameters, they must be set at instantiation (except
    for `tags`):
        name: str | None = None
            The group `name`. If None, the Group instance name assigned in the
            context file will be used.
        title: str | None = None
            The group `title`. If None, the group `name` will be used as title.
        tags: Iterable[str] | str | None = None
            Tags to merge into each variable's tags.
        sep: str = "/"
            The separator between group title and variable title when generating
            variable titles.

    A Group decorator can define the default value of the Group instance `tags`.
    `Tags` will be inherited from parent Group classes unless explicitly
    overridden.
    """
    RESERVED_GROUP_FIELDS = {
        "name": None,
        "title": None,
        "tags": None,
        "sep": "/",
    }

    def wrap(cls):
        # Prevent Group classes from shadowing internal config fields.
        reserved_fields = set(RESERVED_GROUP_FIELDS)
        defined_fields = set(getattr(cls, "__annotations__", {}))
        conflicts = sorted(reserved_fields & defined_fields)
        if conflicts:
            raise GroupError(
                "Group classes cannot define reserved fields: "
                f"{', '.join(conflicts)}"
            )
        explicit_attrs = reserved_fields & set(cls.__dict__)
        if explicit_attrs:
            conflicts = ", ".join(sorted(explicit_attrs))
            raise GroupError(
                "Group classes cannot override reserved attributes: "
                f"{conflicts}"
            )

        # inherit parents' Group tags if not redefined
        parent_config = _inherit_group_config(cls)
        nonlocal tags
        if tags is None:
            tags = parent_config.get("tags")
        tags = _normalize_tags(tags)

        cls.__damnit_group__ = True
        cls.__damnit_group_config__ = RESERVED_GROUP_FIELDS.copy()
        cls.__damnit_group_config__["tags"] = tags

        annotations = dict(getattr(cls, "__annotations__", {}))
        annotations.update({
            "name": str | None,
            "title": str | None,
            "tags": Iterable[str] | str | None,
            "sep": str,
        })
        cls.__annotations__ = annotations

        cls.name = field(default=RESERVED_GROUP_FIELDS["name"], kw_only=True)
        cls.title = field(default=RESERVED_GROUP_FIELDS["title"], kw_only=True)
        cls.sep = field(default=RESERVED_GROUP_FIELDS["sep"], kw_only=True)
        cls.tags = field(default=tags, kw_only=True)

        original_post_init = getattr(cls, "__post_init__", None)

        def __post_init__(self):
            self.tags = _normalize_tags(self.tags)
            if self.title is None:
                self.title = self.name

            if original_post_init is not None:
                original_post_init(self)

        cls.__post_init__ = __post_init__

        return dataclass(cls)

    if _cls is None:
        return wrap
    return wrap(_cls)


def is_group_instance(obj):
    """Return True if obj is an instance of a Group class."""
    return hasattr(type(obj), "__damnit_group__")


class GroupBoundVariable:
    """Proxy for accessing a Group `Variable` on an instance.

    1. It allows you to wire dependencies before the group’s final name is
       known, (because that name can be inferred from the global assignment
       after the context code runs) so we don't have to mutate or reuse the
       shared class-level Variable definition across instances by deferring the
       per-instance binding work until expand_groups().
    2. Prevents accidentally collecting this reference in case it leaks into the
       context top level namespace, which would cause duplicate variable
       definitions.

    The proxy forwards all other attribute access to the original `Variable`, so
    it behaves like a `Variable` for e.g. dependency wiring.
    """
    __damnit_group_bound__ = True

    def __init__(self, group, var_def):
        self._group = group
        self._var_def = var_def

    @property
    def name(self):
        return f"{self._group.name}.{self._var_def.name}"

    def __getattr__(self, attr):
        return getattr(self._var_def, attr)
