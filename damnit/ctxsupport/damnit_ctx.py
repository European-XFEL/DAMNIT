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
from dataclasses import dataclass, fields, is_dataclass, make_dataclass
from enum import Enum
from functools import wraps
from keyword import iskeyword
from typing import Callable, Generator

import h5py
import numpy as np
import xarray as xr

__all__ = ["Cell", "Group", "RunData", "Variable"]

log = logging.getLogger(__name__)


THUMBNAIL_SIZE = 300 # px


def isinstance_no_import(obj, mod: str, cls: str):
    """Check if isinstance(obj, mod.cls) without loading mod"""
    m = sys.modules.get(mod)
    if m is None:
        return False

    return isinstance(obj, getattr(m, cls))


def is_valid_variable_name(name: str) -> bool:
    """Check if the name is a valid Variable name."""
    # must be a string
    if not isinstance(name, str):
        return False

    for part in name.split('.'):
        # must be a non-empty string
        if not part:
            return False
        # prevent using a Python keyword
        if iskeyword(part):
            return False
        # must be a valid Python identifier
        if not part.isidentifier():
            return False

    return True


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
        self.title = title
        self.tags = (tags,) if isinstance(tags, str) else tags
        self.description = description
        self.summary = summary
        self.cluster = cluster
        self.transient = transient
        self._data = data

    # @Variable() is used as a decorator on a function that computes a value
    def __call__(self, func):
        self.func = func
        self.name = func.__name__
        if self.title is None:
            self.title = self.name
        return self

    def check(self):
        problems = []
        if not is_valid_variable_name(self.name):
            problems.append(
                f"The name {self.name!r} is not a valid Variable name."
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


class Group:
    """Base class for grouping Variables.

    Use as a decorator, it transforms a class into a Group-aware dataclass.
    - `@Group` for default behavior.
    - `@Group(title='...', ...)` to provide parameters.
    """

    def __new__(cls, decorated_cls: type = None, **kwargs) -> Callable | type:
        """Allows `Group` to function as a decorator.

        It's called when you use `@Group` or `@Group(...)`.
        """
        # Ensures that when subclasses of Group are instantiated,
        # this decorator logic does not re-run. We only want it to run
        # when `Group` itself is called directly.
        if cls is not Group:
            return super().__new__(cls)

        def wrapper(wrapped_cls: type) -> type:
            return _new_group(wrapped_cls, kwargs)

        if decorated_cls is None:
            # Called as `@Group(...)`. Return the wrapper.
            return wrapper
        else:
            # Called as `@Group` without arguments. Decorate the class immediately.
            return wrapper(decorated_cls)

    def __init_subclass__(cls, **kwargs):
        """Prevents direct subclassing of Group by end-users.
        
        We allow it only for our internal `_GroupTemplate`
        """
        super().__init_subclass__(**kwargs)
        
        # Check if the new subclass's immediate parent is `Group`.
        if Group in cls.__bases__ and cls.__name__ != '_GroupTemplate':
            raise TypeError(
                f"Cannot subclass 'Group' directly (class {cls.__name__!r}). "
                "Use it as a decorator: @Group"
            )


@dataclass
class _GroupTemplate(Group):
    title: str = None
    tags: tuple[str] = None
    sep: str = '/'

    def __post_init__(self):
        """Post-initialization to ensure tags are always a tuple."""
        if isinstance(self.tags, str):
            self.tags = (self.tags,)

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

        new_var.name = f'{prefix}.{new_var.name}'
        new_var.title = f"{self.title or prefix}{self.sep}{new_var.title}"
        new_var.tags = self._merge_tags(new_var.tags)

        return new_var

    def _create_wrapper_function(self, var, annotations, signature):
        """Create a wrapper function that provides group context"""
        func = var.func

        @wraps(func)
        def wrapper(*args, **kwargs):
            args_in_func = inspect.signature(func).parameters

            # Call the original function with group context
            if next(iter(args_in_func)) == 'self':
                return func(self, *args, **kwargs)
            return func(*args, **kwargs)

        wrapper.__name__ = var.name
        wrapper.__annotations__ = annotations
        wrapper.__signature__ = signature
        wrapper.__doc__ = func.__doc__

        return wrapper

    def _variables(self) -> Generator[Variable, None, None]:
        """Get all variables in this group"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Variable):
                yield attr

    def _groups(self) -> Generator[tuple[str, Group], None, None]:
        """Get all variable groups in this group"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Group):
                yield attr_name, attr

    def variables(self, prefix: str) -> dict[str, Variable]:
        """Get all variables in this group"""
        _vars = {}
        for original_var in self._variables():
            var = self._init_group_variable(prefix, original_var)

            # edit annotations to include the group prefix
            annotations = var.annotations().copy()
            for arg_name, annotation in annotations.items():
                if isinstance(annotation, str) and annotation.startswith('self#'):
                    dep_name = annotation.removeprefix('self#')
                    # Dependency is in this instance -> prefix the dep_name
                    annotations[arg_name] = f'var#{prefix}.{dep_name}'

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
        for group_name, group in self._groups():
            group_prefix = f"{prefix}.{group_name}"
            # update title and tags for the group
            group.title = f"{self.title or prefix}{self.sep}{group.title or group_name}"
            group.tags = self._merge_tags(group.tags)
            _vars.update(group.variables(group_prefix))

        return _vars


GROUP_FIELD_NAMES = {f.name for f in fields(_GroupTemplate)}


def _new_group(cls: type, decorator_kwargs: dict) -> type:
    """Transform a class into a Group-based dataclass."""
    if invalid_args := set(decorator_kwargs.keys()).difference(GROUP_FIELD_NAMES):
        raise TypeError(f"Invalid Group arguments: {invalid_args}")

    cls_annotations = cls.__dict__.get('__annotations__', {})
    parent_fields = set()
    is_group_subclass = issubclass(cls, Group)

    if not is_group_subclass:
        if redefined := GROUP_FIELD_NAMES.intersection(cls_annotations):
             raise TypeError(f"Group fields {redefined} redefined in class {cls.__name__!r}")
    else:
        for base in cls.__mro__[1:]:
            if is_dataclass(base):
                parent_fields.update(base.__dataclass_fields__.keys())
        if redefined := parent_fields.intersection(cls_annotations):
            raise TypeError(f"Parent fields redefined with new annotations: {redefined}")

    if is_group_subclass:
        dataclass_cls = dataclass(cls)
    else:
        DynamicGroupBase = make_dataclass(
            f"_DynamicGroupFor_{cls.__name__}",
            fields=[(f.name, f.type, f) for f in fields(_GroupTemplate)],
            bases=(_GroupTemplate,),
        )
        attrs = {k: v for k, v in cls.__dict__.items() if k not in ('__dict__', '__weakref__')}
        new_cls = type(cls.__name__, (cls, DynamicGroupBase), attrs)
        dataclass_cls = dataclass(new_cls)

    if not decorator_kwargs:
        return dataclass_cls

    original_init = dataclass_cls.__init__
    init_sig = inspect.signature(original_init)

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        bound_args = init_sig.bind_partial(self, *args, **kwargs)
        final_kwargs = decorator_kwargs.copy()
        user_provided_args = bound_args.arguments
        user_provided_args.pop('self', None)
        final_kwargs.update(user_provided_args)
        original_init(self, **final_kwargs)

    dataclass_cls.__init__ = new_init
    return dataclass_cls


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
