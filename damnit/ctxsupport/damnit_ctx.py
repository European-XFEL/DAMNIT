"""Core DAMNIT context types and helpers.

This module is made available by manipulating sys.path

We aim to maintain compatibility with older Python 3 versions (currently 3.9+)
than the DAMNIT code in general, to allow running context files in other Python
environments.
"""
import contextvars
import logging
import re
import sys
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr

__all__ = [
    "Cell",
    "Group",
    "GroupError",
    "Pipeline",
    "RunData",
    "Skip",
    "Variable"
]

log = logging.getLogger(__name__)


THUMBNAIL_SIZE = 300 # px


_DEFAULT_PIPELINE_STATE = contextvars.ContextVar(
    "damnit_ctx_default_pipeline", default=None
)


@dataclass
class _PipelineState:
    pipeline: "Pipeline"
    explicit: bool = False
    used: bool = False


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


class Pipeline:
    """Select and execute a set of Variables/Groups with a shared context.

    A Pipeline collects Variables and Group instances, compiles them into a
    ContextFile, and executes them with a run source. The run source can be
    provided directly via ``data=...`` or opened from ``proposal`` and
    ``run_number`` using ``extra_data.open_run``.
    """

    def __init__(
            self,
            *,
            name: str | None = None,
            proposal: int | None = None,
            run_number: int | None = None,
            data: Any | None = None,
            input_vars: dict[str, Any] | None = None,
            _context: "ContextFile" = None,
    ):
        """Initialize a Pipeline.

        Args:
            name: name used for tracking pipelines.
            proposal: Proposal number for meta access or run opening.
            run_number: Run number for meta access or run opening.
            data: Object passed to Variable functions.
            input_vars: Mapping for input# dependencies.
            _context: Precompiled context (e.g. from Pipeline.from_str() or select()).
        """
        self.name = name
        self.proposal = proposal
        self.run_number = run_number
        self.data = data
        self.input_vars = dict(input_vars or {})
        # Compiled ContextFile for this pipeline.
        self._context = _context
        # Results from the last execute() call, if any.
        self._last_results = None

    @classmethod
    def _new_state(cls):
        return _PipelineState(pipeline=cls(name="default"))

    @classmethod
    def default(cls):
        """Return the default Pipeline for the current context execution.

        This is only meaningful while a context file is being executed by
        ContextFile.from_str(), which sets a ContextVar to scope default
        pipelines to that execution.
        """
        state = _DEFAULT_PIPELINE_STATE.get()
        if state is None:
            state = cls._new_state()
            _DEFAULT_PIPELINE_STATE.set(state)
        state.used = True
        return state.pipeline

    @classmethod
    def set_default(cls, pipeline):
        """Set an explicit default Pipeline for the current context execution.

        When set, auto-discovered Variables and Groups are ignored; only the
        explicitly provided pipeline is compiled.
        """
        if not isinstance(pipeline, cls):
            raise TypeError("Pipeline.set_default expects a Pipeline instance")
        state = _DEFAULT_PIPELINE_STATE.get()
        if state is None:
            state = cls._new_state()
            _DEFAULT_PIPELINE_STATE.set(state)
        state.pipeline = pipeline
        state.explicit = True
        state.used = True
        return pipeline

    def copy(self):
        clone = Pipeline(
            name=self.name,
            proposal=self.proposal,
            run_number=self.run_number,
            data=self.data,
            input_vars=self.input_vars,
            _context=self._context,
        )
        return clone

    def add(self, *items):
        """Add Variables or Group instances to this Pipeline.

        Items can be Variables, Group instances, or sequences of them.
        """
        _items = []

        def add_item(item):
            if isinstance(item, Variable) or is_group_instance(item):
                _items.append(item)
                return
            if isinstance(item, (str, bytes, bytearray)):
                raise TypeError(
                    "Pipeline.add accepts Variable or Group instances (or sequences of them)"
                )
            if isinstance(item, Sequence):
                for sub in item:
                    add_item(sub)
                return
            raise TypeError(
                "Pipeline.add accepts Variable or Group instances (or sequences of them)"
            )

        for item in items:
            add_item(item)

        self._build_context(items=_items)
        return self

    def with_context(
            self,
            *,
            name=None,
            proposal=None,
            run_number=None,
            data=None,
            input_vars=None,
    ):
        """Return a new Pipeline with updated context fields."""
        new_pipe = self.copy()
        if name is not None:
            new_pipe.name = name
        if proposal is not None:
            new_pipe.proposal = proposal
        if run_number is not None:
            new_pipe.run_number = run_number
        if data is not None:
            new_pipe.data = data
        if input_vars is not None:
            new_pipe.input_vars = dict(input_vars)
        return new_pipe

    def _normalize_run_data(self, value):
        if value is None:
            return RunData.ALL
        if isinstance(value, RunData):
            return value
        return RunData(value)

    def _build_context(self, code_override=None, items: list | None = None):
        from ctxrunner import _build_context_file

        base = self._context

        vars_by_name = {}
        if base is not None:
            vars_by_name.update(base.vars)

        group_items = []
        for item in items or []:
            if isinstance(item, Variable):
                existing = vars_by_name.get(item.name)
                if existing is not None and existing is not item:
                    raise GroupError(f"Duplicate variable name {item.name!r}")
                vars_by_name[item.name] = item
            elif is_group_instance(item):
                group_items.append(item)
            else:
                raise TypeError(
                    "Pipeline items must be Variable or Group instances"
                )

        if code_override is None:
            if base is not None:
                code_override = base.code
            if code_override is None:
                code_override = ""

        self._context = _build_context_file(
            vars_by_name=vars_by_name,
            code=code_override,
            group_items=group_items,
        )
        return self._context

    @classmethod
    def from_context_file(cls, path, *, name=None):
        """Create a Pipeline from a context file on disk."""
        from ctxrunner import ContextFile

        ctx = ContextFile.from_py_file(Path(path))
        return cls(name=name, _context=ctx)

    @classmethod
    def from_str(cls, code, *, path="<string>", name=None):
        """Create a Pipeline from a context string."""
        from ctxrunner import ContextFile

        ctx = ContextFile.from_str(code, path)
        return cls(name=name, _context=ctx)

    def select(self, *, variables=(), match=(), run_data=None, cluster=None):
        """Return a new Pipeline filtered to a subset of variables."""
        run_data = self._normalize_run_data(run_data)
        filtered = self.context.filter(
            run_data=run_data,
            cluster=cluster,
            name_matches=match,
            variables=variables,
        )

        new_pipe = self.copy()
        new_pipe._context = filtered
        return new_pipe

    def execute(self, *, data=None, input_vars=None):
        """Execute the Pipeline and return Results.

        If ``data`` or ``self.data`` is provided, it is passed directly to
        Variable functions as their first argument. Otherwise, ``proposal`` and
        ``run_number`` are used to open the run via ``extra_data``.
        """
        if self.proposal is None or self.run_number is None:
            raise ValueError("proposal and run_number must be set")

        ctx = self.context

        data_obj = data if data is not None else self.data
        if data_obj is None:
            import extra_data

            if any(var._data == 'proc' for var in ctx.vars.values()):
                try:
                    extra_data.open_run(self.proposal, self.run_number, data='proc')
                except FileNotFoundError:
                    log.warning("Proc data is unavailable, only raw variables will be executed.")
                    ctx = ctx.filter(run_data=RunData.RAW)

            data_obj = extra_data.open_run(self.proposal, self.run_number)

        merged_input = dict(self.input_vars)
        if input_vars is not None:
            if not isinstance(input_vars, dict):
                raise TypeError("input_vars should be a dict.")
            merged_input.update(input_vars)
        res = ctx.execute(data_obj, self.run_number, self.proposal, merged_input)
        self._last_results = res
        return res

    @property
    def results(self):
        """Return the Results from the last execution, if any."""
        return self._last_results

    @property
    def context(self):
        """Return the compiled ContextFile."""
        if self._context is None:
            self._build_context()
        return self._context

    @property
    def vars(self):
        """Return the compiled Variable mapping."""
        return self.context.vars

    def vars_to_dict(self, inc_transient=False):
        """Return variable metadata suitable for database storage."""
        return self.context.vars_to_dict(inc_transient=inc_transient)

    def save_hdf5(self, path, reduced_only=False):
        """Save the last Results to an HDF5 file."""
        if self._last_results is None:
            raise RuntimeError("No results available. Call execute() first.")
        self._last_results.save_hdf5(path, reduced_only=reduced_only)


@contextmanager
def pipeline_scope():
    """Context manager that scopes default pipeline state to a context exec."""

    # Initialize default pipeline state
    token = _DEFAULT_PIPELINE_STATE.set(Pipeline._new_state())
    try:
        # yield current default pipeline state
        yield _DEFAULT_PIPELINE_STATE.get()
    finally:
        # Reset default pipeline state
        _DEFAULT_PIPELINE_STATE.reset(token)
