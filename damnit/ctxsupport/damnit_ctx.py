"""Core DAMNIT context types and helpers.

This module is made available by manipulating sys.path

We aim to maintain compatibility with older Python 3 versions (currently 3.9+)
than the DAMNIT code in general, to allow running context files in other Python
environments.
"""
import contextvars
import inspect
import logging
import re
import sys
from contextlib import contextmanager
from collections.abc import Iterable, Sequence
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
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
        if instance is None:
            return self
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


# TODO: split module?


class GroupError(Exception):
    pass


def _normalize_group_tags(tags):
    if tags is None:
        return None
    if isinstance(tags, str):
        return (tags,)
    return tuple(tags)


def _merge_tags(group_tags, var_tags):
    if not group_tags and not var_tags:
        return None
    if not group_tags:
        return set(var_tags)
    if not var_tags:
        return set(group_tags)

    return set(group_tags) | set(var_tags)


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
        tags = _normalize_group_tags(tags)

        cls.__damnit_group__ = True
        cls.__damnit_group_config__ = RESERVED_GROUP_FIELDS.copy()
        cls.__damnit_group_config__["tags"] = tags

        annotations = dict(getattr(cls, "__annotations__", {}))
        for field_name in reserved_fields:
            annotations.pop(field_name, None)
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
            self.tags = _normalize_group_tags(self.tags)
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


def _group_name(group):
    name = group.name
    if isinstance(name, str) and name:
        return name
    if name is None:
        raise GroupError(
            f"Group instance of {type(group).__name__!r} has no name. "
            "Provide name=... or assign it to a variable in the context file."
        )
    raise GroupError(
        f"Group instance of {type(group).__name__!r} has invalid name {name!r}. "
        "Provide a non-empty string name."
    )


def _assign_group_names(context):
    # Group names can be assigned by context variable names after execution.
    # e.g.: mygroup = MyGroup() -> since MyGroup(name=...) wasn't set, 'mygroup'
    # becomes the name when the context is processed.
    group_refs = {}
    for var_name, value in context.items():
        if var_name.startswith("__"):
            continue
        if is_group_instance(value):
            entry = group_refs.setdefault(id(value), {"obj": value, "names": []})
            entry["names"].append(var_name)

    for entry in group_refs.values():
        group = entry["obj"]
        if group.name is None:
            names = sorted(entry["names"])
            if len(names) == 1:
                group.name = names[0]
            else:
                raise GroupError(
                    f"Group instance of {type(group).__name__!r} is assigned "
                    f"to multiple names {names} without an explicit name."
                )
        if group.title is None:
            group.title = group.name


def _collect_group_instances(objects):
    pending = [obj for obj in objects if is_group_instance(obj)]
    groups = []
    seen_ids = set()

    while pending:
        group = pending.pop()
        group_id = id(group)
        if group_id in seen_ids:
            continue
        seen_ids.add(group_id)
        groups.append(group)

        for value in vars(group).values():
            if is_group_instance(value):
                pending.append(value)

    return groups


def _collect_group_variables(group):
    vars_by_name = {}
    for cls in reversed(type(group).mro()):
        for name, value in cls.__dict__.items():
            if isinstance(value, Variable):
                vars_by_name[name] = value
    return vars_by_name


def _resolve_self_dependency(group, path):
    if not path:
        raise GroupError("Empty self# dependency")

    target = group
    parts = path.split(".")
    for attr in parts[:-1]:
        if not hasattr(target, attr):
            raise GroupError(
                f"Group {type(group).__name__!r} has no attribute {attr!r}"
            )
        target = getattr(target, attr)
        if target is None:
            return None, None, True
        if not is_group_instance(target):
            raise GroupError(
                f"Attribute {attr!r} on group {type(group).__name__!r} "
                "does not reference a Group instance"
            )

    return target, parts[-1], False


class _MissingDependency(Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"


def _resolve_group_attr_variable_name(group, dep_name, param):
    """Resolve a group attribute to a Variable name or missing sentinel."""
    attr = getattr(group, dep_name, None)
    if attr is None:
        if param.default is inspect.Parameter.empty:
            return _MissingDependency.REQUIRED
        return _MissingDependency.OPTIONAL
    if isinstance(attr, (Variable, GroupBoundVariable)):
        return attr.name
    raise GroupError(
        f"Group {type(group).__name__!r} attribute {dep_name!r} is not a Variable"
    )


def _resolve_self_annotation(group, var_defs, arg_name, annotation, param):
    """Resolve a self# annotation into a var# annotation + dependency metadata.

    Returns (resolved_annotation, internal_dep, drop_var, skip_annotation):
    - resolved_annotation: string or None
    - internal_dep: (arg_name, dep_name, required) tuple or None
    - drop_var: True if the variable should be removed entirely
    - skip_annotation: True if the argument should keep its default
    """
    target_group, dep_name, missing = _resolve_self_dependency(
        group, annotation.removeprefix("self#")
    )
    if missing:
        if param.default is inspect.Parameter.empty:
            return None, None, True, False
        return None, None, False, True

    if target_group is group and dep_name in var_defs:
        # Reference to another method-defined variable in this group.
        resolved = f"{_group_name(target_group)}.{dep_name}"
        required = param.default is inspect.Parameter.empty
        return f"var#{resolved}", (arg_name, dep_name, required), False, False

    if target_group is group:
        # Reference to a Variable field (or bound variable) on this group.
        resolved = _resolve_group_attr_variable_name(group, dep_name, param)
    else:
        # Reference to a Variable field on a nested group instance.
        resolved = _resolve_group_attr_variable_name(target_group, dep_name, param)

    if resolved is _MissingDependency.REQUIRED:
        return None, None, True, False
    if resolved is _MissingDependency.OPTIONAL:
        return None, None, False, True
    return f"var#{resolved}", None, False, False


def _expand_group(group):
    """Build Variables for a group instance, resolving "self#" dependencies.

    This binds each @Variable to the group instance, namespaces names and
    titles, and rewrites annotations to "var#..." dependencies. Group-internal
    dependencies are tracked so variables with missing required inputs are
    dropped, and optional dependencies are removed from annotations.
    """
    group_name = _group_name(group)
    if group.title is None:
        group.title = group_name

    var_defs = _collect_group_variables(group)
    var_infos = {}

    for method_name, var_def in var_defs.items():
        # Create a per-instance Variable copy so name/title/annotations can be
        # rewritten without mutating the class definition.
        bound_func = var_def.func.__get__(group, type(group))
        new_var = copy(var_def)
        new_var(bound_func)
        new_var.name = f"{group_name}.{var_def.name}"
        new_var.title = f"{group.title}{group.sep}{var_def.title}"

        annotations = {}
        internal_deps = []
        drop_var = False
        sig = inspect.signature(bound_func)
        original_annotations = getattr(var_def.func, "__annotations__", {})
        for arg_name, param in sig.parameters.items():
            annotation = original_annotations.get(arg_name, param.annotation)
            if not isinstance(annotation, str):
                continue
            if annotation.startswith("self#"):
                # Resolve group-relative dependencies, possibly across nested groups.
                resolved, internal_dep, drop, skip = _resolve_self_annotation(
                    group, var_defs, arg_name, annotation, param
                )
                if drop:
                    drop_var = True
                    break
                if skip:
                    continue
                annotations[arg_name] = resolved
                if internal_dep is not None:
                    internal_deps.append(internal_dep)
            else:
                annotations[arg_name] = annotation

        if drop_var:
            continue

        new_var.tags = _merge_tags(group.tags, new_var.tags)
        var_infos[method_name] = {
            "var": new_var,
            "annotations": annotations,
            "internal_deps": internal_deps,
        }

    active = set(var_infos)
    changed = True
    while changed:
        changed = False
        for method_name in list(active):
            info = var_infos[method_name]
            remaining_deps = []
            for arg_name, dep_name, required in info["internal_deps"]:
                if dep_name not in active:
                    # Drop variables that depend on missing required internal vars;
                    # otherwise remove the dependency and let defaults apply.
                    if required:
                        active.remove(method_name)
                        changed = True
                        remaining_deps = []
                        break
                    info["annotations"].pop(arg_name, None)
                else:
                    remaining_deps.append((arg_name, dep_name, required))
            info["internal_deps"] = remaining_deps

    expanded = {}
    for method_name in active:
        info = var_infos[method_name]
        info["var"]._annotation_overrides = info["annotations"]
        expanded[info["var"].name] = info["var"]

    return expanded


def expand_groups(context, existing_vars=None):
    """Return Variables generated from Group instances in a context file."""
    _assign_group_names(context)
    groups = _collect_group_instances(context.values())
    if not groups:
        return {}

    existing_names = set(existing_vars or {})
    seen_group_names = {}
    for group in groups:
        name = _group_name(group)
        group_id = id(group)
        if name in seen_group_names and seen_group_names[name] != group_id:
            raise GroupError(
                f"Group name {name!r} is used by multiple group instances"
            )
        seen_group_names[name] = group_id

    expanded = {}
    for group in groups:
        group_vars = _expand_group(group)
        for var_name in group_vars:
            if var_name in existing_names or var_name in expanded:
                raise GroupError(
                    f"Duplicate variable name {var_name!r} from group "
                    f"{_group_name(group)!r}"
                )
        expanded.update(group_vars)

    return expanded


class GroupBoundVariable:
    __damnit_group_bound__ = True

    def __init__(self, group, var_def):
        self._group = group
        self._var_def = var_def

    @property
    def name(self):
        return f"{_group_name(self._group)}.{self._var_def.name}"

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
            run_data: str | RunData | None = None,
            data: Any | None = None,
            input_vars: dict[str, Any] | None = None,
            _base_context: "ContextFile" = None,
            _code: str | None = None,
    ):
        """Initialize a Pipeline.

        Args:
            name: name used for tracking pipelines.
            proposal: Proposal number for meta access or run opening.
            run_number: Run number for meta access or run opening.
            run_data: raw/proc/all selection for execution.
            data: Object passed to Variable functions.
            input_vars: Mapping for input# dependencies.
            _base_context: Precompiled context for select().
            _code: Source code for the context file, if known.
        """
        self.name = name
        self.proposal = proposal
        self.run_number = run_number
        self.run_data = run_data
        self.data = data
        self.input_vars = dict(input_vars or {})
        # Items (Variable, Group) explicitly added via Pipeline.add()
        self._items = []
        # Compiled ContextFile used as the base for add()/select().
        self._base_context = _base_context
        # Source code string for the context file (used for to_file()).
        self._code = _code
        # Cached compiled ContextFile for this pipeline (invalidated by add()).
        self._context = _base_context
        # Results from the last execute() call, if any.
        self._last_results = None

    @classmethod
    def _new_state(cls):
        return {"pipeline": cls(name="default"), "explicit": False, "used": False}

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
        state["used"] = True
        return state["pipeline"]

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
        state["pipeline"] = pipeline
        state["explicit"] = True
        state["used"] = True
        return pipeline

    def copy(self):
        clone = Pipeline(
            name=self.name,
            proposal=self.proposal,
            run_number=self.run_number,
            run_data=self.run_data,
            data=self.data,
            input_vars=self.input_vars,
            _base_context=self._base_context,
            _code=self._code,
        )
        clone._items = list(self._items)
        clone._context = self._context
        return clone

    def add(self, *items):
        """Add Variables or Group instances to this Pipeline.

        Items can be Variables, Group instances, or iterables of them.
        """
        def add_item(item):
            if isinstance(item, Variable) or is_group_instance(item):
                self._items.append(item)
                return
            if isinstance(item, Iterable):
                for sub in item:
                    add_item(sub)
                return
            raise TypeError(
                "Pipeline.add accepts Variable or Group instances (or iterables of them)"
            )

        for item in items:
            add_item(item)
        self._context = None
        return self

    def with_context(
            self,
            *,
            name=None,
            proposal=None,
            run_number=None,
            run_data=None,
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
        if run_data is not None:
            new_pipe.run_data = run_data
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

    def _get_context(self):
        if self._context is None:
            self._context = self._build_context()
        return self._context

    def _build_context(self, code_override=None):
        from ctxrunner import ContextFile
        code = code_override if code_override is not None else self._code
        vars_by_name = {}
        if self._base_context is not None:
            vars_by_name.update(self._base_context.vars)
            if not code:
                code = self._base_context.code

        group_items = []
        for item in self._items:
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

        if group_items:
            unnamed = [group for group in group_items if group.name is None]
            if unnamed:
                raise GroupError(
                    "Group instance has no name. Provide name=... before adding to Pipeline."
                )
            seen_names = {}
            for group in group_items:
                existing = seen_names.get(group.name)
                if existing is not None and existing is not group:
                    raise GroupError(
                        f"Group name {group.name!r} is used by multiple group instances"
                    )
                seen_names[group.name] = group
            context = dict(seen_names)
            vars_by_name.update(expand_groups(context, vars_by_name))

        if code is None:
            code = ""
        
        ctx = ContextFile(vars_by_name, code)
        ctx.check()
        return ctx

    def compile(self):
        """Compile the Pipeline into a ContextFile."""
        return self._get_context()

    @classmethod
    def from_context_file(
            cls,
            path,
            *,
            name=None,
    ):
        """Create a Pipeline from a context file on disk."""
        from pathlib import Path
        from ctxrunner import ContextFile

        ctx = ContextFile.from_py_file(Path(path))
        ctx.check()
        pipe = cls(
            name=name,
            _base_context=ctx,
            _code=ctx.code,
        )
        pipe._context = ctx
        return pipe

    @classmethod
    def from_str(
            cls,
            code,
            *,
            path="<string>",
            name=None,
    ):
        """Create a Pipeline from a context string."""
        from ctxrunner import ContextFile

        ctx = build_context_from_code(code, path, ContextFile)
        ctx.check()
        pipe = cls(
            name=name,
            _base_context=ctx,
            _code=ctx.code,
        )
        pipe._context = ctx
        return pipe

    def select(self, *, variables=(), match=(), run_data=None, cluster=None):
        """Return a new Pipeline filtered to a subset of variables."""
        ctx = self._get_context()
        run_data = self._normalize_run_data(run_data or self.run_data)
        filtered = ctx.filter(
            run_data=run_data,
            cluster=cluster,
            name_matches=match,
            variables=variables,
        )
        new_pipe = Pipeline(
            name=self.name,
            proposal=self.proposal,
            run_number=self.run_number,
            run_data=run_data.value,
            data=self.data,
            input_vars=self.input_vars,
            _base_context=filtered,
            _code=filtered.code,
        )
        new_pipe._context = filtered
        return new_pipe

    def _open_run(self, proposal, run_number, run_data):
        try:
            import extra_data
        except ImportError as exc:
            raise RuntimeError(
                "extra_data is required to open runs; pass data=... instead"
            ) from exc

        if run_data == RunData.ALL:
            try:
                return extra_data.open_run(proposal, run_number, data="all"), run_data
            except FileNotFoundError:
                log.warning("Proc data unavailable, using raw only")
                return extra_data.open_run(proposal, run_number, data="raw"), RunData.RAW
        if run_data == RunData.PROC:
            return extra_data.open_run(proposal, run_number, data="all"), run_data
        return extra_data.open_run(proposal, run_number, data=run_data.value), run_data

    def execute(self, *, data=None, input_vars=None):
        """Execute the Pipeline and return Results.

        If ``data`` (or ``self.data``) is provided, it is passed directly to
        Variable functions as their first argument. Otherwise, ``proposal`` and
        ``run_number`` must be set to open the run via ``extra_data``.
        """
        run_data = self._normalize_run_data(self.run_data)

        data_obj = data if data is not None else self.data
        if data_obj is None:
            if self.proposal is None or self.run_number is None:
                raise ValueError(
                    "Pipeline.execute requires data=... or proposal/run_number"
                )
            data_obj, run_data = self._open_run(
                self.proposal, self.run_number, run_data
            )

        ctx = self._get_context().filter(run_data=run_data, cluster=None)
        merged_input = dict(self.input_vars)
        if input_vars is not None:
            merged_input.update(input_vars)
        res = ctx.execute(data_obj, self.run_number, self.proposal, merged_input)
        self._last_results = res
        return res

    @property
    def results(self):
        """Return the Results from the last execution, if any."""
        return self._last_results

    @property
    def vars(self):
        """Return the compiled Variable mapping."""
        return self._get_context().vars

    def vars_to_dict(self, inc_transient=False):
        """Return variable metadata suitable for database storage."""
        return self._get_context().vars_to_dict(inc_transient=inc_transient)

    def save(self, path, reduced_only=False):
        """Save the last Results to an HDF5 file."""
        if self._last_results is None:
            raise RuntimeError("No results available. Call execute() first.")
        self._last_results.save_hdf5(path, reduced_only=reduced_only)

    def to_file(self, path):
        """Write the context source code to a file."""
        ctx = self._get_context()
        if not ctx.code:
            raise RuntimeError("No context code available to write")
        from pathlib import Path

        Path(path).write_text(ctx.code)


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


def build_context_from_code(code, path, context_file_cls):
    """Execute context code and return ContextFile."""
    context = {}
    codeobj = compile(code, path, 'exec')
    with pipeline_scope() as state:
        exec(codeobj, context)

    vars_by_name = {v.name: v for v in context.values() if isinstance(v, Variable)}

    if state is None or not state.get("used"):
        from ctxrunner import ContextFile

        vars_by_name.update(expand_groups(context, vars_by_name))
        return ContextFile(vars_by_name, code)

    _assign_group_names(context)
    groups = [obj for obj in context.values() if is_group_instance(obj)]
    pipe = state["pipeline"]
    if not state.get("explicit"):
        pipe.add(*vars_by_name.values(), *groups)
    ctx = pipe._build_context(code_override=code)
    return ctx
