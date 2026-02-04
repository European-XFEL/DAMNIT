"""Machinery to run a context file in-process

This is invoked by damnit.backend.extract_data as a subprocess,
possibly running in a different Python interpreter. It will run a context file
(or part of one) and save the results
"""

import argparse
import fnmatch
import inspect
import logging
import os
import pickle
import re
import sys
import time
import traceback
from collections import deque
from contextlib import contextmanager
from copy import copy
from datetime import timezone
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import extra_data
import extra_proposal
import h5py
import numpy as np
import xarray as xr

from damnit_ctx import (
    Cell, GroupBoundVariable, GroupError, RunData, Skip, Variable,
    _normalize_tags, is_group_instance, isinstance_no_import
)
from damnit_writing import (
    COMPRESSION_OPTS,
    figure2png,
    generate_thumbnail,
    line_thumbnail,
    plotly2png,
    submit,
)

log = logging.getLogger("ctxrunner")

THUMBNAIL_SIZE = 300 # px


class ContextFileErrors(RuntimeError):
    def __init__(self, problems):
        self.problems = problems

    def __str__(self):
        return "\n".join(self.problems)


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
    # Nested groups inherit names from the attribute path. Shorter paths take
    # precedence over deeper nesting.
    pending = deque()
    candidates = {}
    expanded_depth = {}

    def _record_candidate(group, path, depth):
        group_id = id(group)
        entry = candidates.get(group_id)
        if entry is None or depth < entry["depth"]:
            candidates[group_id] = {
                "group": group,
                "depth": depth,
                "names": {path},
            }
        elif depth == entry["depth"]:
            entry["names"].add(path)

    for var_name, value in context.items():
        if var_name.startswith("__"):
            continue
        if is_group_instance(value):
            group = value
            root_name = group.name or var_name
            _record_candidate(group, root_name, 0)
            pending.append((group, root_name, 0))

    while pending:
        group, path, depth = pending.popleft()
        group_id = id(group)

        entry = candidates.get(group_id)
        if entry is None or depth != entry["depth"]:
            continue

        if expanded_depth.get(group_id) == depth:
            continue
        expanded_depth[group_id] = depth

        for attr, value in vars(group).items():
            if is_group_instance(value):
                child_path = f"{group.name or path}.{attr}"
                _record_candidate(value, child_path, depth + 1)
                pending.append((value, child_path, depth + 1))

    for entry in candidates.values():
        group = entry["group"]
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
            group.title = group.name.replace('.', group.sep)


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

        # check if group name is valid
        _group_name(group)
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


def _make_missing_dependency_placeholder(group, name: str) -> Variable:
    """Create a fake Variable to replace missing group instances dependencies.

    The Variable name is namespaced to the group and a special __missing__
    section to make it clear in error messages that this is a placeholder for a
    missing dependency.
    """
    def _missing(run):
        return None

    var = Variable(transient=True)
    var(_missing)
    # ensure the name does not contain any glob characters
    safe_name = re.sub(r"[!?*\[\]]", "_", name)
    var.name = f'{group.name}.__missing__.{safe_name}'
    var.title = var.name
    return var


def _resolve_self_annotation(annotation: str, group):
    path = annotation.removeprefix("self#")
    target, _, remain = path.partition(".")

    if remain:
        target = getattr(group, target)
        if target is None:
            placeholder = _make_missing_dependency_placeholder(group, path)
            return f'var#{placeholder.name}', placeholder
        return _resolve_self_annotation(remain, target)

    try:
        var_ref = getattr(group, target)
    except AttributeError:
        if any(ch in target for ch in "*?["):
            # glob pattern, will be resolved by ContextRunner.
            return f'var#{group.name}.{target}', None
        raise GroupError(
            f"Attribute {target!r} on group {group.name!r} does not exist, "
            "but is referenced as a dependency in annotation."
        )

    if var_ref is None:
        placeholder = _make_missing_dependency_placeholder(group, target)
        return f'var#{placeholder.name}', placeholder

    if not isinstance(var_ref, (Variable, GroupBoundVariable)):
        raise GroupError(
            f"Attribute {target!r} on group {group.name!r} is of type "
            f"{type(var_ref).__name__!r}, but a Variable is required for a self# dependency."
        )

    return f'var#{var_ref.name}', None


def _expand_group(group):
    """Build Variables for a group instance, resolving "self#" dependencies.

    This binds each @Variable to the group instance, namespaces names and
    titles, and rewrites annotations to "var#..." dependencies.
    """
    if group.title is None:
        group.title = group.name

    var_defs: dict[str, Variable] = _collect_group_variables(group)
    expanded = {}

    for var_def in var_defs.values():
        # Create a per-instance Variable copy so name/title/annotations can be
        # rewritten without mutating the class definition.
        bound_func = var_def.func.__get__(group, type(group))
        new_var = copy(var_def)
        new_var(bound_func)
        new_var.name = f"{group.name}.{var_def.name}"
        new_var.title = f"{group.title}{group.sep}{var_def.title}"

        annotations = {}
        sig = inspect.signature(bound_func)
        original_annotations = getattr(var_def.func, "__annotations__", {})
        for arg_name, param in sig.parameters.items():
            annotation = original_annotations.get(arg_name, param.annotation)
            if not isinstance(annotation, str):
                continue
            if annotation.startswith("self#"):
                # Resolve group-relative dependencies, possibly across nested groups.
                resolved, placeholder = _resolve_self_annotation(annotation, group)
                annotations[arg_name] = resolved
                if placeholder is not None:
                    # missing dependency -> add placeholder Variable
                    expanded[placeholder.name] = placeholder

            else:
                annotations[arg_name] = annotation

        new_var.tags = set(group.tags) | set(_normalize_tags(new_var.tags))
        new_var._annotation_overrides = annotations
        expanded[new_var.name] = new_var

    return expanded


def expand_groups(
        context: dict[str, Any],
        existing_vars: dict[str, Variable] | None = None,
    ) -> dict[str, Variable]:
    """Return Variables generated from Group instances in a context file."""
    _assign_group_names(context)
    groups = _collect_group_instances(context.values())
    if not groups:
        return {}

    existing_names = set(existing_vars or {})
    seen_group_names = {}
    for group in groups:
        group_id = id(group)
        if group.name in seen_group_names and seen_group_names[group.name] != group_id:
            raise GroupError(
                f"Group name {group.name!r} is used by multiple group instances"
            )
        seen_group_names[group.name] = group_id

    expanded = {}
    for group in groups:
        group_vars = _expand_group(group)
        for var_name in group_vars:
            if var_name in existing_names or var_name in expanded:
                raise GroupError(
                    f"Duplicate variable name {var_name!r} from group "
                    f"{group.name!r}"
                )
        expanded.update(group_vars)

    return expanded


class ContextFile:

    def __init__(self, vars, code):
        self.vars = vars
        self.code = code

        # Check for cycles
        try:
            ordered_names = self.ordered_vars()
        except CycleError as e:
            # Tweak the error message to make it clearer
            raise CycleError(f"These Variables have cyclical dependencies, which is not allowed: {e.args[1]}") from e

        # 'Promote' variables to match characters of their dependencies
        for name in ordered_names:
            var = self.vars[name]
            deps = [self.vars[dep] for dep in self.all_dependencies(var)]
            if var._data is None and any(v.data == RunData.PROC for v in deps):
                var._data = RunData.PROC.value

            if any(v.cluster for v in deps):
                var.cluster = True

    def check(self):
        problems = []
        for name, var in self.vars.items():
            problems.extend(var.check())
            if var.data != RunData.RAW:
                continue
            proc_dependencies = [dep for dep in self.all_dependencies(var)
                                 if self.vars[dep].data == RunData.PROC]

            if proc_dependencies:
                if var.data == RunData.RAW:
                    problems.append(
                        f"Variable {name} is triggered by migration of raw data (data='raw'), "
                        f"but depends on these Variables that require proc data: {', '.join(proc_dependencies)}\n"
                        f"Remove data='raw' for {name} or change it to data='proc'"
                    )

        # Check that no variables have duplicate titles
        titles = [var.title for var in self.vars.values() if var.title is not None]
        duplicate_titles = set([title for title in titles if titles.count(title) > 1])
        if len(duplicate_titles) > 0:
            bad_variables = [name for name, var in self.vars.items()
                             if var.title in duplicate_titles]
            problems.append(
                f"These Variables have duplicate titles between them: {', '.join(bad_variables)}"
            )

        # Check that all mymdc dependencies are valid
        for name, var in self.vars.items():
            mymdc_args = var.arg_dependencies("mymdc#")
            for arg_name, annotation in mymdc_args.items():
                if annotation not in ["sample_name", "run_type", "techniques"]:
                    problems.append(f"Argument '{arg_name}' of variable '{name}' has an invalid MyMdC dependency: '{annotation}'")

        if problems:
            raise ContextFileErrors(problems)

    def direct_dependencies(self, variable: Variable) -> set[str]:
        """return a set of names of direct dependencies of the passed Variable
        """
        dependencies = set()
        for dependency in variable.arg_dependencies().values():
            # expand matching patterns to match all variable dependencies
            deps = fnmatch.filter(self.vars, dependency)
            if len(deps) == 0:
                raise KeyError(f"Missing dependency: {dependency!r} for {variable.name!r}")
            dependencies.update(deps)
        return dependencies

    def ordered_vars(self) -> tuple[str]:
        """
        Return a tuple of variables in the context file, topologically sorted.
        """
        ts = TopologicalSorter()

        for name, var in self.vars.items():
            ts.add(name, *self.direct_dependencies(var))

        return tuple(ts.static_order())

    def all_dependencies(self, *variables):
        """
        Return a set of names of all dependencies (direct and indirect) of the
        passed Variable's.
        """
        dependencies = set()

        for var in variables:
            var_deps = self.direct_dependencies(var)
            dependencies |= var_deps

            if len(var_deps) > 0:
                for dep_name in var_deps:
                    dependencies |= self.all_dependencies(self.vars[dep_name])

        return dependencies

    @classmethod
    def from_py_file(cls, path: Path):
        code = path.read_text()
        log.debug("Loading context from %s", path)
        return ContextFile.from_str(code, str(path.absolute()))

    @classmethod
    def from_str(cls, code: str, path='<string>'):
        d = {}
        codeobj = compile(code, path, 'exec')
        exec(codeobj, d)
        vars = {v.name: v for v in d.values() if isinstance(v, Variable)}
        vars.update(expand_groups(d, vars))
        log.debug("Loaded %d variables", len(vars))
        return cls(vars, code)

    def vars_to_dict(self, inc_transient=False):
        """Get a plain dict of variable metadata to store in the database
        
        args:
            inc_transient (bool): include transient Variables in the dict
        """
        return {
            name: {
                'title': v.title,
                'description': v.description,
                'tags': v.tags,
                'attributes': None,
                'type': None,
            }
            for (name, v) in self.vars.items()
            if not v.transient or inc_transient
        }

    def filter(self, run_data=RunData.ALL, cluster=None, name_matches=(), variables=()):
        new_vars = {}
        for name, var in self.vars.items():

            if not isinstance(var, Variable):
                continue

            title = var.title or name

            # If this is being triggered by a migration/calibration message for
            # raw/proc data, then only process the Variable's that require that data.
            data_match = run_data == RunData.ALL or var.data == run_data
            # Skip data tagged cluster unless we're in a dedicated Slurm job
            cluster_match = True if cluster is None else cluster == var.cluster

            if variables:  # --var: exact variable names (not titles)
                name_match = name in variables
            elif name_matches:  # --match: substring in variable titles
                name_match = any(m.lower() in title.lower() for m in name_matches)
            else:
                name_match = True  # No --var or --match specification

            if data_match and cluster_match and name_match:
                new_vars[name] = var

        # Add back any dependencies of the selected variables
        new_vars.update({name: self.vars[name] for name in self.all_dependencies(*new_vars.values())})

        return ContextFile(new_vars, self.code)

    def execute(self, run_data, run_number, proposal, input_vars) -> 'Results':
        dep_results = {'start_time': get_start_time(run_data)}
        res = {'start_time': Cell(dep_results['start_time'])}
        errors = {}
        metadata = None

        for name in self.ordered_vars():
            t0 = time.perf_counter()
            var = self.vars[name]

            try:
                kwargs = {}
                missing_deps = []
                missing_input = []

                annotations = var.annotations()
                for arg_name, param in inspect.signature(var.func).parameters.items():
                    # Not using param.annotation, because the annotation in the
                    # function def can be overridden for variables within a group.
                    annotation = annotations.get(arg_name, inspect.Parameter.empty)
                    if not isinstance(annotation, str):
                        continue

                    # Dependency within the context file
                    if annotation.startswith("var#"):
                        dep_name = annotation.removeprefix("var#")
                        match = fnmatch.filter(dep_results, dep_name)

                        if len(match) == 1 and match[0] == dep_name:
                            kwargs[arg_name] = dep_results[dep_name]
                        elif len(match) >= 1:
                            kwargs[arg_name] = {name: dep_results[name] for name in match}
                        elif param.default is inspect.Parameter.empty:
                            missing_deps.extend(fnmatch.filter(self.vars, dep_name))

                    # Input variable passed from outside
                    elif annotation.startswith("input#"):
                        inp_name = annotation.removeprefix("input#")
                        if inp_name in input_vars:
                            kwargs[arg_name] = input_vars[inp_name]
                        elif param.default is inspect.Parameter.empty:
                            missing_input.append(inp_name)

                    # Mymdc fields
                    elif annotation.startswith("mymdc#"):
                        if metadata is None:
                            metadata = extra_proposal.Proposal(proposal)[run_number]
                        field = annotation.removeprefix('mymdc#')
                        kwargs[arg_name] = getattr(metadata, field)()

                    elif annotation == "meta#run_number":
                        kwargs[arg_name] = run_number
                    elif annotation == "meta#proposal":
                        kwargs[arg_name] = proposal
                    elif annotation == "meta#proposal_path":
                        kwargs[arg_name] = get_proposal_path(run_data)
                    else:
                        raise RuntimeError(f"Unknown path '{annotation}' for variable '{var.title}'")

                if missing_deps:
                    log.warning(f"Skipping {name} because of missing dependencies: {', '.join(missing_deps)}")
                    # get error message from transient dependencies
                    error_message = ''
                    for dep in missing_deps:
                        if self.vars[dep].transient and dep in errors:
                            error_message += f'\ndependency ({dep}) failed: {repr(errors[f"{dep}"])}'
                    if error_message:
                        errors[name] = Exception(error_message)
                    continue
                elif missing_input:
                    log.warning(f"Skipping {name} because of missing input variables: {', '.join(missing_input)}")
                    continue

                cell = var.evaluate(run_data, kwargs)
            except Exception as e:
                sys.stdout.flush()  # As in the else block
                if isinstance(e, Skip):
                    log.error("Skipped %s: %s", name, e)
                else:
                    log.error("Could not get data for %s", name, exc_info=True)
                errors[name] = e
            else:
                # When output is going to a file, stdout is block buffered, so
                # this ensures that print()-ed messages appear before the log
                # message (on stderr) that the variable has finished.
                sys.stdout.flush()
                t1 = time.perf_counter()
                log.info("Computed %s in %.03f s", name, t1 - t0)
                if not var.transient:
                    res[name] = cell
                if cell.data is not None:
                    dep_results[name] = cell.data

        # remove transient results
        for name, var in self.vars.items():
            if var.transient:
                errors.pop(name, None)

        return Results(res, errors, self)


def get_start_time(xd_run):
    ts = xd_run.select_trains(np.s_[:1]).train_timestamps()[0]

    if np.isnan(ts):
        # If the timestamp information is not present (i.e. on old files), then
        # we take the timestamp from the oldest raw file as an approximation.
        files = sorted([f.filename for f in xd_run.files if "raw" in f.filename])
        if len(files) == 0:
            # Some old proposals also copy all the non-detector data into proc,
            # in which case the DataCollection will only open the proc files. In
            # this case we fall back to the timestamp of the proc files, which
            # should be pretty close to the timestamp of the raw files.
            files = sorted([f.filename for f in xd_run.files])
        first_file = Path(files[0])

        # Use the modified timestamp
        return first_file.stat().st_mtime
    else:
        # Convert np datetime64 [ns] -> [us] -> datetime -> float  :-/
        return np.datetime64(ts, 'us').item().replace(tzinfo=timezone.utc).timestamp()


def downsample_line(data):
    from fpcs import downsample

    if isinstance(data, xr.DataArray):
        y = data.values
        x = None
        if data.dims:
            dim = data.dims[0]
            if dim in data.coords:
                x = data.coords[dim].values
        if x is None or np.shape(x) != np.shape(y):
            x = np.arange(y.size)
    else:
        y = np.asarray(data)
        x = np.arange(y.size)

    if not np.issubdtype(x.dtype, np.number):
        if np.issubdtype(x.dtype, np.datetime64):
            x = x.astype("datetime64[ns]").astype("int64") / 1e9
        else:
            x = np.arange(y.shape[0])

    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    if x.size == 0:
        return np.empty((2, 0), dtype=np.float64)

    # ensure we have a monotonic increasing coordinates
    if x.size > 1:
        diffs = np.diff(x)
        if not np.all(diffs >= 0):
            order = np.argsort(x, kind="stable")
            x = x[order]
            y = y[order]

    # We aim to retain ~150 samples
    # (TODO: rather save a ratio of the data with upper bound?)
    # the fpcs algorithm retain ~1.25 point per sampling window
    ratio = max(1, int(x.size / (0.8 * 150)))
    xd, yd = downsample(x, y, ratio=ratio)
    return np.vstack((xd, yd))


def extract_error_info(exc_type, e, tb):
    lineno = -1
    offset = 0

    if isinstance(e, SyntaxError):
        # SyntaxError and its child classes are special, their
        # tracebacks don't include the line number.
        lineno = e.lineno
        offset = e.offset - 1
    else:
        # Look for the frame with a filename matching the context
        for frame in traceback.extract_tb(tb):
            if frame.filename == "<string>":
                lineno = frame.lineno
                break

    stacktrace = "".join(traceback.format_exception(exc_type, e, tb))
    return (stacktrace, lineno, offset)


def get_proposal_path(xd_run):
    files = [f.filename for f in xd_run.files]
    p = Path(files[0])

    return Path(*p.parts[:-3])


@contextmanager
def add_to_h5_file(path) -> h5py.File:
    """Open the file with exponential backoff if it's locked"""
    ex = None

    f = None
    for i in range(6):
        try:
            f = h5py.File(path, 'a')
            break
        except BlockingIOError as e:
            # File is locked for writing; wait 1, 2, 4, ... seconds
            time.sleep(2 ** i)
            ex = e

    if f is not None:
        try:
            yield f
        finally:
            f.close()

            if os.stat(path).st_uid == os.getuid():
                os.chmod(path, 0o666)
    elif ex is not None:
        # This should only be reached after all attempts to open the file failed
        raise ex


def _set_encoding(data_array: xr.DataArray) -> xr.DataArray:
    """Add default compression options to DataArray"""
    encoding = COMPRESSION_OPTS.copy()
    encoding.update(data_array.encoding)
    data_array.encoding = encoding
    return data_array


class Results:
    def __init__(self, cells, errors, ctx):
        self.cells = cells
        self.errors = errors
        self.ctx = ctx
        self._reduced = None

    @property
    def reduced(self):
        if self._reduced is None:
            r = {}
            for name in self.cells:
                v = self.summarise(name)
                if v is not None:
                    r[name] = v
            self._reduced = r
        return self._reduced

    def summarise(self, name):
        cell = self.cells[name]

        if (summary_val := cell.get_summary()) is not None:
            return summary_val

        # If a summary wasn't specified, try some default fallbacks
        data = cell.preview if (cell.preview is not None) else cell.data
        if isinstance(data, str):
            return data
        elif isinstance(data, xr.Dataset):
            size = data.nbytes / 1e6
            return f"Dataset ({size:.2f}MB)"
        elif isinstance_no_import(data, 'matplotlib.figure', 'Figure'):
            # For the sake of space and memory we downsample images to a
            # resolution of THUMBNAIL_SIZE pixels on the larger dimension.
            image_shape = data.get_size_inches() * data.dpi
            zoom_ratio = min(1, THUMBNAIL_SIZE / max(image_shape))
            try:
                return figure2png(data, dpi=(data.dpi * zoom_ratio))
            except:
                logging.error("Error generating thumbnail for %s", name, exc_info=True)
                return "<thumbnail error>"
        elif isinstance_no_import(data, 'plotly.graph_objs', 'Figure'):
            return plotly2png(data)

        elif isinstance(data, (np.ndarray, xr.DataArray)):
            if data.ndim == 0:
                return data
            elif data.ndim == 1:
                try:
                    return downsample_line(data)
                except ModuleNotFoundError:
                    logging.warning(
                        'Downsampling library not found for trendline generation'
                        ', falling back to thumbnail generation for %s', name
                    )
                    try:
                        # fall back to generating thumbnail
                        return line_thumbnail(data)
                    except:
                        logging.error(
                            "Error generating thumbnail for %s", name, exc_info=True)
                        return "<thumbnail error>"
                except:
                    logging.error(
                        "Error generating trendline for %s", name, exc_info=True)
                    return "<trendline error>"
            elif data.ndim == 2:
                if isinstance(data, np.ndarray):
                    data = np.nan_to_num(data)
                else:
                    data = data.fillna(0)

                try:
                    return generate_thumbnail(data)
                except:
                    logging.error("Error generating thumbnail for %s", name, exc_info=True)
                    return "<thumbnail error>"
            else:
                # Describe the full data (cell.data), not the preview data
                return f"{cell.data.dtype}: {cell.data.shape}"

        return None

    def save(self, damnit_dir: Path, proposal: int, run: int):
        submit(damnit_dir, proposal, run, self.cells, self.errors)


def mock_run():
    run = MagicMock()
    run.files = [MagicMock(filename="/tmp/foo/bar.h5")]
    run.train_ids = np.arange(10)

    def select_trains(train_slice):
        return run

    run.select_trains.side_effect = select_trains

    def train_timestamps():
        return np.array(run.train_ids + 1493892000000000000,
                        dtype="datetime64[ns]")

    run.train_timestamps.side_effect = train_timestamps

    return run


def main(argv=None):
    ap = argparse.ArgumentParser()
    subparsers = ap.add_subparsers(required=True, dest="subcmd")

    exec_ap = subparsers.add_parser("exec", help="Execute context file on a run")
    exec_ap.add_argument('proposal', type=int)
    exec_ap.add_argument('run', type=int)
    exec_ap.add_argument('run_data', choices=('raw', 'proc', 'all'))
    exec_ap.add_argument('--mock', action='store_true')
    exec_ap.add_argument('--cluster-job', action="store_true")
    exec_ap.add_argument('--match', action="append", default=[])
    exec_ap.add_argument('--var', action="append", default=[])
    exec_ap.add_argument('--damnit-dir')

    ctx_ap = subparsers.add_parser("ctx", help="Evaluate context file and pickle it to a file")
    ctx_ap.add_argument("context_file", type=Path)
    ctx_ap.add_argument("out_file", type=Path)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if args.subcmd == "exec":
        # Check if we have proc data
        proc_available = False
        if args.mock:
            # If we want to mock a run, assume it's available
            proc_available = True
        else:
            # Otherwise check with open_run()
            try:
                extra_data.open_run(args.proposal, args.run, data="proc")
                proc_available = True
            except FileNotFoundError:
                pass
            except Exception as e:
                log.warning(f"Error when checking if proc data available: {e}")

        run_data = RunData(args.run_data)
        if run_data == RunData.ALL and not proc_available:
            log.warning("Proc data is unavailable, only raw variables will be executed.")
            run_data = RunData.RAW

        ctx_whole = ContextFile.from_py_file(Path('context.py'))
        ctx_whole.check()
        ctx = ctx_whole.filter(
            run_data=run_data, cluster=args.cluster_job, name_matches=args.match,
            variables=args.var,
        )
        log.info("Using %d variables (of %d) from context file %s",
             len(ctx.vars), len(ctx_whole.vars),
             "" if args.cluster_job else "(cluster variables will be processed later)")

        if args.mock:
            run_dc = mock_run()
        else:
            # Make sure that we always select the most data possible, so proc
            # variables have access to raw data too.
            actual_run_data = RunData.ALL if run_data == RunData.PROC else run_data
            run_dc = extra_data.open_run(args.proposal, args.run, data=actual_run_data.value)

        res = ctx.execute(run_dc, args.run, args.proposal, input_vars={})

        res.save(args.damnit_dir, args.proposal, args.run)
    elif args.subcmd == "ctx":
        error_info = None

        try:
            ctx = ContextFile.from_py_file(args.context_file)

            # Strip the functions from the Variable's, these cannot always be
            # pickled.
            for var in ctx.vars.values():
                var.func = None
        except:
            ctx = None
            error_info = extract_error_info(*sys.exc_info())

        args.out_file.write_bytes(pickle.dumps((ctx, error_info)))


if __name__ == '__main__':
    main()
