"""Machinery to run a context file in-process

This is invoked by damnit.backend.extract_data as a subprocess,
possibly running in a different Python interpreter. It will run a context file
(or part of one) and save the results
"""

import argparse
import functools
import inspect
import logging
import os
import pickle
import sys
import time
from datetime import timezone
import traceback
from pathlib import Path
from unittest.mock import MagicMock
from graphlib import CycleError, TopologicalSorter

import extra_data
import h5py
import numpy as np
import xarray
import sqlite3

from damnit_ctx import RunData, Variable, UserEditableVariable

log = logging.getLogger(__name__)


class ContextFile:

    def __init__(self, vars, code):
        self.vars = vars
        self.code = code

        # Check for cycles
        try:
            self.ordered_vars()
        except CycleError as e:
            # Tweak the error message to make it clearer
            raise CycleError(f"These Variables have cyclical dependencies, which is not allowed: {e.args[1]}") from e

        # Temporarily prevent User Editable Variables as dependencies
        user_var_deps = set()
        for name, var in self.vars.items():
            user_var_deps |= {dep for dep in self.all_dependencies(var) if isinstance(self.vars[dep], UserEditableVariable)}
        if len(user_var_deps):
            all_vars_quoted = ", ".join(f'"{name}"' for name in sorted(user_var_deps))
            raise ValueError(f"The following user-editable variables are used as dependencies, this is currently unsupported: {all_vars_quoted}")

        # Check for raw-data variables that depend on proc-data variables
        raw_vars = { name: var for name, var in self.vars.items()
                     if hasattr(var, "data") and var.data == RunData.RAW }
        for name, var in raw_vars.items():
            dependencies = self.all_dependencies(var)
            proc_dependencies = [dep for dep in dependencies
                                 if not hasattr(self.vars[dep], "data") or self.vars[dep].data == RunData.PROC]

            if len(proc_dependencies) > 0:
                # If we have a variable that depends on proc data but didn't
                # explicitly set `data`, then promote this variable to use proc
                # data.
                if var._data == None:
                    var._data = RunData.PROC.value
                # Otherwise, if the user explicitly requested raw data but the
                # variable depends on proc data, that's a problem with the
                # context file and we raise an exception.
                elif var._data == RunData.RAW.value:
                    raise RuntimeError(f"Variable '{name}' is triggered by migration of raw data by data='raw', "
                                       f"but depends on these Variables that require proc data: {', '.join(proc_dependencies)}\n"
                                       f"Either remove data='raw' for '{name}' or change it to data='proc'")

        # Check that no variables have duplicate titles
        titles = [var.title for var in self.vars.values() if var.title is not None]
        duplicate_titles = set([title for title in titles if titles.count(title) > 1])
        if len(duplicate_titles) > 0:
            bad_variables = [name for name, var in self.vars.items()
                             if var.title in duplicate_titles]
            raise RuntimeError(f"These Variables have duplicate titles between them: {', '.join(bad_variables)}")

    def ordered_vars(self):
        """
        Return a tuple of variables in the context file, topologically sorted.
        """
        vars_graph = { name: set(var.arg_dependencies().values()) for name, var in self.vars.items() }

        # Sort and return
        ts = TopologicalSorter(vars_graph)
        return tuple(ts.static_order())

    def all_dependencies(self, *variables):
        """
        Return a set of names of all dependencies (direct and indirect) of the
        passed Variable's.
        """
        dependencies = set()

        for var in variables:
            var_deps = set(var.arg_dependencies().values())
            dependencies |= var_deps

            if len(var_deps) > 0:
                for dep_name in var_deps:
                    dependencies |= self.all_dependencies(self.vars[dep_name])

        return dependencies

    @classmethod
    def from_py_file(cls, path: Path, external_vars={}):
        code = path.read_text()
        log.debug("Loading context from %s", path)
        return ContextFile.from_str(code, external_vars)

    @classmethod
    def from_str(cls, code: str, external_vars={}):
        d = {}
        exec(code, d)
        vars = {v.name: v for v in d.values() if isinstance(v, Variable)}
        log.debug("Loaded %d variables", len(vars))
        clashing_vars = vars.keys() & external_vars.keys()

        if len(clashing_vars) > 0:
            raise RuntimeError(f"These Variables have clashing names: {', '.join(clashing_vars)}")
        return cls(vars | external_vars, code)

    def filter(self, run_data=RunData.ALL, cluster=True, name_matches=()):
        new_vars = {}
        for name, var in self.vars.items():

            if not isinstance(var, Variable):
                continue

            title = var.title or name

            # If this is being triggered by a migration/calibration message for
            # raw/proc data, then only process the Variable's that require that data.
            data_match = run_data == RunData.ALL or var.data == run_data
            # Skip data tagged cluster unless we're in a dedicated Slurm job
            cluster_match = cluster or not var.cluster
            # Skip Variables that don't match the match list
            name_match = (len(name_matches) == 0
                          or any(m.lower() in title.lower() for m in name_matches))

            if data_match and cluster_match and name_match:
                new_vars[name] = var

        # Add back any dependencies of the selected variables
        new_vars.update({name: self.vars[name] for name in self.all_dependencies(*new_vars.values())})

        return ContextFile(new_vars, self.code)


def get_user_variables(conn):
    user_variables = {}
    rows = conn.execute("SELECT name, title, type, description, attributes FROM variables")
    name_to_pos = {ff[0] : ii for ii, ff in enumerate(rows.description)}
    for rr in rows:
        var_name = rr[name_to_pos["name"]]
        new_var = UserEditableVariable(
            var_name,
            title=rr[name_to_pos["title"]],
            variable_type=rr[name_to_pos["type"]],
            description=rr[name_to_pos["description"]],
            attributes=rr[name_to_pos["attributes"]]
        )
        user_variables[var_name] = new_var
    log.debug("Loaded %d user variables", len(user_variables))
    return user_variables


def get_start_time(xd_run):
    ts = xd_run.select_trains(np.s_[:1]).train_timestamps()[0]

    if np.isnan(ts):
        # If the timestamp information is not present (i.e. on old files), then
        # we take the timestamp from the oldest raw file as an approximation.
        files = sorted([f.filename for f in xd_run.files if "raw" in f.filename])
        first_file = Path(files[0])

        # Use the modified timestamp
        return first_file.stat().st_mtime
    else:
        # Convert np datetime64 [ns] -> [us] -> datetime -> float  :-/
        return np.datetime64(ts, 'us').item().replace(tzinfo=timezone.utc).timestamp()


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

    return Path(*p.parts[:7])


def add_to_h5_file(path) -> h5py.File:
    """Open the file with exponential backoff if it's locked"""
    ex = None

    for i in range(6):
        try:
            return h5py.File(path, 'a')
        except BlockingIOError as e:
            # File is locked for writing; wait 1, 2, 4, ... seconds
            time.sleep(2 ** i)
            ex = e

    # This should only be reached after all attempts to open the file failed
    raise ex


class Results:
    def __init__(self, data, ctx):
        self.data = data
        self.ctx = ctx
        self._reduced = None

    @classmethod
    def create(cls, ctx_file: ContextFile, inputs, run_number, proposal):
        res = {'start_time': np.asarray(get_start_time(inputs['run_data']))}

        def get_dep_or_default(var, arg_name, dep_name):
            """
            Helper function to get either the value returned from the dependency
            `dep_name` of `var`, if any, or the default value of the argument in
            the function signature.
            """
            value = res.get(dep_name)
            if value is None:
                value = inspect.signature(var.func).parameters[arg_name].default

            return value

        for name in ctx_file.ordered_vars():
            var = ctx_file.vars[name]

            try:
                # Add all variable dependencies
                kwargs = { arg_name: get_dep_or_default(var, arg_name, dep_name)
                           for arg_name, dep_name in var.arg_dependencies().items() }

                # Check for missing dependencies with no default value
                missing_deps = [key for key, value in kwargs.items() if value is inspect.Parameter.empty]
                if len(missing_deps) > 0:
                    log.warning(f"Skipping {name} because of missing dependencies: {', '.join(missing_deps)}")
                    continue

                # And all meta dependencies
                for arg_name, annotation in var.annotations().items():
                    if not annotation.startswith("meta#"):
                        continue

                    if annotation == "meta#run_number":
                        kwargs[arg_name] = run_number
                    elif annotation == "meta#proposal":
                        kwargs[arg_name] = proposal
                    elif annotation == "meta#proposal_path":
                        kwargs[arg_name] = get_proposal_path(inputs['run_data'])
                    else:
                        raise RuntimeError(f"Unknown path '{annotation}' for variable '{var.title}'")

                func = functools.partial(var.func, **kwargs)

                data = func(inputs)
                if not isinstance(data, (xarray.DataArray, str, type(None))):
                    data = np.asarray(data)
            except Exception:
                log.error("Could not get data for %s", name, exc_info=True)
            else:
                # Only save the result if it's not None
                if data is not None:
                    res[name] = data
        return Results(res, ctx_file)

    @staticmethod
    def _datasets_for_arr(name, arr):
        if isinstance(arr, xarray.DataArray):
            return [
                (f'{name}/data', arr.values),
            ] + [
                (f'{name}/{dim}', coords.values)
                for dim, coords in arr.coords.items()
            ]
        else:
            value = arr if isinstance(arr, str) else np.asarray(arr)
            return [
                (f'{name}/data', value)
            ]

    @property
    def reduced(self):
        if self._reduced is None:
            r = {}
            for name in self.data:
                v = self.summarise(name)
                if v is not None:
                    r[name] = v
            self._reduced = r
        return self._reduced

    def summarise(self, name):
        data = self.data[name]

        if isinstance(data, str):
            return data
        elif data.ndim == 0:
            return data
        elif data.ndim == 2 and self.ctx.vars[name].summary is None:
            from scipy import ndimage

            # For the sake of space and memory we downsample images to a
            # resolution of 150x150.
            zoom_ratio = 150 / max(data.shape)
            if zoom_ratio < 1:
                data = ndimage.zoom(np.nan_to_num(data),
                                    zoom_ratio)

            return data
        elif self.ctx.vars[name].summary is None:
            return f"{data.dtype}: {data.shape}"
        else:
            summary_method = self.ctx.vars[name].summary
            if summary_method is None:
                return None

            if isinstance(data, xarray.DataArray):
                data = data.data

            return np.asarray(getattr(np, summary_method)(data))

    def save_hdf5(self, hdf5_path, reduced_only=False):

        ctx_vars = self.ctx.vars
        implicit_vars = self.data.keys() - self.ctx.vars.keys()

        dsets = [(f'.reduced/{name}', v) for name, v in self.reduced.items() if name in implicit_vars or ctx_vars[name].store_result]
        if not reduced_only:
            for name, arr in self.data.items():
                if name in implicit_vars or ctx_vars[name].store_result:
                    dsets.extend(self._datasets_for_arr(name, arr))

        log.info("Writing %d variables to %d datasets in %s",
                 len(self.data), len(dsets), hdf5_path)

        # We need to open the files in append mode so that when proc Variable's
        # are processed after raw ones, the raw ones won't be lost.
        with add_to_h5_file(hdf5_path) as f:
            # Delete whole groups for the Variable's we're modifying
            for name in self.data.keys():
                if name in f:
                    del f[name]

            # Create datasets before filling them, so metadata goes near the
            # start of the file.
            for path, arr in dsets:
                # Delete the existing datasets so we can overwrite them
                if path in f:
                    del f[path]

                if isinstance(arr, str):
                    f.create_dataset(path, shape=(1,), dtype=h5py.string_dtype())
                else:
                    f.create_dataset(path, shape=arr.shape, dtype=arr.dtype)

            for path, arr in dsets:
                f[path][()] = arr

        if os.stat(hdf5_path).st_uid == os.getuid():
            os.chmod(hdf5_path, 0o666)

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
    exec_ap.add_argument('--save', action='append', default=[])
    exec_ap.add_argument('--save-reduced', action='append', default=[])

    ctx_ap = subparsers.add_parser("ctx", help="Evaluate context file and pickle it to a file")
    ctx_ap.add_argument("context_file", type=Path)
    ctx_ap.add_argument("out_file", type=Path)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    db_conn = sqlite3.connect('runs.sqlite', timeout=30)

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

        ctx_whole = ContextFile.from_py_file(Path('context.py'), external_vars = get_user_variables(db_conn))
        ctx = ctx_whole.filter(
            run_data=run_data, cluster=args.cluster_job, name_matches=args.match
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

        inputs = {
            'run_data' : run_dc,
            'db_conn' : db_conn
        }
        res = Results.create(ctx, inputs, args.run, args.proposal)

        for path in args.save:
            res.save_hdf5(path)
        for path in args.save_reduced:
            res.save_hdf5(path, reduced_only=True)
    elif args.subcmd == "ctx":
        error_info = None

        try:
            ctx = ContextFile.from_py_file(args.context_file, external_vars = get_user_variables(db_conn))

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
