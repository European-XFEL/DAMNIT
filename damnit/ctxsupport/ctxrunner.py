"""Machinery to run a context file in-process

This is invoked by damnit.backend.extract_data as a subprocess,
possibly running in a different Python interpreter. It will run a context file
(or part of one) and save the results
"""

import argparse
import functools
import inspect
import io
import logging
import os
import pickle
import sys
import time
from datetime import timezone
import traceback
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock
from graphlib import CycleError, TopologicalSorter

from matplotlib.axes import Axes
from matplotlib.figure import Figure

import extra_data
import h5py
import numpy as np
import xarray as xr
import requests
import yaml

from damnit_ctx import RunData, Variable

log = logging.getLogger(__name__)

THUMBNAIL_SIZE = 300 # px


# More specific Python types beyond what HDF5/NetCDF4 know about, so we can
# reconstruct Python objects when reading values back in.
class DataType(Enum):
    DataArray = "dataarray"
    Dataset = "dataset"
    Image = "image"
    Timestamp = "timestamp"


class MyMetadataClient:
    def __init__(self, proposal, timeout=10, init_server="https://exfldadev01.desy.de/zwop"):
        self.proposal = proposal
        self.timeout = timeout
        self._cache = {}

        proposal_path = Path(extra_data.read_machinery.find_proposal(f"p{proposal:06d}"))
        credentials_path = proposal_path / "usr/mymdc-credentials.yml"
        if not credentials_path.is_file():
            params = {
                "proposal_no": str(proposal),
                "kinds": "mymdc",
                "overwrite": "false",
                "dry_run": "false"
            }
            response = requests.post(f"{init_server}/api/write_tokens", params=params, timeout=timeout)
            response.raise_for_status()

        with open(credentials_path) as f:
            document = yaml.safe_load(f)
            self.token = document["token"]
            self.server = document["server"]

        self._headers = { "X-API-key": self.token }

    def _run_info(self, run):
        key = (run, "run_info")
        if key not in self._cache:
            response = requests.get(f"{self.server}/api/mymdc/proposals/by_number/{self.proposal}/runs/{run}",
                                    headers=self._headers, timeout=self.timeout)
            response.raise_for_status()
            json = response.json()
            if len(json["runs"]) == 0:
                raise RuntimeError(f"Couldn't get run information from mymdc for p{self.proposal}, r{run}")

            self._cache[key] = json["runs"][0]

        return self._cache[key]

    def sample_name(self, run):
        key = (run, "sample_name")
        if key not in self._cache:
            run_info = self._run_info(run)
            sample_id = run_info["sample_id"]
            response = requests.get(f"{self.server}/api/mymdc/samples/{sample_id}",
                                    headers=self._headers, timeout=self.timeout)
            response.raise_for_status()

            self._cache[key] = response.json()["name"]

        return self._cache[key]

    def run_type(self, run):
        key = (run, "run_type")
        if key not in self._cache:
            run_info = self._run_info(run)
            experiment_id = run_info["experiment_id"]
            response = requests.get(f"{self.server}/api/mymdc/experiments/{experiment_id}",
                                    headers=self._headers, timeout=self.timeout)
            response.raise_for_status()

            self._cache[key] = response.json()["name"]

        return self._cache[key]


class ContextFileErrors(RuntimeError):
    def __init__(self, problems):
        self.problems = problems

    def __str__(self):
        return "\n".join(self.problems)


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

        # Check for raw-data variables that depend on proc-data variables
        for name, var in self.vars.items():
            if var.data != RunData.RAW:
                continue
            proc_dependencies = [dep for dep in self.all_dependencies(var)
                                 if self.vars[dep].data == RunData.PROC]

            if len(proc_dependencies) > 0:
                # If we have a variable that depends on proc data but didn't
                # explicitly set `data`, then promote this variable to use proc
                # data.
                if var._data == None:
                    var._data = RunData.PROC.value

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
                if annotation not in ["sample_name", "run_type"]:
                    problems.append(f"Argument '{arg_name}' of variable '{name}' has an invalid MyMdC dependency: '{annotation}'")

        if problems:
            raise ContextFileErrors(problems)

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
    def from_py_file(cls, path: Path):
        code = path.read_text()
        log.debug("Loading context from %s", path)
        return ContextFile.from_str(code)

    @classmethod
    def from_str(cls, code: str):
        d = {}
        exec(code, d)
        vars = {v.name: v for v in d.values() if isinstance(v, Variable)}
        log.debug("Loaded %d variables", len(vars))
        return cls(vars, code)

    def vars_to_dict(self):
        """Get a plain dict of variable metadata to store in the database"""
        return {
            name: {
                'title': v.title,
                'description': v.description,
                'attributes': None,
                'type': None,
            }
            for (name, v) in self.vars.items()
        }

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

    def execute(self, run_data, run_number, proposal, input_vars) -> 'Results':
        res = {'start_time': np.asarray(get_start_time(run_data))}
        mymdc = None

        for name in self.ordered_vars():
            var = self.vars[name]

            try:
                kwargs = {}
                missing_deps = []
                missing_input = []

                for arg_name, param in inspect.signature(var.func).parameters.items():
                    annotation = param.annotation
                    if not isinstance(annotation, str):
                        continue

                    # Dependency within the context file
                    if annotation.startswith("var#"):
                        dep_name = annotation.removeprefix("var#")
                        if dep_name in res:
                            kwargs[arg_name] = res[dep_name]
                        elif param.default is inspect.Parameter.empty:
                            missing_deps.append(dep_name)

                    # Input variable passed from outside
                    elif annotation.startswith("input#"):
                        inp_name = annotation.removeprefix("input#")
                        if inp_name in input_vars:
                            kwargs[arg_name] = input_vars[inp_name]
                        elif param.default is inspect.Parameter.empty:
                            missing_input.append(inp_name)

                    # Mymdc fields
                    elif annotation.startswith("mymdc#"):
                        if mymdc is None:
                            mymdc = MyMetadataClient(proposal)

                        mymdc_field = annotation.removeprefix("mymdc#")
                        if mymdc_field == "sample_name":
                            kwargs[arg_name] = mymdc.sample_name(run_number)
                        elif mymdc_field == "run_type":
                            kwargs[arg_name] = mymdc.run_type(run_number)

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
                    continue
                elif missing_input:
                    log.warning(f"Skipping {name} because of missing input variables: {', '.join(missing_input)}")
                    continue

                func = functools.partial(var.func, **kwargs)

                data = func(run_data)

                # If the user returns an Axes, save the whole Figure
                if isinstance(data, Axes):
                    data = data.get_figure()

                if not isinstance(data, (xr.Dataset, xr.DataArray, str, type(None), Figure)):
                    arr = np.asarray(data)
                    # Numpy will wrap any Python object, but only native arrays
                    # can be saved in HDF5, not those containing Python objects.
                    if arr.dtype.hasobject:
                        log.error(
                            "Variable %s returned %s which cannot be saved",
                            name, type(data)
                        )
                        data = None
                    else:
                        data = arr
            except Exception:
                log.error("Could not get data for %s", name, exc_info=True)
            else:
                # Only save the result if it's not None
                if data is not None:
                    res[name] = data
        return Results(res, self)


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


def figure2array(fig):
    from matplotlib.backends.backend_agg import FigureCanvas

    canvas = FigureCanvas(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())

class PNGData:
    def __init__(self, data: bytes):
        self.data = data

def figure2png(fig, dpi=None):
    bio = io.BytesIO()
    fig.savefig(bio, dpi=dpi, format='png')
    return PNGData(bio.getvalue())


def generate_thumbnail(image):
    from matplotlib.figure import Figure

    # Create plot
    fig = Figure(figsize=(1, 1))
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    vmin = np.nanquantile(image, 0.01)
    vmax = np.nanquantile(image, 0.99)
    ax.imshow(image, vmin=vmin, vmax=vmax, extent=(0, 1, 1, 0))
    ax.axis('tight')
    ax.axis('off')
    ax.margins(0, 0)

    image_shape = fig.get_size_inches() * fig.dpi
    zoom_ratio = min(1, THUMBNAIL_SIZE / max(image_shape))
    return figure2png(fig, dpi=(fig.dpi * zoom_ratio))


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
        is_array = isinstance(data, (np.ndarray, xr.DataArray))
        is_dataset = isinstance(data, xr.Dataset)

        if isinstance(data, str):
            return data
        elif is_dataset:
            size = data.nbytes / 1e6
            return f"Dataset ({size:.2f}MB)"
        elif is_array and data.ndim == 0:
            return data
        elif isinstance(data, Figure):
            # For the sake of space and memory we downsample images to a
            # resolution of 35 pixels on the larger dimension.
            image_shape = data.get_size_inches() * data.dpi
            zoom_ratio = min(1, THUMBNAIL_SIZE / max(image_shape))
            return figure2png(data, dpi=(data.dpi * zoom_ratio))
        elif is_array and data.ndim == 2 and self.ctx.vars[name].summary is None:
            from scipy import ndimage
            zoom_ratio = min(1, THUMBNAIL_SIZE / max(data.shape))
            data = ndimage.zoom(np.nan_to_num(data), zoom_ratio)
            return generate_thumbnail(data)
        elif self.ctx.vars[name].summary is None:
            return f"{data.dtype}: {data.shape}"
        else:
            summary_method = self.ctx.vars[name].summary
            if summary_method is None:
                return None

            if isinstance(data, xr.DataArray):
                data = data.data

            return np.asarray(getattr(np, summary_method)(data))

    def save_hdf5(self, hdf5_path, reduced_only=False):

        ctx_vars = self.ctx.vars
        implicit_vars = self.data.keys() - self.ctx.vars.keys()

        xarray_dsets = []
        obj_type_hints = {}
        dsets = [(f'.reduced/{name}', v) for name, v in self.reduced.items()]
        if not reduced_only:
            for name, obj in self.data.items():
                if isinstance(obj, (xr.DataArray, xr.Dataset)):
                    xarray_dsets.append((name, obj))
                    obj_type_hints[name] = (
                        DataType.DataArray if isinstance(obj, xr.DataArray)
                        else DataType.Dataset
                    )
                else:
                    if isinstance(obj, Figure):
                        value =  figure2array(obj)
                        obj_type_hints[name] = DataType.Image
                    elif isinstance(obj, str):
                        value = obj
                    else:
                        value = np.asarray(obj)

                    dsets.append((f'{name}/data', value))

        log.info("Writing %d variables to %d datasets in %s",
                 len(self.data), len(dsets), hdf5_path)

        # We need to open the files in append mode so that when proc Variable's
        # are processed after raw ones, the raw ones won't be lost.
        with add_to_h5_file(hdf5_path) as f:
            # Delete whole groups for the Variable's we're modifying
            for name in self.data.keys():
                if name in f:
                    del f[name]

            for grp_name, hint in obj_type_hints.items():
                f.require_group(grp_name).attrs['_damnit_objtype'] = hint.value

            # Create datasets before filling them, so metadata goes near the
            # start of the file.
            for path, obj in dsets:
                # Delete the existing datasets so we can overwrite them
                if path in f:
                    del f[path]

                if isinstance(obj, str):
                    f.create_dataset(path, shape=(), dtype=h5py.string_dtype())
                elif isinstance(obj, PNGData):  # Thumbnail
                    f.create_dataset(path, shape=len(obj.data), dtype=np.uint8)
                else:
                    f.create_dataset(path, shape=obj.shape, dtype=obj.dtype)

            # Fill with data
            for path, obj in dsets:
                if isinstance(obj, PNGData):
                    f[path][()] = np.frombuffer(obj.data, dtype=np.uint8)
                else:
                    f[path][()] = obj

            # Assign attributes for reduced datasets
            for name, data in self.data.items():
                reduced_ds = f[f".reduced/{name}"]

                if (
                    isinstance(data, (np.ndarray, xr.DataArray))
                    and data.size > 1
                ):
                    reduced_ds.attrs["max_diff"] = abs(
                        np.subtract(np.nanmax(data), np.nanmin(data), dtype=np.float64)
                    )

                var_obj = ctx_vars.get(name)
                if var_obj is not None:
                    reduced_ds.attrs['summary_method'] = var_obj.summary or ''

        for name, obj in xarray_dsets:
            # HDF5 doesn't allow slashes in names :(
            if isinstance(obj, xr.DataArray) and obj.name is not None and "/" in obj.name:
                obj.name = obj.name.replace("/", "_")
            elif isinstance(obj, xr.Dataset):
                data_vars = list(obj.keys())
                for var_name in data_vars:
                    dataarray = obj[var_name]
                    if dataarray.name is not None and "/" in dataarray.name:
                        dataarray.name = dataarray.name.replace("/", "_")

            obj.to_netcdf(hdf5_path, mode="a", format="NETCDF4", group=name, engine="h5netcdf")

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

        res = ctx.execute(run_dc, args.run, args.proposal, input_vars={})

        for path in args.save:
            res.save_hdf5(path)
        for path in args.save_reduced:
            res.save_hdf5(path, reduced_only=True)
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
