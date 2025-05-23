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
import traceback
from datetime import timezone
from enum import Enum
from functools import wraps
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from contextlib import contextmanager

import extra_data
import h5py
import numpy as np
import requests
import xarray as xr
import yaml

from damnit_ctx import RunData, Variable, Cell, Skip, isinstance_no_import

log = logging.getLogger(__name__)

THUMBNAIL_SIZE = 300 # px
COMPRESSION_OPTS = {'compression': 'gzip', 'compression_opts': 1, 'shuffle': True}

# More specific Python types beyond what HDF5/NetCDF4 know about, so we can
# reconstruct Python objects when reading values back in.
class DataType(Enum):
    DataArray = "dataarray"
    Dataset = "dataset"
    Image = "image"
    Timestamp = "timestamp"
    PlotlyFigure = "PlotlyFigure"


class MyMetadataClient:
    def __init__(self, proposal, timeout=10, init_server="https://exfldadev01.desy.de/zwop"):
        self.proposal = proposal
        self.timeout = timeout
        self._cached_data = {}

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

    def _cache(func):
        @wraps(func)
        def wrapper(self, run):
            key = (run, func.__name__)
            if key in self._cached_data:
                return self._cached_data[key]
            self._cached_data[key] = func(self, run)
            return self._cached_data[key]
        return wrapper

    @_cache
    def _run_info(self, run: int) -> dict[str, Any]:
        response = requests.get(f"{self.server}/api/mymdc/proposals/by_number/{self.proposal}/runs/{run}",
                                headers=self._headers, timeout=self.timeout)
        response.raise_for_status()
        json = response.json()
        if len(json["runs"]) == 0:
            raise RuntimeError(f"Couldn't get run information from mymdc for p{self.proposal}, r{run}")

        return json["runs"][0]

    @_cache
    def techniques(self, run: int) -> dict[str, Any]:
        run_info = self._run_info(run)
        response = requests.get(f'{self.server}/api/mymdc/runs/{run_info["id"]}',
                                headers=self._headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()['techniques']

    @_cache
    def sample_name(self, run: int) -> str:
        run_info = self._run_info(run)
        sample_id = run_info["sample_id"]
        response = requests.get(f"{self.server}/api/mymdc/samples/{sample_id}",
                                headers=self._headers, timeout=self.timeout)
        response.raise_for_status()

        return response.json()["name"]

    @_cache
    def run_type(self, run: int) -> str:
        run_info = self._run_info(run)
        experiment_id = run_info["experiment_id"]
        response = requests.get(f"{self.server}/api/mymdc/experiments/{experiment_id}",
                                headers=self._headers, timeout=self.timeout)
        response.raise_for_status()

        return response.json()["name"]


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
        return ContextFile.from_str(code, str(path.absolute()))

    @classmethod
    def from_str(cls, code: str, path='<string>'):
        d = {}
        codeobj = compile(code, path, 'exec')
        exec(codeobj, d)
        vars = {v.name: v for v in d.values() if isinstance(v, Variable)}
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
        mymdc = None

        for name in self.ordered_vars():
            t0 = time.perf_counter()
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
                        if dep_name in dep_results:
                            dep_data = dep_results[dep_name]
                            kwargs[arg_name] = dep_data
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
                        metadata = annotation.removeprefix('mymdc#')
                        kwargs[arg_name] = getattr(mymdc, metadata)(run_number)

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


def figure2array(fig):
    from matplotlib.backends.backend_agg import FigureCanvas

    canvas = FigureCanvas(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())


class PNGData:
    def __init__(self, data: bytes):
        self.data = data


def is_png_data(obj):
    # insinstance(obj, PNGData) returns false if the PNGData object has been
    # instantiated from the context file code, so we need to check the data
    # attribute.
    try:
        return obj.data.startswith(b'\x89PNG\r\n\x1a\n')
    except:
        return False


def figure2png(fig, dpi=None):
    bio = io.BytesIO()
    fig.savefig(bio, dpi=dpi, format='png')
    return PNGData(bio.getvalue())


def plotly2png(figure):
    """Generate a png from a Plotly Figure

    largest dimension set to THUMBNAIL_SIZE
    """
    from PIL import Image
    png_data = figure.to_image(format='png')
    # resize with PIL (scaling in plotly does not play well with text)
    img = Image.open(io.BytesIO(png_data))
    largest_dim = max(img.width, img.height)
    width = int(img.width / largest_dim * THUMBNAIL_SIZE)
    height = int(img.height / largest_dim * THUMBNAIL_SIZE)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    # convert to PNG
    buff = io.BytesIO()
    img.save(buff, format='PNG')
    return PNGData(buff.getvalue())


def generate_thumbnail(image):
    from matplotlib.figure import Figure

    # Create plot
    fig = Figure(figsize=(1, 1))
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    vmin = np.nanquantile(image, 0.01)
    vmax = np.nanquantile(image, 0.99)
    if isinstance(image, np.ndarray):
        ax.imshow(image, vmin=vmin, vmax=vmax)
    else:
        # Use DataArray's own plotting method
        image.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, add_colorbar=False)
    ax.axis('tight')
    ax.axis('off')
    ax.margins(0, 0)

    # The figure is 1 inch square, so setting the DPI to THUMBNAIL_SIZE will
    # save a figure of size THUMBNAIL_SIZE x THUMBNAIL_SIZE.
    return figure2png(fig, dpi=THUMBNAIL_SIZE)


def line_thumbnail(arr):
    from matplotlib.figure import Figure

    # width = 3 * height; roughly fits table cells
    fig = Figure(figsize=(2, 2/3))
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if isinstance(arr, np.ndarray):
        ax.plot(arr)
    else:
        # Use DataArray's own plotting method
        arr.plot(ax=ax)

    ax.axis('tight')
    ax.axis('off')
    ax.margins(0, 0)

    return figure2png(fig, dpi=THUMBNAIL_SIZE)


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
                    return line_thumbnail(data)
                except:
                    logging.error("Error generating thumbnail for %s", name, exc_info=True)
                    return "<thumbnail error>"
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

    def save_hdf5(self, hdf5_path, reduced_only=False):
        xarray_dsets = []
        dsets = []
        obj_type_hints = {}

        for name, cell in self.cells.items():
            summary_val = self.summarise(name)
            dsets.append((f'.reduced/{name}', summary_val, cell.summary_attrs()))
            dsets.append((f'.errors/{name}', None, {}))  # Delete any previous error
            if not reduced_only:
                obj = cell.data
                if isinstance(obj, (xr.DataArray, xr.Dataset)):
                    xarray_dsets.append((name, obj))
                    obj_type_hints[name] = (
                        DataType.DataArray if isinstance(obj, xr.DataArray)
                        else DataType.Dataset
                    )
                else:
                    if isinstance_no_import(obj, 'matplotlib.figure', 'Figure'):
                        value = figure2array(obj)
                        obj_type_hints[name] = DataType.Image
                    elif isinstance_no_import(obj, 'plotly.graph_objs', 'Figure'):
                        # we want to compress plotly figures in HDF5 files
                        # so we need to convert the data to array of uint8
                        value = np.frombuffer(obj.to_json().encode('utf-8'), dtype=np.uint8)
                        obj_type_hints[name] = DataType.PlotlyFigure
                    elif isinstance(obj, str):
                        value = obj
                    elif obj is None:
                        value = None  # Will delete any previous data in file
                    else:
                        value = np.asarray(obj)

                    dsets.append((f'{name}/data', value, {}))

                if (obj := cell.preview) is None:
                    # Delete any previous preview
                    dsets.append((f'.preview/{name}', None, {}))
                elif isinstance(obj, xr.DataArray):
                    xarray_dsets.append((f'.preview/{name}', obj))
                else:
                    attrs = {}
                    if isinstance_no_import(obj, 'matplotlib.figure', 'Figure'):
                        obj = figure2array(obj)
                        attrs['_damnit_objtype'] = DataType.Image.value
                    elif isinstance_no_import(obj, 'plotly.graph_objs', 'Figure'):
                        obj = np.frombuffer(obj.to_json().encode('utf-8'), dtype=np.uint8)
                        attrs['_damnit_objtype'] = DataType.PlotlyFigure.value
                    dsets.append((f'.preview/{name}', obj, attrs))

        for name, exc in self.errors.items():
            dsets.append((f'.errors/{name}', str(exc), {'type': type(exc).__name__}))

        log.info("Writing %d variables to %s",
                 len(self.cells), hdf5_path)

        # We need to open the files in append mode so that when proc Variable's
        # are processed after raw ones, the raw ones won't be lost.
        with add_to_h5_file(hdf5_path) as f:
            # Delete whole groups for the Variables we're modifying
            for name in self.cells.keys():
                if name in f:
                    del f[name]

            for grp_name, hint in obj_type_hints.items():
                f.require_group(grp_name).attrs['_damnit_objtype'] = hint.value

            f.require_group('.reduced')
            f.require_group('.errors')

            # Create datasets before filling them, so metadata goes near the
            # start of the file.
            for path, obj, attrs in dsets:
                # Delete the existing datasets so we can overwrite them
                if path in f:
                    del f[path]

                if obj is None:
                    continue  # Deleted without replacement
                elif isinstance(obj, str):
                    f.create_dataset(path, shape=(), dtype=h5py.string_dtype())
                elif is_png_data(obj):  # Thumbnail
                    f.create_dataset(path, shape=len(obj.data), dtype=np.uint8)
                elif obj.ndim > 0 and (
                        np.issubdtype(obj.dtype, np.number) or
                        np.issubdtype(obj.dtype, np.bool_)):
                    f.create_dataset(path, shape=obj.shape, dtype=obj.dtype, **COMPRESSION_OPTS)
                else:
                    f.create_dataset(path, shape=obj.shape, dtype=obj.dtype)

                f[path].attrs.update(attrs)

            # Fill with data
            for path, obj, _ in dsets:
                if obj is None:
                    continue  # Deleted dataset
                elif is_png_data(obj):
                    f[path][()] = np.frombuffer(obj.data, dtype=np.uint8)
                else:
                    f[path][()] = obj

        for name, obj in xarray_dsets:
            if isinstance(obj, xr.DataArray):
                # HDF5 doesn't allow slashes in names :(
                if obj.name is not None and "/" in obj.name:
                    obj.name = obj.name.replace("/", "_")
                obj = _set_encoding(obj)
            elif isinstance(obj, xr.Dataset):
                vars_names = {}
                for var_name, dataarray in obj.items():
                    if var_name is not None and "/" in var_name:
                        vars_names[var_name] = var_name.replace("/", "_")
                    dataarray = _set_encoding(dataarray)
                obj = obj.rename_vars(vars_names)

            obj.to_netcdf(
                hdf5_path,
                mode="a",
                format="NETCDF4",
                group=name,
                engine="h5netcdf",
            )


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
