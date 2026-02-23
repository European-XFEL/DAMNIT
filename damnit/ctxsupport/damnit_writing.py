import io
import os
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from tempfile import mkstemp

import h5py
import numpy as np

from damnit_ctx import Cell, isinstance_no_import

OBJTYPE_ATTR = '_damnit_objtype'
THUMBNAIL_SIZE = 300 # px
COMPRESSION_OPTS = {'compression': 'gzip', 'compression_opts': 1, 'shuffle': True}

if "AMORE_BROKER" in os.environ:
    UPDATE_BROKERS = [os.environ["AMORE_BROKER"]]
else:
    UPDATE_BROKERS = ['exflwgs06.desy.de:9091']


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


def downsample_line(data):
    from fpcs import downsample

    if isinstance_no_import(data, 'xarray', 'DataArray'):
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


# More specific Python types beyond what HDF5/NetCDF4 know about, so we can
# reconstruct Python objects when reading values back in.
class DataType(Enum):
    DataArray = "dataarray"
    Dataset = "dataset"
    Image = "image"
    Timestamp = "timestamp"
    PlotlyFigure = "PlotlyFigure"


@contextmanager
def atomic_create_h5(dir, prefix):
    """Write to a new HDF5 file, renaming it when finished."""
    fd, tmp_path = mkstemp(dir=dir, prefix=prefix, suffix=".writing.h5")
    try:
        os.close(fd)
        final_path = tmp_path.removesuffix(".writing.h5") + ".ready.h5"
        with h5py.File(tmp_path, 'w') as f:
            # Minor hack: attach the path it will be renamed to to the h5py
            # File object so the caller can get it.
            f.final_path = final_path
            yield f

        os.replace(tmp_path, final_path)
    except:
        os.unlink(tmp_path)
        raise


def save_dataset_netcdf(f: h5py.File, group: str, dset):
    """Save an xarray DataSet in NetCDF4 format without reopening the file"""
    import h5netcdf
    from xarray.backends import H5NetCDFStore

    # HDF5 doesn't allow slashes in names :(
    vars_names = {}
    for var_name, dataarray in dset.items():
        if var_name is not None and "/" in var_name:
            vars_names[var_name] = var_name.replace("/", "_")
    dset = dset.rename_vars(vars_names)

    with h5netcdf.File(f, 'a') as nf:
        store = H5NetCDFStore(nf, group=group, mode='w')
        dset.dump_to_store(store, encoding={k: COMPRESSION_OPTS for k in dset})


def save_dataarray_netcdf(f: h5py.File, group: str, darr):
    """Save an xarray DataArray in NetCDF4 format without reopening the file"""
    # ----------------
    # This block of code is from xarray.DataArray.to_netcdf
    # Copyright 2014-2024 xarray Developers
    # Used under the Apache 2.0 license
    from xarray.backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE

    if darr.name is None:
        # If no name is set then use a generic xarray name
        dataset = darr.to_dataset(name=DATAARRAY_VARIABLE)
    elif darr.name in darr.coords or darr.name in darr.dims:
        # The name is the same as one of the coords names, which netCDF
        # doesn't support, so rename it but keep track of the old name
        dataset = darr.to_dataset(name=DATAARRAY_VARIABLE)
        dataset.attrs[DATAARRAY_NAME] = darr.name
    else:
        # No problems with the name - so we're fine!
        dataset = darr.to_dataset()
    # ------------

    save_dataset_netcdf(f, group, dataset)


class DamnitFileWriter:
    def __init__(self, file: h5py.File):
        self.file = file
        file.require_group('.reduced')  # Summaries
        file.require_group('.preview')
        file.require_group('.errors')

    def store_summary(self, name, obj, attrs):
        if isinstance(obj, PNGData):  # PNG thumbnail
            obj = np.frombuffer(obj.data, dtype=np.uint8)
        ds = self.file.create_dataset(f'.reduced/{name}', data=obj)
        ds.attrs.update(attrs)

    def store_preview(self, name, obj):
        """Convert the preview data to what we'll store in HDF5 (object, attrs)"""
        path = f'.preview/{name}'

        if isinstance_no_import(obj, 'xarray', 'DataArray'):
            attrs = {OBJTYPE_ATTR: DataType.DataArray.value}
            save_dataarray_netcdf(self.file, path, obj)
        else:
            if isinstance_no_import(obj, 'matplotlib.figure', 'Figure'):
                attrs = {OBJTYPE_ATTR: DataType.Image.value}
                obj = figure2array(obj)
            elif isinstance_no_import(obj, 'plotly.graph_objs', 'Figure'):
                attrs = {OBJTYPE_ATTR: DataType.PlotlyFigure.value}
                obj = np.frombuffer(obj.to_json().encode('utf-8'), dtype=np.uint8)
            else:
                attrs = {}

            self.file[path] = obj

        self.file[path].attrs.update(attrs)

    def store_data(self, name, obj):
        grp = self.file.create_group(name)
        if isinstance_no_import(obj, 'xarray', 'DataArray'):
            attrs = {OBJTYPE_ATTR: DataType.DataArray.value}
            save_dataarray_netcdf(self.file, name, obj)
        elif isinstance_no_import(obj, 'xarray', 'Dataset'):
            attrs = {OBJTYPE_ATTR: DataType.Dataset.value}
            save_dataset_netcdf(self.file, name, obj)
        else:
            if isinstance_no_import(obj, 'matplotlib.figure', 'Figure'):
                attrs = {OBJTYPE_ATTR: DataType.Image.value}
                obj = figure2array(obj)
            elif isinstance_no_import(obj, 'plotly.graph_objs', 'Figure'):
                attrs = {OBJTYPE_ATTR: DataType.PlotlyFigure.value}
                # we want to compress plotly figures in HDF5 files
                # so we need to convert the data to array of uint8
                obj = np.frombuffer(obj.to_json().encode('utf-8'), dtype=np.uint8)
            else:
                attrs = {}

            grp['data'] = obj

        grp.attrs.update(attrs)

    def store_error(self, name, exc: Exception):
        ds = self.file.create_dataset(f'.errors/{name}', data=str(exc))
        ds.attrs['type'] = type(exc).__name__


def save_fragment(damnit_dir: Path, proposal: int, run: int, vars: dict[str, Cell],
           errors: dict[str, Exception]):
    """Add one or more results into a DAMNIT store"""
    results_dir = damnit_dir / "extracted_data"
    results_dir.mkdir(parents=True, exist_ok=True)
    if results_dir.stat().st_uid == os.getuid():
        os.chmod(results_dir, 0o777)

    with atomic_create_h5(dir=results_dir, prefix=f"p{proposal}_r{run}.") as f:
        writer = DamnitFileWriter(f)
        for name, cell in vars.items():
            if (summary := cell.get_summary()) is not None:
                summary_attrs = cell.summary_attrs()
                if isinstance(summary, np.ndarray):
                    if summary.ndim == 2 and summary.shape[0] == 2:
                        summary_attrs["summary_type"] = "trendline"
                writer.store_summary(name, summary, summary_attrs)

            if cell.preview is not None:
                writer.store_preview(name, cell.preview)

            if cell.data is not None:
                writer.store_data(name, cell.data)

        for name, exc in errors.items():
            writer.store_error(name, exc)

    os.chmod(f.final_path, 0o666)

    return Path(f.final_path)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import plotly.express as px
    import xarray as xr
    from extra_data import SourceNameError
    from damnit_ctx import Cell

    results_folder = Path("test/extracted_data")
    results_folder.mkdir(parents=True, exist_ok=True)

    da = xr.DataArray(np.zeros((2, 4, 5)), dims=['x', 'y', 'z'])
    fig, ax = plt.subplots()
    ax.plot(np.arange(20))

    save_fragment(results_folder.parent, 1, 1, vars={
        'numpy': Cell(np.arange(10), summary_value=5, bold=True),
        'xarray': Cell(da, preview=da.mean('x')),
        'mpl': Cell(fig),
        'plotly': Cell(px.bar(x=['a', 'b', 'c'], y=[1, 3, 2]))
    }, errors={
        'sourceerr': SourceNameError("No source named FOO/BAR/BAZ in this data"),
    })
