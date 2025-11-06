import io
import json
import os
import socket
from contextlib import contextmanager
from enum import Enum
from getpass import getuser
from pathlib import Path
from tempfile import mkstemp

import h5py
import numpy as np
from kafka import KafkaProducer

from damnit_ctx import Cell, isinstance_no_import

OBJTYPE_ATTR = '_damnit_objtype'
THUMBNAIL_SIZE = 300 # px
COMPRESSION_OPTS = {'compression': 'gzip', 'compression_opts': 1, 'shuffle': True}
KAFKA_TOPIC = "test.damnit.file_submissions"

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


# More specific Python types beyond what HDF5/NetCDF4 know about, so we can
# reconstruct Python objects when reading values back in.
class DataType(Enum):
    DataArray = "dataarray"
    Dataset = "dataset"
    Image = "image"
    Timestamp = "timestamp"
    PlotlyFigure = "PlotlyFigure"


def summary_to_store(summary):
    if isinstance(summary, bytes):  # PNG thumbnail
        return np.frombuffer(summary, dtype=np.uint8)
    return summary

def preview_to_store(obj):
    if isinstance_no_import(obj, 'matplotlib.figure', 'Figure'):
        return figure2array(obj), {OBJTYPE_ATTR: DataType.Image.value}
    elif isinstance_no_import(obj, 'plotly.graph_objs', 'Figure'):
        a = np.frombuffer(obj.to_json().encode('utf-8'), dtype=np.uint8)
        return a, {OBJTYPE_ATTR: DataType.PlotlyFigure.value}
    elif isinstance_no_import(obj, 'xarray', 'DataArray'):
        return obj, {OBJTYPE_ATTR: DataType.DataArray.value}
    return obj, {}  # Numpy/xarray array

def data_to_store(obj):
    if isinstance_no_import(obj, 'xarray', 'DataArray'):
        return obj, {OBJTYPE_ATTR: DataType.DataArray.value}
    elif isinstance_no_import(obj, 'xarray', 'Dataset'):
        return obj, {OBJTYPE_ATTR: DataType.Dataset.value}
    elif isinstance_no_import(obj, 'matplotlib.figure', 'Figure'):
        return figure2array(obj), {OBJTYPE_ATTR: DataType.Image.value}
    elif isinstance_no_import(obj, 'plotly.graph_objs', 'Figure'):
        # we want to compress plotly figures in HDF5 files
        # so we need to convert the data to array of uint8
        a = np.frombuffer(obj.to_json().encode('utf-8'), dtype=np.uint8)
        return a, {OBJTYPE_ATTR: DataType.PlotlyFigure.value}
    elif isinstance(obj, str) or obj is None:
        return obj, {}
    return np.asarray(obj), {}


@contextmanager
def atomic_create_h5(dir, prefix):
    fd, tmp_path = mkstemp(dir=dir, prefix=prefix, suffix=".writing.h5")
    try:
        os.close(fd)
        final_path = tmp_path.removesuffix(".writing.h5") + ".ready.h5"
        with h5py.File(tmp_path, 'w') as f:
            f.final_path = final_path
            yield f

        os.replace(tmp_path, final_path)
    except:
        os.unlink(tmp_path)
        raise


def save_dataset_netcdf(f: h5py.File, group: str, dset):
    import h5netcdf
    from xarray.backends import H5NetCDFStore
    from xarray.backends.api import dump_to_store

    with h5netcdf.File(f, 'a') as nf:
        store = H5NetCDFStore(nf, group=group, mode='w')
        dump_to_store(dset, store, encoding={k: COMPRESSION_OPTS for k in dset})


def save_dataarray_netcdf(f: h5py.File, group: str, darr):
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


def submit(damnit_dir: Path, proposal: int, run: int, vars: dict[str, Cell],
           errors: dict[str, tuple]):
    results_dir = damnit_dir / "extracted_data"
    with atomic_create_h5(dir=results_dir, prefix=f"p{proposal}_r{run}.") as f:
        f.require_group('.reduced')  # Summaries
        f.require_group('.preview')
        f.require_group('.errors')

        for name, cell in vars.items():
            if (summary := summary_to_store(cell.get_summary())) is not None:
                ds = f.create_dataset(f".reduced/{name}", data=summary)
                ds.attrs.update(cell.summary_attrs())

            preview, attrs = preview_to_store(cell.preview)
            if attrs.get(OBJTYPE_ATTR) == DataType.DataArray.value:
                f.require_group(f".preview/{name}").attrs.update(attrs)
                save_dataarray_netcdf(f, f".preview/{name}", preview)
            elif preview is not None:
                ds = f.create_dataset(f".preview/{name}", data=preview)
                ds.attrs.update(attrs)

            data, attrs = data_to_store(cell.data)
            objtype = attrs.get(OBJTYPE_ATTR)
            if objtype == DataType.DataArray.value:
                f.require_group(name).attrs.update(attrs)
                save_dataarray_netcdf(f, name, data)
            elif objtype == DataType.Dataset.value:
                f.require_group(name).attrs.update(attrs)
                save_dataset_netcdf(f, name, data)
            elif data is not None:
                if data.ndim > 0 and (
                    np.issubdtype(data.dtype, np.number) or
                    np.issubdtype(data.dtype, np.bool_)
                ):
                    kwargs = COMPRESSION_OPTS
                else:
                    kwargs = {}
                ds = f.create_dataset(f"{name}/data", data=data, **kwargs)
                ds.attrs.update(attrs)

        for name, (etype, msg) in errors.items():
            ds = f.create_dataset(f'.errors/{name}', data=msg)
            ds.attrs['type'] = etype

    # Announce via Kafka that this file is ready to be combined
    prod = KafkaProducer(
        bootstrap_servers=UPDATE_BROKERS,
        value_serializer=lambda d: json.dumps(d).encode('utf-8')
    )
    prod.send(KAFKA_TOPIC, {
        'damnit_dir': damnit_dir,
        'new_file': f.final_path,
        'proposal': proposal,  # int
        'run': run,  # int
        'computed_by': {
            'hostname': socket.gethostname(),
            'username': getuser(),
        }
    })
    prod.flush(timeout=10)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import plotly.express as px
    import xarray as xr

    results_folder = Path("test/extracted_data")
    results_folder.mkdir(parents=True, exist_ok=True)

    da = xr.DataArray(np.zeros((2, 4, 5)), dims=['x', 'y', 'z'])
    fig, ax = plt.subplots()
    ax.plot(np.arange(20))

    submit(results_folder.parent, 1, 1, vars={
        'numpy': Cell(np.arange(10), summary_value=5, bold=True),
        'xarray': Cell(da, preview=da.mean('x')),
        'mpl': Cell(fig),
        'plotly': Cell(px.bar(x=['a', 'b', 'c'], y=[1, 3, 2]))
    }, errors={
        'sourceerr': ('SourceNameError', "No source named FOO/BAR/BAZ in this data")
    })
