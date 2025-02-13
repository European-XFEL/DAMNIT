import glob
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
from sklearn.neighbors import KernelDensity

from .context import add_to_h5_file


class StatusbarStylesheet(Enum):
    NORMAL = "QStatusBar {}"
    ERROR = "QStatusBar {background: red; color: white; font-weight: bold;}"


def timestamp2str(timestamp):
    if timestamp is None or pd.isna(timestamp):
        return None

    dt_utc = datetime.fromtimestamp(timestamp, timezone.utc)
    dt_local = dt_utc.astimezone()
    return dt_local.strftime("%H:%M:%S %d/%m/%Y")


def icon_path(name):
    """
    Helper function to get the path to an icon file stored under ico/.
    """
    return str(Path(__file__).parent / "gui/ico" / name)


def make_finite(data):
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    return data.astype('object').fillna(np.nan)


def bool_to_numeric(data):
    if infer_dtype(data) == 'boolean':
        data = data.astype('float')
    return data


def fix_data_for_plotting(data):
    return bool_to_numeric(make_finite(data))


def delete_variable(db, name):
    # Remove from the database
    db.delete_variable(name)

    # And the HDF5 files
    for h5_path in glob.glob(f"{db.path.parent}/extracted_data/*.h5"):
        with add_to_h5_file(h5_path) as f:
            if name in f:
                del f[f".reduced/{name}"]
                del f[name]


def kde(x: np.ndarray, npoints: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """1D kernel density estimation.

    Args:
        x (np.ndarray): 1D array of data points.
        npoints (int, optional): Number of points to evaluate the KDE at. Defaults to 1000.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (x, y) coordinates of the KDE.
    """
    xplot = np.linspace(x.min(), x.max(), npoints)[:, np.newaxis]
    kde = KernelDensity().fit(x[:, None])
    log_dens = kde.score_samples(xplot)
    y = np.exp(log_dens)
    return xplot.squeeze(), y


def _format_number(value: float, value_range: float) -> str:
    """Format a number with appropriate precision based on the data range."""

    def _formatter(_value, _range):
        # Calculate appropriate precision based on range
        magnitude = abs(_range)
        if magnitude >= 1000:
            return f"{_value:.0f}"
        elif magnitude >= 100:
            return f"{_value:.1f}"
        elif magnitude >= 1:
            return f"{_value:.2f}"
        else:
            return str(_value)

    if value_range == 0:
        return _formatter(value, value)
    else:
        return _formatter(value, value_range)
