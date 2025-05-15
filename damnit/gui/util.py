from enum import Enum
from pathlib import Path

import numpy as np
from sklearn.neighbors import KernelDensity

from ..context import add_to_h5_file

class StatusbarStylesheet(Enum):
    NORMAL = "QStatusBar {}"
    ERROR = "QStatusBar {background: red; color: white; font-weight: bold;}"


def icon_path(name):
    """
    Helper function to get the path to an icon file stored under ico/.
    """
    return str(Path(__file__).parent / "ico" / name)


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


def delete_variable(db, name):
    # Remove from the database
    db.delete_variable(name)

    # And the HDF5 files
    for h5_path in (db.path.parent / "extracted_data").glob("*.h5"):
        with add_to_h5_file(h5_path) as f:
            if name in f:
                del f[f".reduced/{name}"]
                del f[name]
