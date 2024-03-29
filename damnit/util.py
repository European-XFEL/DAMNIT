import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype


class StatusbarStylesheet(Enum):
    NORMAL = "QStatusBar {}"
    ERROR = "QStatusBar {background: red; color: white; font-weight: bold;}"


def wait_until(condition, timeout=1):
    """
    Re-evaluate `condition()` until it either returns true or we've waited
    longer than `timeout`.
    """
    slept_for = 0
    sleep_interval = 0.2

    while slept_for < timeout and not condition():
        time.sleep(sleep_interval)
        slept_for += sleep_interval

    if slept_for >= timeout:
        raise TimeoutError("Condition timed out")


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