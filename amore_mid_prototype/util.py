import time
from enum import Enum
from datetime import datetime, timezone

import pandas as pd


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

    dt_naive = datetime.fromtimestamp(timestamp)
    dt_local = dt_naive.replace(tzinfo=timezone.utc).astimezone()
    return dt_local.strftime("%H:%M:%S %d/%m/%Y")
