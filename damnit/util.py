import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype


def timestamp2str(timestamp):
    if timestamp is None or pd.isna(timestamp):
        return None

    dt_utc = datetime.fromtimestamp(timestamp, timezone.utc)
    dt_local = dt_utc.astimezone()
    return dt_local.strftime("%H:%M:%S %d/%m/%Y")


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


def isinstance_no_import(obj, mod: str, cls: str):
    """Check if isinstance(obj, mod.cls) without loading mod"""
    m = sys.modules.get(mod)
    if m is None:
        return False

    return isinstance(obj, getattr(m, cls))
