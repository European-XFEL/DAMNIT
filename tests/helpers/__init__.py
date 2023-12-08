import textwrap
from unittest.mock import patch

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from damnit.cli import main
from damnit.context import ContextFile
from damnit.backend.extract_data import ReducedData


def reduced_data_from_dict(input_dict):
    """
    Create a dictionary of name -> ReducedData objects for add_to_db().
    """

    return {
        name: ReducedData(data)
        for name, data in input_dict.items()
    }

def amore_proto(args):
    """
    Execute the main function with the specified arguments.

    It is the callers responsibility to make sure the PWD is a database
    directory.
    """
    with (patch("sys.argv", ["amore-proto", *args]),
          patch("damnit.backend.extract_data.KafkaProducer")):
        main()

def mkcontext(code, **kwargs):
    return ContextFile.from_str(textwrap.dedent(code), **kwargs)
