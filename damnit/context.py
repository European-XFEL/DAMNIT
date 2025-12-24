import sys
from pathlib import Path

ctxsupport_dir = str(Path(__file__).parent / 'ctxsupport')
if ctxsupport_dir not in sys.path:
    sys.path.insert(0, ctxsupport_dir)

# Exposing these here for compatibility
from damnit_ctx import RunData, Variable
from damnit_writing import DataType, PNGData
from ctxrunner import (
    ContextFileErrors, ContextFile, Results,
    add_to_h5_file, get_proposal_path,
)
