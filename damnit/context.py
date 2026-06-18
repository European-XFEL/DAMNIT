import sys
from pathlib import Path

ctxsupport_dir = str(Path(__file__).parent / 'ctxsupport')
if ctxsupport_dir not in sys.path:
    sys.path.insert(0, ctxsupport_dir)

from ctxrunner import (
    ContextFile, ContextFileErrors, Results, add_to_h5_file, get_proposal_path
)
# Exposing these here for compatibility
from damnit_ctx import (
    Cell, Group, GroupError, Pipeline, RunData, Skip, Variable
)
from damnit_writing import DataType, PNGData, generate_thumbnail, save_fragment

__all__ = [
    "add_to_h5_file",
    "Cell",
    "ContextFile",
    "ContextFileErrors",
    "DataType",
    "generate_thumbnail",
    "get_proposal_path",
    "Group",
    "GroupError",
    "Pipeline",
    "PNGData",
    "Results",
    "RunData",
    "save_fragment",
    "Skip",
    "Variable",
]
