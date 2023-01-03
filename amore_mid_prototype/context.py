import sys
from pathlib import Path

ctxsupport_dir = str(Path(__file__).parent / 'ctxsupport')
if ctxsupport_dir not in sys.path:
    sys.path.insert(0, ctxsupport_dir)

# Exposing these here for compatibility
from damnit_ctx import Variable, UserEditableVariable, RunData, types_map, get_type_from_name
from ctxrunner import ContextFile, Results, get_proposal_path, get_user_variables
