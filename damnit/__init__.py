"""Prototype for extracting and showing metadata (AMORE project)"""

__version__ = '0.2.1'

from .api import Damnit, RunVariables, VariableData


def submit(proposal: int, run: int, variables, *, errors: dict[str, Exception] =None, damnit_dir=None):
    """Add some results into DAMNIT's store

    Args:
        proposal (int): Proposal number
        run (int): Run number
        variables (dict): Mapping of names to arrays or DAMNIT Cell objects.
        errors (dict): Mapping of names to exceptions, to make error messages
            visible in the table.
        damnit_dir (Path or str, optional): The DAMNIT directory to write into.
            If not specified, it will find the default directory for the
            relevant proposal.
    """
    from pathlib import Path
    from .api import find_proposal
    from .backend.extract_data import notify_new_file
    from .context import Cell, save_fragment

    if damnit_dir is None:
        damnit_dir = find_proposal(proposal) / "usr/Shared/amore"
    else:
        damnit_dir = Path(damnit_dir)

    variables = {k: (v if isinstance(v, Cell) else Cell(v))
                 for (k, v) in variables.items()}
    errors = errors or {}

    path = save_fragment(damnit_dir, proposal, run, variables, errors)
    notify_new_file(damnit_dir, proposal, run, str(path))
