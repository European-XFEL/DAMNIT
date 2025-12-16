import textwrap
from unittest.mock import patch


from damnit.backend.db import DamnitDB
from damnit.backend.extract_data import ReducedData, RunExtractor
from damnit.cli import main
from damnit.context import ContextFile


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
    with patch("damnit.backend.extract_data.KafkaProducer"):
        main(args)

def extract_mock_run(run_num: int, match=()):
    """Run the context file in the CWD on the specified run"""
    with patch("damnit.backend.extract_data.KafkaProducer"):
        db = DamnitDB()
        prop = db.metameta['proposal']
        extr = RunExtractor(prop, run_num, match=match, mock=True)
        extr.update_db_vars()
        extr.extract_and_ingest()


def mkcontext(code):
    return ContextFile.from_str(textwrap.dedent(code))
