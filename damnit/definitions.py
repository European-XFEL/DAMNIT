import os
from enum import StrEnum

# Kafka for sending updates around
if "AMORE_BROKER" in os.environ:
    UPDATE_BROKERS = [os.environ["AMORE_BROKER"]]
else:
    UPDATE_BROKERS = ['exflwgs06.desy.de:9091']

UPDATE_TOPIC = "test.damnit.db-{}"  # Fill in ID stored in database
FILE_SUBMIT_TOPIC = "test.damnit.file_submissions"

DEFAULT_CONTEXT_PYTHON = os.path.realpath("/gpfs/exfel/sw/software/euxfel-environment-management/current-python-env/bin/python")
DEFAULT_DAMNIT_PYTHON = "/gpfs/exfel/sw/software/xfel_anaconda3/amore-mid/.pixi/envs/default/bin/python"

# Attribute names
class VariableAttributes(StrEnum):
    PARAM_DEFAULT = "param_default"
    PARAM_VALUE_NEW_RUN = "param_value_new_run"
