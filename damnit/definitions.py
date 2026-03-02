import os

# Kafka for sending updates around
def update_brokers():
    if "AMORE_BROKER" in os.environ:
        return [os.environ["AMORE_BROKER"]]
    else:
        return ['exflwgs06.desy.de:9091']

UPDATE_TOPIC = "test.damnit.db-{}"  # Fill in ID stored in database

DEFAULT_CONTEXT_PYTHON = "/gpfs/exfel/sw/software/euxfel-environment-management/environments/202502/.pixi/envs/default/bin/python"
DEFAULT_DAMNIT_PYTHON = "/gpfs/exfel/sw/software/xfel_anaconda3/amore-mid/.pixi/envs/default/bin/python"
