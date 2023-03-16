import os

# Kafka for sending updates around
if "AMORE_BROKER" in os.environ:
    UPDATE_BROKERS = [os.environ["AMORE_BROKER"]]
else:
    UPDATE_BROKERS = ['exflwebstor01.desy.de:9102']

# Fill in ID stored in database
UPDATE_TOPIC = "amore-db-{}"
GUI_UPDATE_TOPIC = "amore-gui-{}"
