import os

# Kafka for sending updates around
if "AMORE_BROKER" in os.environ:
    UPDATE_BROKERS = [os.environ["AMORE_BROKER"]]
else:
    UPDATE_BROKERS = ['exflwgs06.desy.de:9091']

UPDATE_TOPIC = "test.damnit.db-{}"  # Fill in ID stored in database
