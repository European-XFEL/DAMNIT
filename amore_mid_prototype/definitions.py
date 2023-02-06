import os

# Kafka for sending updates around
if "AMORE_BROKER" in os.environ:
    UPDATE_BROKERS = [os.environ["AMORE_BROKER"]]
else:
    UPDATE_BROKERS = ['exflwebstor01.desy.de:9102']

UPDATE_TOPIC = "amore-db-{}"  # Fill in ID stored in database

LOG_FORMAT = '%(asctime)4s  %(levelname)3s  %(name)3s  %(message)s'
