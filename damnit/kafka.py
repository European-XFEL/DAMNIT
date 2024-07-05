import pickle
import logging

from kafka import KafkaProducer

from .backend.db import MsgKind, msg_dict
from .definitions import UPDATE_BROKERS

log = logging.getLogger(__name__)


class UpdateProducer:
    def __init__(self, default_topic) -> None:
        self.default_topic = default_topic

        self.kafka_prd = KafkaProducer(bootstrap_servers=UPDATE_BROKERS,
                                       value_serializer=lambda d: pickle.dumps(d))

    def run_values_updated(self, proposal, run, name, value, flush=False, topic=None):
        if topic is None:
            topic = self.default_topic

        message = msg_dict(MsgKind.run_values_updated,
                           {
                               "proposal": proposal,
                               "run": run,
                               "values": {
                                   name: value
                               }
                           })
        self.kafka_prd.send(topic, message)

        if flush:
            self.kafka_prd.flush()

    def variable_set(self, name, title, variable_type, flush=False, topic=None):
        if topic is None:
            topic = self.default_topic

        message = msg_dict(MsgKind.variable_set,
                           {
                               "name": name,
                               "title": title,
                               "attributes": None,
                               "type": variable_type
                           })
        self.kafka_prd.send(topic, message)

        if flush:
            self.kafka_prd.flush()
