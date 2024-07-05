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
        message = msg_dict(MsgKind.run_values_updated,
                           {
                               "proposal": proposal,
                               "run": run,
                               "values": {
                                   name: value
                               }
                           })

        self._send(message, flush, topic)

    def variable_set(self, name, title, variable_type, flush=False, topic=None):
        message = msg_dict(MsgKind.variable_set,
                           {
                               "name": name,
                               "title": title,
                               "attributes": None,
                               "type": variable_type
                           })
        self._send(message, flush, topic)

    def processing_submitted(self, info, flush=False, topic=None):
        self._send(msg_dict(
            MsgKind.processing_state_set, info,
        ), flush, topic)

    def _send(self, message, flush, topic):
        if topic is None:
            topic = self.default_topic

        # Note: the send() function returns a future that we don't await
        # immediately, the caller is responsible for passing `flush=True` or
        # calling `.flush()` to ensure that all messages are sent.
        self.kafka_prd.send(topic, message)
        if flush:
            self.flush()

    def flush(self):
        self.kafka_prd.flush(timeout=10)
