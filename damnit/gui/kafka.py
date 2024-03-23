import pickle
import logging

from kafka import KafkaConsumer, KafkaProducer
from PyQt5 import QtCore

from ..backend.db import MsgKind, msg_dict
from ..definitions import UPDATE_BROKERS, UPDATE_TOPIC

log = logging.getLogger(__name__)


class UpdateAgent(QtCore.QObject):
    message = QtCore.pyqtSignal(object)

    def __init__(self, db_id: str) -> None:
        QtCore.QObject.__init__(self)
        self.update_topic = UPDATE_TOPIC.format(db_id)

        self.kafka_cns = KafkaConsumer(
            self.update_topic, bootstrap_servers=UPDATE_BROKERS
        )
        self.kafka_prd = KafkaProducer(bootstrap_servers=UPDATE_BROKERS,
                                       value_serializer=lambda d: pickle.dumps(d))
        self.running = False

    def listen_loop(self) -> None:
        self.running = True

        while self.running:
            # Note: this doesn't throw an exception on timeout, it just returns
            # an empty dict.
            topic_messages = self.kafka_cns.poll(timeout_ms=100)

            for topic, messages in topic_messages.items():
                for msg in messages:
                    try:
                        unpickled_msg = pickle.loads(msg.value)
                    except Exception:
                        log.error("Kafka event could not be un-pickled.", exc_info=True)
                        continue

                    self.message.emit(unpickled_msg)

    def run_values_updated(self, proposal, run, name, value):
        message = msg_dict(MsgKind.run_values_updated,
                           {
                               "proposal": proposal,
                               "run": run,
                               "values": {
                                   name: value
                               }
                           })

        # Note: the send() function returns a future that we don't await. This
        # is potentially dangerous, but the reason is because it would cause a
        # delay in the GUI and because people don't usually close the GUI
        # immediately after editing a variable so it's unlikely that the process
        # will die before the update is sent (as happened with the backend).
        self.kafka_prd.send(self.update_topic, message)

    def variable_set(self, name, title, description, variable_type):
        message = msg_dict(MsgKind.variable_set,
                           {
                               "name": name,
                               "title": title,
                               "attributes": None,
                               "type": variable_type
                           })
        self.kafka_prd.send(self.update_topic, message)

    def stop(self):
        self.running = False


if __name__ == "__main__":
    monitor = UpdateAgent("tcp://localhost:5556")

    for record in monitor.kafka_cns:
        print(record.value.decode())
