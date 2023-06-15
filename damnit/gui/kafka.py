import pickle
import logging

from kafka import KafkaConsumer
from PyQt5 import QtCore

from ..definitions import UPDATE_BROKERS, UPDATE_TOPIC

log = logging.getLogger(__name__)


class UpdateReceiver(QtCore.QObject):
    message = QtCore.pyqtSignal(object)

    def __init__(self, db_id: str) -> None:
        QtCore.QObject.__init__(self)

        self.kafka_cns = KafkaConsumer(
            UPDATE_TOPIC.format(db_id), bootstrap_servers=UPDATE_BROKERS
        )
        self.running = False

    def loop(self) -> None:
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

    def stop(self):
        self.running = False


if __name__ == "__main__":
    recevier = UpdateReceiver("tcp://localhost:5556")

    for record in recevier.kafka_cns:
        print(record.value.decode())
