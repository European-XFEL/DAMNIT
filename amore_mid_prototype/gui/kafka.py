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

    def loop(self) -> None:
        for record in self.kafka_cns:
            try:
                msg = pickle.loads(record.value)
            except Exception:
                log.error("Kafka event could not be un-pickled.", exc_info=True)
                continue

            self.message.emit(msg)


if __name__ == "__main__":
    recevier = UpdateReceiver("tcp://localhost:5556")

    for record in recevier.kafka_cns:
        print(record.value.decode())
