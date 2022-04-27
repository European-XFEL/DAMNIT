import pickle

import zmq
from PyQt5 import QtCore


class ZmqStreamReceiver(QtCore.QObject):
    message = QtCore.pyqtSignal(object)

    def __init__(self, endpoint: str) -> None:
        QtCore.QObject.__init__(self)

        self.endpoint = endpoint
        self._is_getting_messages = False

        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect(self.endpoint)
        self.socket.subscribe(b"")

        self._is_connected = True

    def loop(self) -> None:
        while self._is_connected:
            msg = self.socket.recv()
            self.message.emit(pickle.loads(msg))

            self._is_getting_messages = True


if __name__ == "__main__":
    zmq_receiver = ZmqStreamReceiver("tcp://localhost:5556")

    while True:
        string = zmq_receiver.socket.recv_json()
        print(string)
