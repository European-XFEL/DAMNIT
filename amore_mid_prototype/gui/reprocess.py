import logging

from PyQt5 import QtCore
from queue import Queue
from time import sleep

from ..backend.extract_data import Extractor

log = logging.getLogger(__name__)

class Reprocessor(QtCore.QObject):
    message = QtCore.pyqtSignal(object)

    def __init__(self):
        QtCore.QObject.__init__(self)

        self.reprocess_queue = Queue(500)
        self.running = False

    def loop(self) -> None:
        self.running = True
        run = 0

        try:
            while self.running:

                if not self.reprocess_queue.empty(): 
                    log.info(f'Request to reprocess run {run} recieved')
                    run = self.reprocess_queue.get()

                    try:
                        extr = Extractor()
                        extr.extract_and_injest(None, run)
                    except Exception: 
                        log.error(f'Can not reprocess {run}', exc_info=True)
                        continue
                else:
                    sleep(0.5)

        except Exception: 
            log.error('An error occurred in the reprocessing loop', exc_info=True)

    def stop(self):
        self.running = False


