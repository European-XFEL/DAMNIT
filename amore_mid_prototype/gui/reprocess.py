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
        run = None
        try:
            while self.running:
                if not self.reprocess_queue.empty(): 
                    run = self.reprocess_queue.get()
                    log.info(f'Request to reprocess run {run} recieved')
                    try:
                        extr = Extractor()
                        extr.extract_and_ingest(None, run)

                    except Exception: 
                        log.error(f'Can not reprocess {run}', exc_info=True)
                        continue
                else:
                    sleep(0.1)

        except Exception: 
            log.error('An error occurred in the reprocessing loop', exc_info=True)

    def stop(self):
        self.running = False


