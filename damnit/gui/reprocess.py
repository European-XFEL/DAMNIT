import logging

from PyQt5 import QtCore
from queue import Queue
from time import sleep

from ..backend.extract_data import Extractor

log = logging.getLogger(__name__)

class Reprocessor(QtCore.QObject):
    color_scheme = QtCore.pyqtSignal(object, bool)
    failed_reprocessing = QtCore.pyqtSignal()

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
                    self.reprocess_finished = False
                    self.color_scheme.emit(run, self.reprocess_finished)
                    try:
                        extr = Extractor()
                        extr.extract_and_ingest(None, run)
                        self.reprocess_finished = True
                        self.color_scheme.emit(run, self.reprocess_finished)

                    except Exception: 
                        log.error(f'Can not reprocess run {run}', exc_info=True)
                        self.reprocess_finished = True
                        if self.reprocess_queue.empty():
                            self.color_scheme.emit(None, self.reprocess_finished)
                            self.failed_reprocessing.emit()
                        else:
                            self.color_scheme.emit(run, self.reprocess_finished)
                        continue

                    self.reprocess_finished = True
                    if self.reprocess_queue.empty():
                        self.color_scheme.emit(None, self.reprocess_finished)

                else:
                    sleep(0.1)

        except Exception: 
            log.error('An error occurred in the reprocessing loop', exc_info=True)

    def stop(self):
        self.running = False
