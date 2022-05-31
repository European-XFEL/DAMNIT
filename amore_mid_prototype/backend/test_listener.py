import logging
from dataclasses import dataclass
from datetime import datetime
from glob import glob

from .listener import EventProcessor

log = logging.getLogger(__name__)

@dataclass
class DummyRecord:
    timestamp: int

class TestEventProcessor(EventProcessor):
    def run(self):
        while True:
            try:
                run = input(f'Run for proposal {self.proposal}: ')
                # If the user presses 'Enter' without typing the input will be empty
                if len(run) == 0:
                    continue

                path = glob(f'/gpfs/exfel/exp/*/*/p{self.proposal:>06}/raw/r{run:>04}')[0]
                inst, cycle = path.split('/')[4:6]
                msg = {'proposal': self.proposal, 'run': run, 'path': path,
                       'instrument': inst, 'cycle': cycle}
                record = DummyRecord(timestamp=int(datetime.utcnow().timestamp() * 1000))
                self.handle_migration_complete(record, msg)
            except EOFError:
                break  # Allow Ctrl-D to close it
            except Exception:
                log.error("Error processing event", exc_info=True)


def listen(topics):
    try:
        with TestEventProcessor(topics) as processor:
            processor.run()
    except KeyboardInterrupt:
        print("Stopping on Ctrl-C")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    listen(["correction_complete"])
