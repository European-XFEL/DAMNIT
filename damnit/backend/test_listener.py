import logging
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path

from damnit.backend.listener import EventProcessor
from damnit.backend.extract_data import RunData

log = logging.getLogger(__name__)

@dataclass
class DummyRecord:
    timestamp: int

class TestEventProcessor(EventProcessor):
    def run(self):
        print()
        print("Enter run numbers, optionally with the run data source appended.\n" \
              "e.g., '42 raw' to process raw data, '42 proc' for proc data, or '42 all' for all.\n"
              "If not present, the data source defaults to 'all'.")

        while True:
            try:
                run_and_data = input("Enter a '<proposal> <run> [raw|proc|all] run_path' string: ")
                # If the user presses 'Enter' without typing the input will be empty
                if len(run_and_data) == 0:
                    continue

                # Extract the proposal and run number, and optionally the run data source
                input_split = run_and_data.split()
                proposal = input_split[0]
                run = input_split[1]
                if len(input_split) == 3:
                    run_data = RunData(input_split[2])
                else:
                    run_data = RunData.ALL
                if len(input_split) == 4:
                    path = input_split[3]
                else:
                    path = glob(f'/gpfs/exfel/exp/*/*/p{proposal:>06}/raw/r{run:>04}')[0]

                # Create the fake Kafka message
                inst, cycle = path.split('/')[4:6]
                msg = {'proposal': proposal, 'run': run, 'path': path,
                       'instrument': inst, 'cycle': cycle}
                record = DummyRecord(timestamp=int(datetime.utcnow().timestamp() * 1000))

                # Run the Kafka message handlers
                if run_data == RunData.RAW:
                    self.handle_migration_complete(record, msg)
                elif run_data == RunData.PROC:
                    self.handle_run_corrections_complete(record, msg)
                elif run_data == RunData.ALL:
                    self.handle_migration_complete(record, msg)
                    self.handle_run_corrections_complete(record, msg)
            except EOFError:
                break  # Allow Ctrl-D to close it
            except Exception:
                log.error("Error processing event", exc_info=True)


def listen(db_path):
    try:
        with TestEventProcessor(db_path) as processor:
            processor.run()
    except KeyboardInterrupt:
        print("Stopping on Ctrl-C")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    listen(Path.cwd())
