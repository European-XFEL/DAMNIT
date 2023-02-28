import getpass
import json
import logging
import os
import platform
import queue
import subprocess
import sys
from pathlib import Path
from socket import gethostname
from threading import Thread

from kafka import KafkaConsumer

from .db import open_db, get_meta
from .extract_data import RunData

# For now, the migration & calibration events come via DESY's Kafka brokers,
# but the AMORE updates go via XFEL's test instance.
# BROKERS_IN = [f'it-kafka-broker{i:02}.desy.de' for i in range(1, 4)]
CONSUMER_ID = 'xfel-da-amore-prototype-{}'
KAFKA_CONF = {
    'maxwell': {
        'brokers': [f'it-kafka-broker{i:02}.desy.de' for i in range(1, 4)],
        'topics': ["xfel-test-r2d2", "xfel-test-offline-cal"],
        'events': ["migration_complete", "correction_complete"],
    },
    'onc': {
        'brokers': ['exflwgs06:9091'],
        'topics': ['test.euxfel.sa2.hed.daq', 'test.euxfel.sa2.hed.cal'],
        'events': ['run_complete', 'correction_complete'],
    }
}

log = logging.getLogger(__name__)


def watch_processes_finish(q: queue.Queue):
    procs_by_prop_run = {}
    while True:
        # Get new subprocesses from the main thread
        try:
            prop, run, popen = q.get(timeout=1)
            procs_by_prop_run[prop, run] = popen
        except queue.Empty:
            pass

        # Check if any of the subprocesses we're tracking have finished
        to_delete = set()
        for (prop, run), popen in procs_by_prop_run.items():
            returncode = popen.poll()
            if returncode is None:
                continue  # Still running

            # Can't delete from a dict while iterating over it
            to_delete.add((prop, run))
            if returncode == 0:
                log.info("Data extraction for p%d r%d succeeded", prop, run)
            else:
                log.error(
                    "Data extraction for p%d, r%d failed with exit code %d",
                    prop, run, returncode
                )

        for prop, run in to_delete:
            del procs_by_prop_run[prop, run]


class EventProcessor:

    def __init__(self, context_dir=Path('.')):
        self.context_dir = context_dir
        self.db = open_db(context_dir / 'runs.sqlite')
        # Fail fast if read-only - https://stackoverflow.com/a/44707371/434217
        self.db.execute("pragma user_version=0;")
        self.proposal = get_meta(self.db, 'proposal')
        log.info(f"Will watch for events from proposal {self.proposal}")

        if gethostname().startswith('exflonc'):
            # running on the online cluster
            kafka_conf = KAFKA_CONF['onc']
        else:
            kafka_conf = KAFKA_CONF['maxwell']

        consumer_id = CONSUMER_ID.format(get_meta(self.db, 'db_id'))
        print(kafka_conf)
        self.kafka_cns = KafkaConsumer(*kafka_conf['topics'],
                                       bootstrap_servers=kafka_conf['brokers'],
                                       group_id=consumer_id)
        self.events = kafka_conf['events']

        self.extract_procs_queue = queue.Queue()
        self.extract_procs_watcher = Thread(
            target=watch_processes_finish,
            args=(self.extract_procs_queue,),
            daemon=True
        )
        self.extract_procs_watcher.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kafka_cns.close()
        self.db.close()
        return False

    def run(self):
        for record in self.kafka_cns:
            try:
                self._process_kafka_event(record)
            except Exception:
                log.error("Unepected error handling Kafka event.", exc_info=True)

    def _process_kafka_event(self, record):
        msg = json.loads(record.value.decode())
        event = msg.get('event')
        if event in self.events:
            log.debug("Processing %s event from Kafka", event)
            getattr(self, f'handle_{event}')(record, msg)
        else:
            log.debug("Unexpected %s event from Kafka", event)

    def handle_run_complete(self, record, msg: dict):
        self.handle_event(record, msg, RunData.RAW)

    def handle_migration_complete(self, record, msg: dict):
        self.handle_event(record, msg, RunData.RAW)

    def handle_correction_complete(self, record, msg: dict):
        self.handle_event(record, msg, RunData.PROC)

    def handle_event(self, record, msg: dict, run_data: RunData):
        proposal = int(msg['proposal'])
        run = int(msg['run'])

        if proposal != self.proposal:
            return

        with self.db:
            self.db.execute("""
                INSERT INTO runs (proposal, runnr, added_at) VALUES (?, ?, ?)
                ON CONFLICT (proposal, runnr) DO NOTHING
            """, (proposal, run, record.timestamp / 1000))
        log.info(f"Added p%d r%d ({run_data.value} data) to database", proposal, run)

        # Create subprocess to process the run
        extract_proc = subprocess.Popen([
            sys.executable, '-m', 'amore_mid_prototype.backend.extract_data',
            str(proposal), str(run), run_data.value
        ], cwd=self.context_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.extract_procs_queue.put((proposal, run, extract_proc))

        # Create thread to log the subprocess's output
        logger_thread = Thread(target=self.log_subprocess,
                               args=(extract_proc.stdout, run, run_data))
        logger_thread.start()

    def log_subprocess(self, stdout_pipe, run, run_data):
        with stdout_pipe:
            for line_bytes in iter(stdout_pipe.readline, b""):
                # Bytes to string, and remove trailing newline
                line = line_bytes.decode().rstrip("\n")
                log.info(f"r{run} ({run_data.value}): {line}")

def listen():
    # Set up logging to a file
    file_handler = logging.FileHandler("amore.log")
    formatter = logging.root.handlers[0].formatter
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)

    log.info(f"Running on {platform.node()} under user {getpass.getuser()}, PID {os.getpid()}")
    try:
        with EventProcessor() as processor:
            processor.run()
    except KeyboardInterrupt:
        log.error("Stopping on Ctrl + C")
    except Exception:
        log.error("Stopping on unexpected error", exc_info=True)

    # Flush all logs
    logging.shutdown()

    # Ensure that the log file is writable by everyone (so that different users
    # can start the backend).
    os.chmod("amore.log", 0o666)

if __name__ == '__main__':
    listen()
