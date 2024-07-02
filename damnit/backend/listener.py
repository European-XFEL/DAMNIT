import getpass
import json
import logging
import os
import platform
from pathlib import Path
from socket import gethostname

from kafka import KafkaConsumer

from ..context import RunData
from .db import DamnitDB
from .extraction_control import ExtractionRequest, ExtractionSubmitter

# For now, the migration & calibration events come via DESY's Kafka brokers,
# but the AMORE updates go via XFEL's test instance.
CONSUMER_ID = 'xfel-da-amore-prototype-{}'
KAFKA_CONF = {
    'maxwell': {
        'brokers': ['exflwgs06:9091'],
        'topics': ["test.r2d2", "cal.offline-corrections"],
        'events': ["migration_complete", "run_corrections_complete"],
    },
    'onc': {
        'brokers': ['exflwgs06:9091'],
        'topics': ['test.euxfel.hed.daq', 'test.euxfel.hed.cal'],
        'events': ['daq_run_complete', 'online_correction_complete'],
    }
}

log = logging.getLogger(__name__)


class EventProcessor:

    def __init__(self, context_dir=Path('.')):
        self.context_dir = context_dir
        self.db = DamnitDB.from_dir(context_dir)
        self.submitter = ExtractionSubmitter(context_dir, self.db)
        # Fail fast if read-only - https://stackoverflow.com/a/44707371/434217
        self.db.conn.execute("pragma user_version=0;")
        self.proposal = self.db.metameta['proposal']
        log.info(f"Will watch for events from proposal {self.proposal}")

        if gethostname().startswith('exflonc'):
            # running on the online cluster
            kafka_conf = KAFKA_CONF['onc']
        else:
            kafka_conf = KAFKA_CONF['maxwell']

        consumer_id = CONSUMER_ID.format(self.db.metameta['db_id'])
        self.kafka_cns = KafkaConsumer(*kafka_conf['topics'],
                                       bootstrap_servers=kafka_conf['brokers'],
                                       group_id=consumer_id,
                                       consumer_timeout_ms=600_000,
                                       )
        self.events = kafka_conf['events']


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kafka_cns.close()
        self.db.close()
        return False

    def run(self):
        while True:
            for record in self.kafka_cns:
                try:
                    self._process_kafka_event(record)
                except Exception:
                    log.error("Unepected error handling Kafka event.", exc_info=True)

            # After 10 minutes with no messages, check if the listener should stop
            if self.db.metameta.get('no_listener', 0):
                log.info("Found no_listener flag in database, shutting down.")
                return

    def _process_kafka_event(self, record):
        msg = json.loads(record.value.decode())
        event = msg.get('event')
        if event in self.events:
            log.debug("Processing %s event from Kafka", event)
            getattr(self, f'handle_{event}')(record, msg)
        else:
            log.debug("Unexpected %s event from Kafka", event)

    def handle_daq_run_complete(self, record, msg: dict):
        self.handle_event(record, msg, RunData.RAW)

    def handle_online_correction_complete(self, record, msg: dict):
        self.handle_event(record, msg, RunData.PROC)

    def handle_migration_complete(self, record, msg: dict):
        self.handle_event(record, msg, RunData.RAW)

    def handle_run_corrections_complete(self, record, msg: dict):
        self.handle_event(record, msg, RunData.PROC)

    def handle_event(self, record, msg: dict, run_data: RunData):
        proposal = int(msg['proposal'])
        run = int(msg['run'])

        if proposal != self.proposal:
            return

        self.db.ensure_run(proposal, run, record.timestamp / 1000)
        log.info(f"Added p%d r%d ({run_data.value} data) to database", proposal, run)

        req = ExtractionRequest(run, proposal, run_data)
        self.submitter.submit(req)


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
    if os.stat("amore.log").st_uid == os.getuid():
        os.chmod("amore.log", 0o666)

if __name__ == '__main__':
    listen()
