import getpass
import json
import logging
import os
import platform
from pathlib import Path
from socket import gethostname

from extra_data.read_machinery import find_proposal
from kafka import KafkaConsumer

from ..context import RunData
from .db import DamnitDB
from .extraction_control import ExtractionRequest, ExtractionSubmitter

# For now, the migration & calibration events come via DESY's Kafka brokers,
# but the AMORE updates go via XFEL's test instance.
CONSUMER_ID = "xfel-da-amore-prototype-{}"
KAFKA_BROKERS = ["exflwgs06:9091"]
KAFKA_TOPICS = ["test.r2d2", "cal.offline-corrections", "test.euxfel.hed.daq", "test.euxfel.hed.cal"]
KAFKA_EVENTS = ["migration_complete", "run_corrections_complete", "daq_run_complete", "online_correction_complete"]
BACKEND_HOSTS_TO_ONLINE = ['max-exfl-display003.desy.de', 'max-exfl-display004.desy.de']
ONLINE_HOSTS ={
    'FXE': 'sa1-onc-fxe.desy.de',
    'HED': 'sa2-onc-hed.desy.de',
    'MID': 'sa2-onc-mid.desy.de',
    # 'SA1': '',
    # 'SA2': '',
    # 'SA3': '',
    'SCS': 'sa3-onc-scs.desy.de',
    'SPB': 'sa1-onc-spb.desy.de',
    'SQS': 'sa3-onc-sqs.desy.de',
    'SXP': 'sa3-onc-sxp.desy.de',
}

log = logging.getLogger(__name__)


class EventProcessor:

    def __init__(self, context_dir=Path('.')):
        if gethostname().startswith('exflonc'):
            log.warning('Running the DAMNIT listener on the online cluster is not allowed')
            exit(1)

        self.context_dir = context_dir
        self.db = DamnitDB.from_dir(context_dir)
        self.submitter = ExtractionSubmitter(context_dir, self.db)
        # Fail fast if read-only - https://stackoverflow.com/a/44707371/434217
        self.db.conn.execute("pragma user_version=0;")
        self.proposal = self.db.metameta['proposal']

        log.info(f"Will watch for events from proposal {self.proposal}")

        consumer_id = CONSUMER_ID.format(self.db.metameta['db_id'])
        self.kafka_cns = KafkaConsumer(*KAFKA_TOPICS,
                                       bootstrap_servers=KAFKA_BROKERS,
                                       group_id=consumer_id,
                                       consumer_timeout_ms=600_000,
                                       )

        # check backend host and connection to online cluster
        self.online_data_host = None
        self.run_online = self.db.metameta.get('run_online', False) is not False
        if self.run_online and gethostname() not in BACKEND_HOSTS_TO_ONLINE:
            log.warning(f"Disabled online processing, the backend must run on one of: {BACKEND_HOSTS_TO_ONLINE}")
            self.run_online = False
        if self.run_online:
            topic = Path(find_proposal(f'p{self.proposal:06}')).parts[-3]
            if (remote_host := ONLINE_HOSTS.get(topic)) is None:
                log.warning(f"Can't run online processing for topic '{topic}'")
                self.run_online = False
            else:
                self.online_data_host = remote_host
        log.info("Processing online data? %s", self.run_online)

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
        if event in KAFKA_EVENTS:
            log.debug("Processing %s event from Kafka", event)
            getattr(self, f'handle_{event}')(record, msg)
        else:
            log.debug("Unexpected %s event from Kafka", event)

    def handle_daq_run_complete(self, record, msg: dict):
        if self.run_online:
            self.handle_event(record, msg, RunData.RAW, self.online_data_host)

    def handle_online_correction_complete(self, record, msg: dict):
        if self.run_online:
            self.handle_event(record, msg, RunData.PROC, self.online_data_host)

    def handle_migration_complete(self, record, msg: dict):
        self.handle_event(record, msg, RunData.RAW)

    def handle_run_corrections_complete(self, record, msg: dict):
        self.handle_event(record, msg, RunData.PROC)

    def handle_event(self, record, msg: dict, run_data: RunData,
                     data_location: str = "localhost"):
        proposal = int(msg['proposal'])
        run = int(msg['run'])

        if proposal != self.proposal:
            return

        self.db.ensure_run(proposal, run, record.timestamp / 1000)
        log.info(f"Added p%d r%d ({run_data.value} data) to database", proposal, run)

        req = ExtractionRequest(run, proposal, run_data, data_location)
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
