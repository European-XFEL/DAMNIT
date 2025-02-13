import getpass
import json
import logging
import os
import platform
import sqlite3
import traceback
from pathlib import Path
from socket import gethostname
from threading import Thread

from kafka import KafkaConsumer

from ..context import RunData
from ..api import find_proposal
from .db import DamnitDB, KeyValueMapping
from .extraction_control import ExtractionRequest, ExtractionSubmitter

# For now, the migration & calibration events come via DESY's Kafka brokers,
# but the DAMNIT updates go via XFEL's test instance.
CONSUMER_ID = 'xfel-da-damnit-{}'
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

SCHEMA = """
CREATE TABLE IF NOT EXISTS proposal_databases(proposal, db_dir UNIQUE, official);

-- Settings for the listener
CREATE TABLE IF NOT EXISTS settings(key PRIMARY KEY NOT NULL, value)
"""

log = logging.getLogger(__name__)

# tracking number of local threads running in parallel
# only relevant if slurm isn't available
MAX_CONCURRENT_THREADS = min(os.cpu_count() // 2, 10)
local_extraction_threads = []


def execute_direct(submitter, request):
    for th in local_extraction_threads.copy():
        if not th.is_alive():
            local_extraction_threads.pop(local_extraction_threads.index(th))

    if len(local_extraction_threads) >= MAX_CONCURRENT_THREADS:
        log.warning(f'Too many events processing ({MAX_CONCURRENT_THREADS}), '
                    f'skip event (p{request.proposal}, r{request.run}, {request.run_data.value})')
        return

    extr = Thread(target=submitter.execute_direct, args=(request, ))
    local_extraction_threads.append(extr)
    extr.start()

class ListenerDB:
    def __init__(self, db_dir):
        self.conn = sqlite3.connect(db_dir.absolute() / "listener.sqlite")
        self.conn.executescript(SCHEMA)
        self._settings = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    @property
    def settings(self):
        if self._settings is None:
            self._settings = KeyValueMapping(self.conn, "settings")

        return self._settings

    def proposal_db_dirs(self, proposal):
        official_path = find_proposal(proposal) / "usr/Shared/amore"
        if (official_path / "runs.sqlite").is_file():
            with self.conn:
                self.conn.execute("""
                    INSERT INTO proposal_databases (proposal, db_dir, official) VALUES (?, ?, ?)
                    ON CONFLICT (db_dir) DO NOTHING
                """, (proposal, str(official_path), True))

        rows = self.conn.execute("SELECT db_dir from proposal_databases WHERE proposal=?",
                                 (proposal,)).fetchall()
        return [Path(row[0]) for row in rows]

    def add_proposal_db(self, proposal: int, db_dir, official: bool):
        with self.conn:
            self.conn.execute("""
                INSERT INTO proposal_databases (proposal, db_dir, official) VALUES (?, ?, ?)
            """, (proposal, str(db_dir), official))

    def remove_proposal_db(self, db_dir):
        with self.conn:
            self.conn.execute("DELETE FROM proposal_databases WHERE db_dir=?", (str(db_dir),))

class EventProcessor:
    def __init__(self, listener_dir: Path):
        self.submitters = dict()

        self.db = ListenerDB(listener_dir)

        # To help prevent accidentally starting multiple listeners we default to
        # disabling it entirely.
        self.db.settings.setdefault("proposal", -1)

        hostname = gethostname()
        if hostname.startswith('exflonc'):
            # running on the online cluster
            kafka_conf = KAFKA_CONF['onc']
        else:
            kafka_conf = KAFKA_CONF['maxwell']

        consumer_id = CONSUMER_ID.format(f"{hostname}-{os.getpid()}")
        self.kafka_cns = KafkaConsumer(*kafka_conf['topics'],
                                       bootstrap_servers=kafka_conf['brokers'],
                                       group_id=consumer_id,
                                       consumer_timeout_ms=600_000,
                                       )
        self.events = kafka_conf['events']
        log.info("Started listener")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kafka_cns.close()
        return False

    def run(self):
        while True:
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

        # If a specific proposal is requested and this event isn't for it, do nothing
        if self.db.settings["proposal"] != "all" and self.db.settings["proposal"] != proposal:
            return

        for path in self.db.proposal_db_dirs(proposal):
            try:
                db = DamnitDB(path / "runs.sqlite")

                # Fail fast if read-only - https://stackoverflow.com/a/44707371/434217
                db.conn.execute("pragma user_version=0;")

                db.ensure_run(proposal, run, record.timestamp / 1000)
                log.info(f"Added p%d r%d ({run_data.value} data) to database", proposal, run)

                if path not in self.submitters:
                    self.submitters[path] = ExtractionSubmitter(db.path.parent, db)
                submitter = self.submitters[path]

                req = ExtractionRequest(run, proposal, run_data)
                try:
                    submitter.submit(req)
                except FileNotFoundError:
                    log.warning('Slurm not available, starting process locally.')
                    execute_direct(submitter, req)
                except Exception:
                    log.error("Slurm job submission failed, starting process locally.", exc_info=True)
                    execute_direct(submitter, req)

                db.close()
            except:
                log.error(f"Processing p{proposal}, r{run} for {path} failed:")
                log.error(traceback.format_exc())

def listen(db_path):
    # Set up logging to a file
    file_handler = logging.FileHandler("damnit.log")
    formatter = logging.root.handlers[0].formatter
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)

    log.info(f"Running on {platform.node()} under user {getpass.getuser()}, PID {os.getpid()}")
    try:
        with EventProcessor(db_path) as processor:
            processor.run()
    except KeyboardInterrupt:
        log.error("Stopping on Ctrl + C")
    except Exception:
        log.error("Stopping on unexpected error", exc_info=True)

    # Flush all logs
    logging.shutdown()

if __name__ == '__main__':
    listen()
