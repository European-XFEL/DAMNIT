import getpass
import json
import logging
import os
import platform
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from socket import gethostname
from threading import Thread

from kafka import KafkaConsumer

from ..context import RunData
from ..api import find_proposal
from .db import DamnitDB, KeyValueMapping, db_path
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
READONLY_WAIT_REOPEN = 2  # Wait N seconds to reopen after read-only error

SCHEMA = """
CREATE TABLE IF NOT EXISTS proposal_databases(proposal, db_dir UNIQUE, official);
CREATE INDEX IF NOT EXISTS proposals ON proposal_databases (proposal);

-- Settings for the listener
CREATE TABLE IF NOT EXISTS settings(key PRIMARY KEY NOT NULL, value)
"""

log = logging.getLogger(__name__)

# tracking number of local threads running in parallel
# only relevant if slurm isn't available
MAX_CONCURRENT_THREADS = min(os.cpu_count() // 2, 10)
local_extraction_threads = []


@dataclass
class ProposalDBInfo:
    db_dir: Path
    official: bool


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
        self._settings = KeyValueMapping(self.conn, "settings")

        # To help prevent accidentally starting multiple listeners we default to
        # putting it in static mode.
        self.settings.setdefault("static_mode", True)
        self.settings.setdefault("allow_local_processing", False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    @property
    def settings(self):
        return self._settings

    def all_proposals(self):
        rows = self.conn.execute("SELECT proposal, db_dir, official FROM proposal_databases").fetchall()
        result = { }
        for proposal, db_dir, official in rows:
            if proposal not in result:
                result[proposal] = []
            result[proposal].append(ProposalDBInfo(Path(db_dir), bool(official)))
        return result

    def proposal_db_dirs(self, proposal: int):
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
        self._listener_dir = listener_dir
        self.db = ListenerDB(listener_dir)

        hostname = gethostname()
        if hostname.startswith('exflonc'):
            # running on the online cluster
            kafka_conf = KAFKA_CONF['onc']
        else:
            kafka_conf = KAFKA_CONF['maxwell']

        group_id = CONSUMER_ID.format(str(listener_dir).replace("/", "_"))
        client_id = CONSUMER_ID.format(f"{hostname}-{os.getpid()}")
        self.kafka_cns = KafkaConsumer(*kafka_conf['topics'],
                                       bootstrap_servers=kafka_conf['brokers'],
                                       group_id=group_id,
                                       client_id=client_id,
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
                except sqlite3.OperationalError as e:
                    if e.sqlite_errorcode == sqlite3.SQLITE_READONLY:
                        log.error("SQLite database is read only. Pause, reopen, retry.")
                        self.db.close()
                        time.sleep(READONLY_WAIT_REOPEN)
                        self.db = ListenerDB(self._listener_dir)
                        self._process_kafka_event(record)
                    else:
                        log.error("Unexpected error handling Kafka event.", exc_info=True)
                except Exception:
                    log.error("Unexpected error handling Kafka event.", exc_info=True)

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

        # If it's the first time we've seen this proposal and we're not in
        # static mode, add it to the database.
        try:
            official_path = find_proposal(proposal) / "usr/Shared/amore"
        except FileNotFoundError:
            log.warning(f"Could not find proposal directory for p{proposal}")
            official_path = None

        if official_path and db_path(official_path).is_file() and not self.db.settings["static_mode"]:
            if official_path not in self.db.proposal_db_dirs(proposal):
                self.db.add_proposal_db(proposal, official_path, True)

        sandbox_args = self.db.settings.get("sandbox_args", "")
        allow_local_processing = self.db.settings["allow_local_processing"]
        for path in self.db.proposal_db_dirs(proposal):
            try:
                # Note that we don't explicitly close this connection because
                # it's passed to the ExtractionSubmitter.
                db = DamnitDB.from_dir(path)

                # Fail fast if read-only - https://stackoverflow.com/a/44707371/434217
                db.conn.execute("pragma user_version=0;")

                db.ensure_run(proposal, run, record.timestamp / 1000)
                log.info(f"Added p%d r%d ({run_data.value} data) to database", proposal, run)

                # Set the default to the stable DAMNIT module if not already set
                damnit_python = db.metameta.setdefault("damnit_python", "/gpfs/exfel/sw/software/xfel_anaconda3/amore-mid/.pixi/envs/default/bin/python")
                submitter = ExtractionSubmitter(db.path.parent, db)
                req = ExtractionRequest(run, proposal, run_data, sandbox_args, damnit_python)
                try:
                    submitter.submit(req)
                except Exception as e:
                    if allow_local_processing:
                        log.error("Slurm job submission failed, starting process locally.", exc_info=True)
                        execute_direct(submitter, req)
                    else:
                        raise e
            except Exception:
                log.error(f"Processing p{proposal}, r{run} for {path} failed:", exc_info=True)

def listen(db_dir):
    # Set up logging to a file
    file_handler = logging.FileHandler("damnit.log")
    formatter = logging.root.handlers[0].formatter
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)

    log.info(f"Running on {platform.node()} under user {getpass.getuser()}, PID {os.getpid()}")
    try:
        with EventProcessor(db_dir) as processor:
            processor.run()
    except KeyboardInterrupt:
        log.error("Stopping on Ctrl + C")
    except Exception:
        log.error("Stopping on unexpected error", exc_info=True)

    # Flush all logs
    logging.shutdown()

if __name__ == '__main__':
    listen()
