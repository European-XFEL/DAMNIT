import getpass
import json
import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from socket import gethostname
from threading import Thread

import psycopg
from psycopg.rows import dict_row
from kafka import KafkaConsumer

from ..context import RunData
from ..definitions import ADMIN_TOPIC, DEFAULT_DAMNIT_PYTHON, UPDATE_BROKERS
from ..api import find_proposal
from .db import DamnitDB, KeyValueMapping, MsgKind, db_path
from .extraction_control import ExtractionRequest, ExtractionSubmitter
from .pg_admin import create_proposal_db
from .service import notify_ready

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
CREATE TABLE IF NOT EXISTS proposal_databases(
    proposal INTEGER,
    db_dir TEXT UNIQUE,
    official BOOLEAN
);
CREATE INDEX IF NOT EXISTS proposals ON proposal_databases (proposal);

-- Settings for the listener
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY NOT NULL, value TEXT);
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
    
    def _run():
        try:
            submitter.execute_direct(request)
        except Exception:
            log.error(f"Local extraction of p{request.proposal}, r{request.run} failed:", exc_info=True)

    extr = Thread(target=_run)
    local_extraction_threads.append(extr)
    extr.start()


class ListenerDB:
    def __init__(self, admin_dsn):
        self.conn = psycopg.connect(admin_dsn, row_factory=dict_row)
        with self.conn.transaction():
            self.conn.execute(SCHEMA)
        self._settings = KeyValueMapping(self.conn, "settings")

        # To help prevent accidentally starting multiple listeners we default to
        # putting it in static mode.
        self.settings.setdefault("static_mode", True)
        self.settings.setdefault("allow_local_processing", False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.conn.close()

    @property
    def settings(self):
        return self._settings

    def all_proposals(self):
        rows = self.conn.execute(
            "SELECT proposal, db_dir, official FROM proposal_databases"
        ).fetchall()
        result = { }
        for row in rows:
            proposal = row["proposal"]
            if proposal not in result:
                result[proposal] = []
            result[proposal].append(
                ProposalDBInfo(Path(row["db_dir"]), bool(row["official"]))
            )
        return result

    def proposal_db_dirs(self, proposal: int):
        rows = self.conn.execute(
            "SELECT db_dir from proposal_databases WHERE proposal=%s", (proposal,)
        ).fetchall()
        return [Path(row["db_dir"]) for row in rows]

    def add_proposal_db(self, proposal: int, db_dir, official: bool):
        with self.conn.transaction():
            self.conn.execute("""
                INSERT INTO proposal_databases (proposal, db_dir, official) VALUES (%s, %s, %s)
            """, (proposal, str(db_dir), official))

    def remove_proposal_db(self, db_dir):
        with self.conn.transaction():
            self.conn.execute(
                "DELETE FROM proposal_databases WHERE db_dir=%s", (str(db_dir),)
            )

class EventProcessor:
    def __init__(self, listener_dir: Path):
        self._listener_dir = listener_dir
        self.admin_dsn = os.environ["DAMNIT_PG_ADMIN_DSN"]
        self.db = ListenerDB(self.admin_dsn)

        hostname = gethostname()
        if hostname.startswith('exflonc'):
            # running on the online cluster
            kafka_conf = KAFKA_CONF['onc']
        else:
            kafka_conf = KAFKA_CONF['maxwell']

        group_id = CONSUMER_ID.format(str(listener_dir).replace("/", "_"))
        client_id = CONSUMER_ID.format(f"{hostname}-{os.getpid()}")
        self.kafka_cns = KafkaConsumer(*kafka_conf['topics'], ADMIN_TOPIC,
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
                except Exception:
                    log.error("Unexpected error handling Kafka event.", exc_info=True)

    def _process_kafka_event(self, record):
        msg = json.loads(record.value.decode())
        if record.topic == ADMIN_TOPIC:
            self._process_admin_message(msg)
            return
        event = msg.get('event')
        if event in self.events:
            log.debug("Processing %s event from Kafka", event)
            getattr(self, f'handle_{event}')(record, msg)
        else:
            log.debug("Unexpected %s event from Kafka", event)

    def _process_admin_message(self, msg: dict):
        kind = msg.get('msg_kind')
        data = msg.get('data', {})
        if kind == MsgKind.create_proposal_db.value:
            proposal = int(data['proposal'])
            proposal_dir = Path(data['proposal_dir'])
            requester = data.get('requester_username', '?')
            log.info(
                "Provisioning DAMNIT database for p%d (requested by %s)",
                proposal, requester,
            )
            create_proposal_db(self.admin_dsn, proposal, proposal_dir)
        else:
            log.debug("Ignoring admin message kind %r", kind)

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
                damnit_python = db.metameta.setdefault("damnit_python", DEFAULT_DAMNIT_PYTHON)
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
    notify_ready()
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
