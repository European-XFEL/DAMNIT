import json
import logging
import subprocess
import sys
from pathlib import Path

from kafka import KafkaConsumer, KafkaProducer

from ..definitions import UPDATE_BROKERS, UPDATE_TOPIC
from .db import open_db, get_meta
from .extract_data import add_to_db, load_reduced_data

# For now, the migration & calibration events come via DESY's Kafka brokers,
# but the AMORE updates go via XFEL's test instance.
BROKERS_IN = [f'it-kafka-broker{i:02}.desy.de' for i in range(1, 4)]
MIGRATION_TOPIC = 'xfel-test-r2d2'
CALIBRATION_TOPIC = "xfel-test-offline-cal"
CONSUMER_ID = 'xfel-da-amore-mid-prototype'

log = logging.getLogger(__name__)

class EventProcessor:
    def __init__(self, context_dir=Path('.')):
        self.context_dir = context_dir
        self.db = open_db(context_dir / 'runs.sqlite')
        # Fail fast if read-only - https://stackoverflow.com/a/44707371/434217
        self.db.execute("pragma user_version=0;")
        self.proposal = get_meta(self.db, 'proposal')
        log.info(f"Will watch for events from proposal {self.proposal}")

        self.kafka_cns = KafkaConsumer(CALIBRATION_TOPIC, bootstrap_servers=BROKERS_IN, group_id=CONSUMER_ID)

        self.kafka_prd = KafkaProducer(
            bootstrap_servers=UPDATE_BROKERS,
            value_serializer=lambda d: json.dumps(d).encode('utf-8'),
        )
        self.update_topic = UPDATE_TOPIC.format(get_meta(self.db, 'db_id'))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kafka_cns.close()
        self.db.close()
        # Closing the producer can block (if messages are still being sent)
        # so do this after the other steps.
        self.kafka_prd.close(timeout=2)  # timeout in seconds
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
        if event in self.EXPECTED_EVENTS:
            log.debug("Processing %s event from Kafka", event)
            getattr(self, f'handle_{event}')(record, msg)
        else:
            log.debug("Unexpected %s event from Kafka", event)

    EXPECTED_EVENTS = {'correction_complete'}

    def handle_migration_complete(self, record, msg: dict):
        proposal = int(msg['proposal'])
        run = int(msg['run'])
        run_dir = msg['path']

        if proposal != self.proposal:
            return

        with self.db:
            self.db.execute("""
                INSERT INTO runs (proposal, runnr, added_at) VALUES (?, ?, ?)
                ON CONFLICT (proposal, runnr) DO NOTHING
            """, (proposal, run, record.timestamp / 1000))
        log.info("Added p%d r%d to database", proposal, run)

        out_path = self.context_dir / 'extracted_data' / f'p{proposal}_r{run}.h5'
        out_path.parent.mkdir(parents=True, exist_ok=True)

        extract_res = subprocess.run([
            sys.executable, '-m', 'amore_mid_prototype.backend.extract_data',
            str(proposal), str(run), out_path
        ])
        if extract_res.returncode != 0:
            log.error("Data extraction failed; exit code was %d", extract_res.returncode)
        else:
            reduced_data = load_reduced_data(out_path)
            log.info("Reduced data has %d fields", len(reduced_data))
            add_to_db(reduced_data, self.db, proposal, run)

            reduced_data['Proposal'] = proposal
            reduced_data['Run'] = run
            self.kafka_prd.send(self.update_topic, reduced_data)
            log.info("Sent Kafka update")

    def handle_correction_complete(self, record, msg: dict):
        proposal = int(msg['proposal'])
        run = int(msg['run'])

        if msg.get('detector') != 'agipd' or proposal != self.proposal:
            return

        with self.db:
            run_count = self.db.execute("""
                SELECT COUNT(runnr)
                FROM runs
                WHERE proposal = ? AND runnr = ?
            """, (proposal, run)).fetchone()[0]

            if run_count > 0:
                log.info(f"Already processed run {run}, skipping re-processing")
                return

        with self.db:
            self.db.execute("""
                INSERT INTO runs (proposal, runnr, added_at) VALUES (?, ?, ?)
                ON CONFLICT (proposal, runnr) DO NOTHING
            """, (proposal, run, record.timestamp / 1000))
        log.info("Added p%d r%d to database", proposal, run)

        out_path = self.context_dir / 'extracted_data' / f'p{proposal}_r{run}.h5'
        out_path.parent.mkdir(parents=True, exist_ok=True)

        extract_res = subprocess.run([
            sys.executable, '-m', 'amore_mid_prototype.backend.extract_data', str(proposal), str(run), out_path
        ])
        if extract_res.returncode != 0:
            log.error("Data extraction failed; exit code was %d", extract_res.returncode)
        else:
            reduced_data = load_reduced_data(out_path)
            log.info("Reduced data has %d fields", len(reduced_data))
            add_to_db(reduced_data, self.db, proposal, run)

            reduced_data['Proposal'] = proposal
            reduced_data['Run'] = run
            self.kafka_prd.send(self.update_topic, reduced_data)
            log.info("Sent Kafka update")


def listen_migrated():
    try:
        with EventProcessor() as processor:
            processor.run()
    except KeyboardInterrupt:
        print("Stopping on Ctrl-C")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    listen_migrated()
