import json
import logging
import socket
import subprocess
import sys
from pathlib import Path

from kafka import KafkaConsumer
import zmq

from .db import open_db
from .extract_data import add_to_db, load_reduced_data

BROKERS = [f'it-kafka-broker{i:02}.desy.de' for i in range(1, 4)]
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

        self.kafka_cns = KafkaConsumer(CALIBRATION_TOPIC, bootstrap_servers=BROKERS, group_id=CONSUMER_ID)

        self.zmq_sock: zmq.Socket = zmq.Context.instance().socket(zmq.PUB)
        zmq_port = self.zmq_sock.bind_to_random_port('tcp://*')
        self.zmq_addr = f"tcp://{socket.gethostname()}:{zmq_port}"
        log.info("ZMQ address: %s", self.zmq_addr)

        self._zmq_addr_file = context_dir / '.zmq_extraction_events'
        with self._zmq_addr_file.open('w') as f:
            f.write(self.zmq_addr)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._zmq_addr_file.unlink()
        except FileNotFoundError:
            pass
        self.zmq_sock.close()
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
        if event in self.EXPECTED_EVENTS:
            log.debug("Processing %s event from Kafka", event)
            getattr(self, f'handle_{event}')(record, msg)
        else:
            log.debug("Unexpected %s event from Kafka", event)

    EXPECTED_EVENTS = {'correction_complete'}

    def handle_migration_complete(self, record, msg: dict):
        if msg.get('instrument') != 'MID':
            return

        proposal = int(msg['proposal'])
        run = int(msg['run'])
        run_dir = msg['path']

        with self.db:
            self.db.execute("""
                INSERT INTO runs (proposal, runnr, added_at) VALUES (?, ?, ?)
                ON CONFLICT (proposal, runnr) DO NOTHING
            """, (proposal, run, record.timestamp / 1000))
        log.info("Added p%d r%d to database", proposal, run)

        out_path = self.context_dir / 'extracted_data' / f'p{proposal}_r{run}.h5'
        out_path.parent.mkdir(parents=True, exist_ok=True)

        extract_res = subprocess.run([
            sys.executable, '-m', 'amore_mid_prototype.backend.extract_data', run_dir, out_path
        ])
        if extract_res.returncode != 0:
            log.error("Data extraction failed; exit code was %d", extract_res.returncode)
        else:
            reduced_data = load_reduced_data(out_path)
            log.info("Reduced data has %d fields", len(reduced_data))
            add_to_db(reduced_data, self.db, proposal, run)

            reduced_data['Proposal'] = proposal
            reduced_data['Run'] = run
            self.zmq_sock.send_json(reduced_data)
            log.info("Sent ZMQ message")

    def handle_correction_complete(self, record, msg: dict):
        proposal = int(msg['proposal'])
        run = int(msg['run'])

        if msg.get('detector') != 'agipd' and run == 3217:
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
            sys.executable, '-m', 'amore_mid_prototype.backend.extract_data', run_dir, out_path
        ])
        if extract_res.returncode != 0:
            log.error("Data extraction failed; exit code was %d", extract_res.returncode)
        else:
            reduced_data = load_reduced_data(out_path)
            log.info("Reduced data has %d fields", len(reduced_data))
            add_to_db(reduced_data, self.db, proposal, run)

            reduced_data['Proposal'] = proposal
            reduced_data['Run'] = run
            self.zmq_sock.send_json(reduced_data)
            log.info("Sent ZMQ message")


def listen_migrated():
    try:
        with EventProcessor() as processor:
            processor.run()
    except KeyboardInterrupt:
        print("Stopping on Ctrl-C")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    listen_migrated()
