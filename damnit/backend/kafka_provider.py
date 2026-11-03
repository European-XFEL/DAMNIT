"""European XFEL: migration & calibration events from Kafka"""
import json
import logging
import os
from socket import gethostname

from kafka import KafkaConsumer

from ..api import find_proposal
from ..context import RunData
from .events import EventProvider, XFELEvent

log = logging.getLogger(__name__)

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

# Which kind of data each Kafka event announces
KAFKA_EVENT_RUN_DATA = {
    'daq_run_complete': RunData.RAW,
    'online_correction_complete': RunData.PROC,
    'migration_complete': RunData.RAW,
    'run_corrections_complete': RunData.PROC,
}


class KafkaEventProvider(EventProvider):
    name = "kafka"

    def __init__(self, listener_dir):
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
        self.kafka_events = kafka_conf['events']

    def close(self):
        self.kafka_cns.close()

    def events(self):
        while True:
            for record in self.kafka_cns:
                try:
                    event = self.parse_record(record)
                except Exception:
                    log.error("Unexpected error parsing Kafka event.", exc_info=True)
                    continue
                if event is not None:
                    yield event

    def parse_record(self, record) -> XFELEvent | None:
        msg = json.loads(record.value.decode())
        event = msg.get('event')
        if event not in self.kafka_events:
            log.debug("Unexpected %s event from Kafka", event)
            return None

        log.debug("Processing %s event from Kafka", event)
        return XFELEvent(
            proposal=int(msg['proposal']),
            run=int(msg['run']),
            run_data=KAFKA_EVENT_RUN_DATA[event],
            timestamp=record.timestamp / 1000,  # Kafka timestamps are in ms
            metadata=msg,
        )

    def official_db_dir(self, proposal):
        try:
            return find_proposal(proposal) / "usr/Shared/amore"
        except FileNotFoundError:
            log.warning(f"Could not find proposal directory for p{proposal}")
            return None
