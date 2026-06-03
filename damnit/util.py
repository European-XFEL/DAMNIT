import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from .definitions import update_brokers


def timestamp2str(timestamp):
    if timestamp is None or pd.isna(timestamp):
        return None

    dt_utc = datetime.fromtimestamp(timestamp, timezone.utc)
    dt_local = dt_utc.astimezone()
    return dt_local.strftime("%H:%M:%S %d/%m/%Y")


def isinstance_no_import(obj, mod: str, cls: str):
    """Check if isinstance(obj, mod.cls) without loading mod"""
    m = sys.modules.get(mod)
    if m is None:
        return False

    return isinstance(obj, getattr(m, cls))


class StubKafkaProducer:
    def send(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


def kafka_producer(dummy=False, **kwargs):
    """Create a KafkaProducer, or a dummy

    Pass dummy=True or set AMORE_BROKER=none to use the dummy
    """
    brokers = update_brokers()
    if dummy or brokers == ["none"]:
        return StubKafkaProducer()

    from kafka import KafkaProducer
    return KafkaProducer(
        bootstrap_servers=update_brokers(),
        value_serializer=lambda d: json.dumps(d).encode('utf-8')
    )


class StubKafkaConsumer:
    def poll(self, timeout_ms=0, **kwargs):
        time.sleep(timeout_ms / 1000)
        return {}


def kafka_consumer(*topics, dummy=False, **kwargs):
    """Create a KafkaProducer, or a dummy

    Pass dummy=True or set AMORE_BROKER=none to use the dummy
    """
    brokers = update_brokers()
    if dummy or brokers == ["none"]:
        return StubKafkaConsumer()

    from kafka import KafkaConsumer
    return KafkaConsumer(*topics, bootstrap_servers=brokers, **kwargs)
