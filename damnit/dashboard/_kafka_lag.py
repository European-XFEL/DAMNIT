"""Read committed lag without subscribing, polling records or joining either service group."""
import json
import sys
from contextlib import closing

def read_lag(brokers, topics, group):
    from kafka import KafkaAdminClient, KafkaConsumer, TopicPartition

    config = dict(bootstrap_servers=brokers, client_id="damnit-service-dashboard",
                  request_timeout_ms=3000, api_version_auto_timeout_ms=2000)
    with closing(KafkaAdminClient(**config)) as admin:
        metadata = {row["topic"]: row for row in admin.describe_topics(topics)}
        partitions = []
        for topic in topics:
            row = metadata.get(topic)
            if row and not row["error_code"]:
                partitions.extend(TopicPartition(topic, item["partition"])
                                  for item in row["partitions"] if not item["error_code"])
        committed = admin.list_consumer_group_offsets(group, partitions=partitions) if partitions else {}
    ends = {}
    if partitions:
        with closing(KafkaConsumer(group_id=None, enable_auto_commit=False,
                                   allow_auto_create_topics=False, **config)) as consumer:
            ends = consumer.end_offsets(partitions)

    details = []
    for topic in topics:
        row = metadata.get(topic)
        size = len(row["partitions"]) if row and not row["error_code"] else None
        offsets = []
        for part in partitions:
            if part.topic != topic or part not in committed or part not in ends:
                continue
            position = committed[part].offset
            if 0 <= position <= ends[part]:
                offsets.append(ends[part] - position)
        lag = sum(offsets) if size and len(offsets) == size else None
        details.append(dict(topic=topic, lag=lag, partitions=size, known_partitions=len(offsets)))

    if all(row["lag"] is not None for row in details):
        return dict(status="ok", lag=sum(row["lag"] for row in details), topics=details, error=None)
    return dict(status="partial", lag=None, topics=details,
                error="Offsets are unavailable for some partitions.")


if __name__ == "__main__":
    try:
        result = read_lag(**json.load(sys.stdin))
    except ImportError:
        result = dict(status="unavailable", lag=None, topics=[],
                      error="Kafka client unavailable. Set DAMNIT_DASHBOARD_KAFKA_PYTHON to a service Python with kafka-python.")
    except (OSError, RuntimeError) as exc:
        result = dict(status="unavailable", lag=None, topics=[],
                      error=f"Kafka offsets unavailable ({type(exc).__name__}).")
    print(json.dumps(result))
