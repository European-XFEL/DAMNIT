import json

import numpy as np
from kafka import TopicPartition

from damnit.api import Damnit, submit
from damnit.backend.combine import FileSubmissionProcessor
from damnit.backend.db import MsgKind
from damnit.definitions import FILE_SUBMIT_TOPIC

def test_combiner_service(mock_db, mock_kafka_broker):
    db_dir, db = mock_db

    combiner = FileSubmissionProcessor(consumer_config={
        'auto_offset_reset': 'earliest',  # Read from the 1st submission message
    })

    with mock_kafka_broker.assert_produces(FILE_SUBMIT_TOPIC):
        submit(1234, 56, {
            # We need start_time in the DB for the API to see the run
            "array": np.arange(10), "start_time": 1776335722
        }, provenance="test", damnit_dir=db_dir)

    with mock_kafka_broker.assert_produces(db.kafka_topic) as new_records:
        combiner.handle_one_message()
        combiner.producer.flush(timeout=5)

    msgs = [json.loads(r.value) for r in new_records]
    assert [m['msg_kind'] for m in msgs] == [MsgKind.run_values_updated.value]
    assert set(msgs[0]['data']['values']) == {"array", "start_time"}

    api_obj = Damnit(db_dir)
    np.testing.assert_array_equal(api_obj[56, "array"].read(), np.arange(10))

    # Overwrite the variable with a second fragment
    with mock_kafka_broker.assert_produces(FILE_SUBMIT_TOPIC):
        submit(
            1234, 56, {"array": np.arange(15)}, provenance="test", damnit_dir=db_dir
        )

    with mock_kafka_broker.assert_produces(db.kafka_topic):
        combiner.handle_one_message()
        combiner.producer.flush(timeout=5)

    np.testing.assert_array_equal(api_obj[56, "array"].read(), np.arange(15))
