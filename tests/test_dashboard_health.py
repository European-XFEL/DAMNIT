import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import kafka
import pytest
from kafka import TopicPartition

from damnit.dashboard import _kafka_lag as probe
from damnit.dashboard import health


COMBINER_GROUP = "xfel-da-damnit-combiner"
FILE_SUBMIT_TOPIC = "test.damnit.file_submissions"


@pytest.mark.parametrize(("positions", "status", "lag", "known"), [
    ([12, 5], "ok", 3, 2),
    ([-1, 5], "partial", None, 1),
    ([30, 5], "partial", None, 1),
    ([None, None], "partial", None, 0),
])
def test_group_lag(monkeypatch, positions, status, lag, known):
    partitions = [TopicPartition("notifications", partition) for partition in (0, 1)]
    admin = Mock()
    admin.describe_topics.return_value = [
        dict(topic="notifications", error_code=0,
             partitions=[dict(partition=part.partition, error_code=0) for part in partitions])
    ]
    admin.list_consumer_group_offsets.return_value = {
        part: SimpleNamespace(offset=position)
        for part, position in zip(partitions, positions) if position is not None
    }
    consumer = Mock()
    consumer.end_offsets.return_value = dict(zip(partitions, [15, 5]))
    factory = Mock(return_value=consumer)
    monkeypatch.setattr(kafka, "KafkaAdminClient", Mock(return_value=admin))
    monkeypatch.setattr(kafka, "KafkaConsumer", factory)

    result = probe.read_lag(["broker"], ["notifications"], "service-group")

    assert (result["status"], result["lag"]) == (status, lag)
    assert result["topics"] == [dict(topic="notifications", lag=lag, partitions=2, known_partitions=known)]
    admin.list_consumer_group_offsets.assert_called_once_with("service-group", partitions=partitions)
    assert factory.call_args.args == ()
    for key, value in dict(group_id=None, enable_auto_commit=False, allow_auto_create_topics=False).items():
        assert factory.call_args.kwargs[key] == value
    consumer.poll.assert_not_called()
    consumer.commit.assert_not_called()
    admin.close.assert_called_once()
    consumer.close.assert_called_once()


def test_missing_topic_is_unknown(monkeypatch):
    admin = Mock()
    admin.describe_topics.return_value = [dict(topic="missing", error_code=3, partitions=[])]
    monkeypatch.setattr(kafka, "KafkaAdminClient", Mock(return_value=admin))
    result = probe.read_lag(["broker"], ["missing"], "group")
    assert result["status"] == "partial"
    assert result["lag"] is None
    assert result["topics"][0]["partitions"] is None
    admin.list_consumer_group_offsets.assert_not_called()


def test_health_and_probe_without_kafka():
    # Use only the standard library and the dashboard package, as on an
    # installation where Kafka is available exclusively to the service.
    dashboard_parent = Path(health.__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-I", "-S", "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); from dashboard import health, _kafka_lag",
         str(dashboard_parent)], check=True,
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", probe.__file__], text=True, capture_output=True, check=True,
        input=json.dumps(dict(brokers=["127.0.0.1"], topics=["notifications"], group="listener")),
    )
    reading = json.loads(result.stdout)
    assert reading["status"] == "unavailable"
    assert reading["lag"] is None
    assert "kafka-python" in reading["error"]


def test_probe_uses_service_interpreter(monkeypatch, tmp_path):
    monkeypatch.delenv("DAMNIT_DASHBOARD_KAFKA_PYTHON", raising=False)
    python = tmp_path / "python"
    python.symlink_to(sys.executable)
    for executable in (python, tmp_path / "damnit"):
        assert health._probe_python(f"{{ path={executable} ; argv[]={executable} ; }}") == str(python)
    assert health._probe_python("") == sys.executable
    monkeypatch.setenv("DAMNIT_DASHBOARD_KAFKA_PYTHON", "/custom/bin/python")
    assert health._probe_python("") == "/custom/bin/python"


def test_brokers_and_groups_match_service_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("AMORE_BROKER", "private:1234")
    monkeypatch.delenv("DAMNIT_DASHBOARD_LISTENER_GROUP", raising=False)
    monkeypatch.setattr(health.socket, "gethostname", lambda: "exflonc123")
    run = Mock(return_value=SimpleNamespace(stdout='{"status":"ok"}'))
    monkeypatch.setattr(health.subprocess, "run", run)
    physical = tmp_path / "real"
    physical.mkdir()
    linked = tmp_path / "link"
    linked.symlink_to(physical)
    health._kafka_health("listener", {"WorkingDirectory": str(linked)})
    args = json.loads(run.call_args.kwargs["input"])
    assert args == dict(group="xfel-da-damnit-" + str(physical).replace("/", "_"), brokers=["exflwgs06:9091"],
                        topics=["test.euxfel.hed.daq", "test.euxfel.hed.cal"])
    health._kafka_health("combiner", {})
    assert json.loads(run.call_args.kwargs["input"]) == dict(
        group=COMBINER_GROUP, brokers=["private:1234"], topics=[FILE_SUBMIT_TOPIC])
    assert health._kafka_health("listener", {})["status"] == "unavailable"
    monkeypatch.setenv("DAMNIT_DASHBOARD_LISTENER_GROUP", "custom-listener")
    health._kafka_health("listener", {})
    assert json.loads(run.call_args.kwargs["input"])["group"] == "custom-listener"


def test_unit_probe(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=(
        "LoadState=loaded\nActiveState=active\nSubState=running\nResult=success\n"
        "NRestarts=3\nActiveEnterTimestamp=Thu 2026-01-01 00:00:00 UTC\nWorkingDirectory=/srv/listener\n"
    )))
    monkeypatch.setattr(health.subprocess, "run", run)
    unit, properties = health._unit_health("listener")
    assert unit == dict(status="ok", load_state="loaded", active_state="active", sub_state="running",
                        restarts=3, active_since="2026-01-01T00:00:00+00:00", result="success", error=None)
    assert properties["WorkingDirectory"] == "/srv/listener"
    assert run.call_args.args[0][-1] == "damnit-listener.service"
    assert run.call_args.kwargs["env"]["TZ"] == "UTC"
    run.return_value = SimpleNamespace(returncode=4, stdout="LoadState=not-found\nActiveState=inactive\n")
    assert health._unit_health("listener")[0]["load_state"] == "not-found"
    run.side_effect = subprocess.TimeoutExpired("probe", 4)
    assert health._unit_health("listener")[0]["status"] == "unavailable"
    assert health._kafka_health("combiner", {})["status"] == "unavailable"


@pytest.mark.parametrize(("key", "message", "matches"), [
    ("last_input", "Added p1234 r17 (raw data) to database", True),
    ("last_input", "Processing p1234, r17 for /outside failed:", False),
    ("last_success", "Launched Slurm (maxwell) job 42 to run context file", True),
    ("last_success", "Local extraction of p1234, r17 failed:", False),
    ("last_success", "Slurm job submission failed, starting process locally.", False),
])
def test_listener_activity_markers(key, message, matches):
    line = "2026-01-01 00:00:01 INFO damnit.backend.listener " + message
    assert bool(re.search(health.ACTIVITY_PATTERNS["listener"][key], line)) is matches


def test_health_snapshot_isolates_sources_and_coalesces_refreshes(monkeypatch):
    def journal(args, *, allow_no_matches, timeout_seconds):
        assert allow_no_matches and timeout_seconds <= 5
        assert any(arg.startswith("--since=@") for arg in args)
        assert any(arg.startswith("--until=@") for arg in args)
        if "--user-unit=damnit-listener.service" in args:
            raise health.JournalError("offline")
        if any("DB change" in arg for arg in args):
            return iter([])
        return iter([{"__REALTIME_TIMESTAMP": "1767225600123456"}])

    units = Mock(return_value=(dict(status="ok", active_state="active"), {"WorkingDirectory": "/srv/listener"}))
    monkeypatch.setattr(health, "_unit_health", units)
    monkeypatch.setattr(health, "_kafka_health", lambda *_: dict(status="partial", lag=None))
    monkeypatch.setattr(health, "_journal_records", journal)
    monkeypatch.setattr(health, "_snapshot", None)
    monkeypatch.setattr(health, "_expires", 0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: health.get_health(), range(4)))
    assert units.call_count == len(health.SERVICES)
    assert len({r["sampled_at"] for r in results}) == 1
    services = results[0]["services"]
    assert services["listener"]["activity"]["status"] == "unavailable"
    assert services["listener"]["unit"]["active_state"] == "active"
    assert services["combiner"]["activity"]["last_input"] == "2026-01-01T00:00:00.123456+00:00"
    assert services["combiner"]["activity"]["last_success"] is None
    assert services["combiner"]["kafka"]["lag"] is None
    sampled = datetime.fromisoformat(results[0]["sampled_at"])
    since = datetime.fromisoformat(services["combiner"]["activity"]["since"])
    assert sampled - since == timedelta(hours=24)
    monkeypatch.setattr(health, "_expires", 0)
    health.get_health()
    assert units.call_count == 2 * len(health.SERVICES)


def test_probe_reads_broker_without_advancing_group(nafka_broker, monkeypatch):
    monkeypatch.setenv("DAMNIT_DASHBOARD_KAFKA_PYTHON", sys.executable)
    monkeypatch.setenv("AMORE_BROKER", nafka_broker.address)
    for message in ("one", "two", "three"):
        nafka_broker.add_record(FILE_SUBMIT_TOPIC, message)
    nafka_broker.store.commit_consumer_offset(COMBINER_GROUP, FILE_SUBMIT_TOPIC, 1)

    result = health._kafka_health("combiner", {})

    assert result["status"] == "ok", result
    assert result["lag"] == 2
    assert nafka_broker.consumer_offset(COMBINER_GROUP, FILE_SUBMIT_TOPIC) == 1
    assert nafka_broker.next_offset(FILE_SUBMIT_TOPIC) == 3
