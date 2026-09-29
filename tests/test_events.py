import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from damnit.backend.events import DESYEvent, Event, XFELEvent, make_event_provider
from damnit.backend.kafka_provider import KafkaEventProvider
from damnit.context import RunData


def test_event_models():
    with pytest.raises(TypeError):
        Event()  # Abstract

    xfel = XFELEvent(1234, 5, RunData.PROC)
    assert (xfel.damnit_proposal, xfel.damnit_run) == (1234, 5)
    assert str(xfel) == "p1234 r5 (proc data)"

    desy = DESYEvent(beamtime_id=11012345, scan_number=7)
    assert (desy.damnit_proposal, desy.damnit_run) == (11012345, 7)
    assert desy.run_data == RunData.RAW
    assert str(desy) == "beamtime 11012345 scan 7 (raw data)"


def kafka_record(msg, timestamp_ms=1_700_000_000_000):
    return SimpleNamespace(value=json.dumps(msg).encode(), timestamp=timestamp_ms)


def test_kafka_parse_record(tmp_path):
    with patch('damnit.backend.kafka_provider.KafkaConsumer'):
        provider = KafkaEventProvider(tmp_path)

    provider.kafka_events = ["migration_complete", "run_corrections_complete"]
    event = provider.parse_record(kafka_record(
        {"event": "migration_complete", "proposal": "1234", "run": "5"}))
    assert event == XFELEvent(1234, 5, RunData.RAW, 1_700_000_000)

    event = provider.parse_record(kafka_record(
        {"event": "run_corrections_complete", "proposal": 1234, "run": 5}))
    assert event.run_data == RunData.PROC

    assert provider.parse_record(kafka_record({"event": "something_else"})) is None


def test_make_event_provider(tmp_path):
    with patch('damnit.backend.kafka_provider.KafkaConsumer'):
        assert isinstance(make_event_provider({}, tmp_path), KafkaEventProvider)
        assert isinstance(make_event_provider({"event_provider": "kafka"}, tmp_path),
                          KafkaEventProvider)

    with pytest.raises(ValueError):
        make_event_provider({"event_provider": "nope"}, tmp_path)


class FakeScan:
    def __init__(self, key, states, info, session="test_session", number=1):
        from blissdata.redis_engine.scan import ScanState
        self.key = key
        self.name = key
        self.session = session
        self.number = number
        self.proposal = None
        self._states = [ScanState[s] for s in states]
        self.state = self._states.pop(0)
        self.info = info

    def update(self, block=True, timeout=0):
        if self._states:
            self.state = self._states.pop(0)
            return True
        return False


class FakeDataStore:
    def __init__(self, scans):
        self.scans = {s.key: s for s in scans}
        self.new_keys = [s.key for s in scans]

    def get_last_scan_timetag(self):
        return "0-0"

    def get_next_scan(self, since=None, timeout=0):
        from damnit.backend.blissdata_provider import NoScanAvailable
        if not self.new_keys:
            raise NoScanAvailable
        return since, self.new_keys.pop(0)

    def load_scan(self, key):
        return self.scans[key]


def take(it, n, max_iter=100):
    """Collect n events, iterating the generator at most max_iter times"""
    out = []
    for _, event in zip(range(max_iter), it):
        out.append(event)
        if len(out) == n:
            break
    return out


def test_blissdata_provider():
    bp = pytest.importorskip("damnit.backend.blissdata_provider")

    info = {"beamtime_id": "11012345", "scan_nb": 7, "end_reason": "SUCCESS",
            "end_time": "2026-09-14T17:44:34+02:00", "filename": "/data/tst_00007.nxs"}
    scans = [
        # Only emitted once it reaches CLOSED
        FakeScan("a", ["CREATED", "STARTED", "STOPPED", "CLOSED"], info),
        # Wrong session -> ignored
        FakeScan("b", ["CLOSED"], info | {"scan_nb": 8}, session="other"),
        # No usable beamtime ID -> skipped
        FakeScan("c", ["CLOSED"], {"scan_nb": 9}),
        # Aborted scans are processed too by default
        FakeScan("d", ["CLOSED"], info | {"scan_nb": 10, "end_reason": "USER_ABORT"}),
        # Still running -> not emitted yet
        FakeScan("e", ["CREATED", "STARTED"], info | {"scan_nb": 11}),
    ]
    provider = bp.BlissdataEventProvider(
        session="test_session", poll_interval=0, data_store=FakeDataStore(scans))

    events = take(provider.events(), 2)
    assert all(isinstance(e, DESYEvent) for e in events)
    assert [(e.beamtime_id, e.scan_number) for e in events] == [(11012345, 7), (11012345, 10)]

    # The running scan stays pending until it reaches CLOSED
    provider._add_scan("e")
    assert list(provider._check_pending()) == []
    assert list(provider._pending) == ["e"]
    scan7 = events[0]
    assert scan7.filename == "/data/tst_00007.nxs"
    assert scan7.session == "test_session"
    assert scan7.end_reason == "SUCCESS"
    assert scan7.timestamp == pytest.approx(1789400674)


def test_blissdata_only_successful():
    bp = pytest.importorskip("damnit.backend.blissdata_provider")

    scans = [
        FakeScan("a", ["CLOSED"], {"beamtime_id": 1, "scan_nb": 1, "end_reason": "USER_ABORT"}),
        FakeScan("b", ["CLOSED"], {"beamtime_id": 1, "scan_nb": 2, "end_reason": "SUCCESS"}),
    ]
    provider = bp.BlissdataEventProvider(
        only_successful=True, poll_interval=0, data_store=FakeDataStore(scans))
    assert [e.scan_number for e in take(provider.events(), 1)] == [2]


def test_blissdata_from_settings(monkeypatch, tmp_path):
    bp = pytest.importorskip("damnit.backend.blissdata_provider")
    from blissdata.redis_engine.scan import ScanState

    assert isinstance(make_event_provider({"event_provider": "blissdata"}, tmp_path),
                      bp.BlissdataEventProvider)

    monkeypatch.delenv("BLISSDATA_URL", raising=False)
    provider = bp.BlissdataEventProvider.from_settings({})
    assert provider.redis_url == "redis://localhost:6380"
    assert provider.session is None
    assert provider.trigger_state == ScanState.CLOSED

    provider = bp.BlissdataEventProvider.from_settings({
        "blissdata_url": "redis://somehost:6380", "blissdata_session": "s1",
        "blissdata_trigger_state": "stopped",
    })
    assert provider.redis_url == "redis://somehost:6380"
    assert provider.session == "s1"
    assert provider.trigger_state == ScanState.STOPPED
