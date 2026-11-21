import pytest
from fastapi.testclient import TestClient

from damnit.dashboard import app as dashboard
from damnit.dashboard import journal
from damnit.dashboard.journal import JournalError


def test_raw_logs_api(monkeypatch):
    calls = []
    rows = [{"timestamp": "2024-01-01T00:00:00+00:00", "message": "error\n  details  "}]

    def read(service, query, limit):
        calls.append((service, query, limit))
        return rows if not query else []

    monkeypatch.setattr(dashboard, "read_log_entries", read)
    with TestClient(dashboard.app) as client:
        latest = client.get("/api/logs", params={"service": "combiner"})
        assert latest.headers["cache-control"] == "no-store"
        assert latest.json() == {"entries": rows}
        response = client.get("/api/logs", params={"service": "listener", "q": "-.*", "limit": 500})
        assert response.status_code == 200
        assert response.json() == {"entries": []}
        for params in [{}, {"service": "*"}, {"service": "combiner", "q": "a" * 257},
                       {"service": "listener", "limit": 0}, {"service": "listener", "limit": 501}]:
            assert client.get("/api/logs", params=params).status_code == 422
    assert calls == [("combiner", "", 100), ("listener", "-.*", 500)]


def test_raw_logs_unavailable(monkeypatch):
    def unavailable(service, query, limit):
        raise JournalError("Journal unavailable")

    monkeypatch.setattr(dashboard, "read_log_entries", unavailable)
    with TestClient(dashboard.app) as client:
        response = client.get("/api/logs", params={"service": "combiner"})
        assert response.status_code == 503
        assert response.json()["detail"] == "Journal unavailable"


@pytest.mark.parametrize(("match", "older_record", "newer_record", "expected"), [
    (True, False, False, "available"),
    (False, True, False, "available"),
    (False, False, True, "outside_retention"),
    (False, False, False, "unknown"),
])
def test_scoped_journal_search(monkeypatch, match, older_record, newer_record, expected):
    record = {"__REALTIME_TIMESTAMP": "1704067200000000", "MESSAGE": "DB rolled back"}

    def read(options, *, allow_no_matches=False):
        if "--user-unit=damnit-combiner.service" in options:
            assert {"--since=@1704067200.000000", "--until=@1704067259.999999",
                    "--grep=rolled", "--case-sensitive=no"} <= set(options)
            assert allow_no_matches
            present = match
        elif "--until=@1704067259.999999" in options:
            present = older_record
        else:
            present = newer_record
        if present:
            yield record

    monkeypatch.setattr(journal, "_journal_records", read)
    params = dict(service="combiner", q="rolled", start="2024-01-01T01:00:00+01:00",
                  end="2024-01-01T01:01:00+01:00")
    with TestClient(dashboard.app) as client:
        response = client.get("/api/logs", params=params)
        assert response.status_code == 200
        assert response.json()["availability"] == expected
        assert [row["message"] for row in response.json()["entries"]] == ([record["MESSAGE"]] if match else [])


@pytest.mark.parametrize("start", ["2024-01-01T00:00:00", "2024-01-02T00:00:00Z"])
def test_invalid_journal_bounds(start):
    with TestClient(dashboard.app) as client:
        assert client.get("/api/logs", params={
            "service": "listener", "start": start, "end": "2024-01-02T00:00:00Z",
        }).status_code == 422
