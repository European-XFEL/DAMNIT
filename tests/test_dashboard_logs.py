from fastapi.testclient import TestClient

from damnit.dashboard import app as dashboard
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
