# Service dashboard

Run the dashboard on the service host, under the account that owns both
systemd user units.

```bash
pixi run -e dashboard uvicorn damnit.dashboard.app:app --host 127.0.0.1 --port 8000
```

For remote access, Kafka probe configuration, incident investigation, and
cache upgrades, see the [operator guide](../../docs/internals.md#central-service-statistics).
