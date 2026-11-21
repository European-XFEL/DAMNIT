import json
import os
import re
from datetime import datetime, timezone

import pytest

from damnit.dashboard import journal


START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = START.replace(hour=1, microsecond=123456)


@pytest.fixture
def fake_journal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)

    def install(script):
        command = tmp_path / "journalctl"
        command.write_text("#!/bin/sh\n" + script)
        command.chmod(0o755)

    return install


def test_read_journal(fake_journal, tmp_path):
    fake_journal(
        """printf '%s\\n' "$@" > arguments
echo '{"_SYSTEMD_USER_UNIT":"damnit-combiner.service","MESSAGE":"first"}'
echo '{"_SYSTEMD_USER_UNIT":"damnit-listener.service","MESSAGE":"second"}'
"""
    )
    assert [row["MESSAGE"] for row in journal.read_journal(START, END)] == ["first", "second"]
    arguments = (tmp_path / "arguments").read_text().splitlines()
    assert {
        "--user", "--no-pager", "--output=json", "--all",
        "--user-unit=damnit-combiner.service", "--user-unit=damnit-listener.service",
        "--since=@1704067200.000000", "--until=@1704070800.123455",
    } <= set(arguments)
    fields, = [arg.removeprefix("--output-fields=") for arg in arguments if arg.startswith("--output-fields=")]
    assert {"MESSAGE", "_SYSTEMD_USER_UNIT", "USER_UNIT", "OBJECT_SYSTEMD_USER_UNIT"} <= set(fields.split(","))


def test_empty_journal(fake_journal):
    fake_journal("exit 0")
    assert list(journal.read_journal(START, END)) == []


@pytest.mark.parametrize(("script", "error"), [
    ("echo 'No journal files were opened due to insufficient permissions.' >&2", "journal access"),
    ("exit 1", "journal access"),
    ("echo 'invalid JSON'", "JSON output"),
])
def test_journal_failure(fake_journal, script, error):
    fake_journal(script)
    with pytest.raises(journal.JournalError, match=error):
        list(journal.read_journal(START, END))


def test_timeout_mid_record_reaps_reader(fake_journal, tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "JOURNAL_TIMEOUT_SECONDS", 0.5)
    fake_journal("""echo $$ > reader.pid
printf '{"MESSAGE":'
exec sleep 30
""")
    with pytest.raises(journal.JournalError, match="timed out"):
        list(journal.read_journal(START, END))
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "reader.pid").read_text()), 0)


def test_literal_log_search_command_and_raw_messages(fake_journal, tmp_path):
    literal = "-ALERT.*[0]\\E\nsecond\0"
    binary = list(b"first line\x00\nsecond\xff")
    records = [
        {"__REALTIME_TIMESTAMP": "1704067201000020", "MESSAGE": "  raw\nMESSAGE\n"},
        {"__REALTIME_TIMESTAMP": "1704067200000000", "MESSAGE": binary},
    ]
    (tmp_path / "records").write_text("".join(json.dumps(record) + "\n" for record in records))
    fake_journal("""printf '%s\\0' "$@" > args
cat records
""")
    assert journal.read_log_entries("listener", literal, 2) == [
        {"timestamp": "2024-01-01T00:00:01.000020+00:00", "message": "  raw\nMESSAGE\n"},
        {"timestamp": "2024-01-01T00:00:00+00:00", "message": "first line\0\nsecond\ufffd"},
    ]
    args = (tmp_path / "args").read_bytes().decode().split("\0")
    assert {"--user", "--user-unit=damnit-listener.service", "--all", "--output-fields=MESSAGE",
            "--reverse", "--lines=2", "--case-sensitive=no"} <= set(args)
    assert [arg for arg in args if arg.startswith("--user-unit=")] == ["--user-unit=damnit-listener.service"]
    assert not any(arg.startswith(("--since", "--until")) for arg in args)
    pattern, = [arg.removeprefix("--grep=") for arg in args if arg.startswith("--grep=")]
    assert re.fullmatch(pattern, literal.swapcase(), re.IGNORECASE)
    assert not re.fullmatch(pattern, literal.replace(".*[0]", "0"), re.IGNORECASE)


def test_raw_logs_without_filter(fake_journal, tmp_path):
    fake_journal("""printf '%s\\n' "$@" > args
echo '{"__REALTIME_TIMESTAMP":"1","MESSAGE":"one"}'
echo '{"__REALTIME_TIMESTAMP":"0","MESSAGE":"two"}'
""")
    assert len(journal.read_log_entries("combiner", "", 1)) == 1
    args = (tmp_path / "args").read_text().splitlines()
    assert "--user-unit=damnit-combiner.service" in args
    assert not any(arg.startswith("--grep") for arg in args)


def test_no_matches_and_search_failure(fake_journal):
    fake_journal("exit 1")
    assert journal.read_log_entries("combiner", "absent", 3) == []
    with pytest.raises(journal.JournalError):
        journal.read_log_entries("combiner", "", 3)
    for script in ["echo 'access denied' >&2; exit 1",
                   """echo '{"__REALTIME_TIMESTAMP":"1","MESSAGE":"partial"}'; exit 1""",
                   "exit 2"]:
        fake_journal(script)
        with pytest.raises(journal.JournalError):
            journal.read_log_entries("combiner", "missing", 3)
