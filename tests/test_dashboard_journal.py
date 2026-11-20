import os
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
        "--user", "--no-pager", "--output=json",
        "--user-unit=damnit-combiner.service", "--user-unit=damnit-listener.service",
        "--since=@1704067200.000000", "--until=@1704070800.123456",
    } <= set(arguments)


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
