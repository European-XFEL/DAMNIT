import json
import logging
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from tempfile import TemporaryFile
from threading import Timer


log = logging.getLogger(__name__)
JOURNAL_TIMEOUT_SECONDS = 60


class JournalError(Exception):
    """The user journal could not be read."""


def _time_options(start, end):
    options = []
    if start is not None:
        options.append(f"--since=@{start.timestamp():.6f}")
    if end is not None:
        until = end - timedelta(microseconds=1)  # journalctl includes its upper bound.
        options.append(f"--until=@{until.timestamp():.6f}")
    return options


def read_journal(start: datetime, end: datetime):
    """Yield journal records for both services in the half-open interval [start, end)."""
    return _journal_records([
        "--user-unit=damnit-combiner.service", "--user-unit=damnit-listener.service",
        "--output-fields=MESSAGE,_SYSTEMD_USER_UNIT,USER_UNIT,OBJECT_SYSTEMD_USER_UNIT,"
        "SYSLOG_IDENTIFIER,_PID,_SYSTEMD_INVOCATION_ID",
        *_time_options(start, end),
    ])


def preceding_combiner_records(before, sources):
    """Read potential predecessor records for a batch of workers, newest first."""
    options = ["--user-unit=damnit-combiner.service", "--reverse", *_time_options(None, before)]
    # Multiple matches on _PID are ORed by journalctl. Validate the full
    # boot/invocation/PID identity when reading. Bound the command-line size.
    pids = {source[2] for source in sources}
    if None not in pids and len(pids) <= 100:
        options.extend(f"_PID={pid}" for pid in sorted(map(str, pids)))
    return _journal_records(options)


def read_log_entries(service: str, query: str, limit: int, start=None, end=None):
    options = [f"--user-unit=damnit-{service}.service", "--output-fields=MESSAGE",
               "--reverse", f"--lines={limit}", *_time_options(start, end)]
    if query:
        # An argv string cannot contain NUL; PCRE accepts its hexadecimal escape.
        literal = re.escape(query).replace("\0", r"\x00")
        options.extend([f"--grep={literal}", "--case-sensitive=no"])
    entries = []
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    for record in _journal_records(options, allow_no_matches=bool(query)):
        message = record.get("MESSAGE")
        try:
            if isinstance(message, list):
                # journalctl encodes a binary field as an array of unsigned bytes.
                message = bytes(message).decode("utf-8", errors="replace")
            timestamp = epoch + timedelta(microseconds=int(record["__REALTIME_TIMESTAMP"]))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if isinstance(message, str) and len(entries) < limit:
            entries.append({"timestamp": timestamp.isoformat(), "message": message})
    return entries


def journal_availability(end):
    """Classify an empty scoped search using the whole readable user journal.

    A unit's first message can be much later than the journal's retention
    boundary. Only classify a window as older when all retained records are newer.
    """
    latest = ["--reverse", "--lines=1", "--output-fields=__REALTIME_TIMESTAMP"]
    if list(_journal_records([*latest, *_time_options(None, end)])):
        return "available"
    if end is not None and list(_journal_records(latest)):
        return "outside_retention"
    return "unknown"


def _journal_records(options, *, allow_no_matches=False, timeout_seconds=None):
    command = ["journalctl", "--user", "--no-pager", "--all", "--output=json", *options]
    # A separate stderr file lets us stream stdout without a second pipe filling.
    with TemporaryFile() as errors:
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=errors,
                env={**os.environ, "LC_ALL": "C"},
            )
        except OSError as exc:
            log.warning("Could not start journalctl: %s", exc)
            raise JournalError("Could not run journalctl on the dashboard host.") from exc

        timed_out = False
        read_error = None
        emitted = False

        def stop_reader():
            nonlocal timed_out
            timed_out = True
            process.kill()

        with process:
            timer = Timer(JOURNAL_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds,
                          stop_reader)
            timer.start()
            try:
                for line in process.stdout:
                    emitted = True
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Expected a JSON object for a journal record")
                    yield record
                process.wait()
            except (ValueError, OSError) as exc:
                read_error = exc
            finally:
                timer.cancel()
                timer.join()
                if process.poll() is None:
                    process.kill()

        if timed_out:
            raise JournalError("The journal query timed out.")
        if read_error:
            raise JournalError("Could not read journalctl JSON output.") from read_error
        errors.seek(0)
        diagnostic = errors.read(8192).decode(errors="replace").strip()
        if allow_no_matches and process.returncode == 1 and not emitted and not diagnostic:
            return  # journalctl --grep exits 1 when the journal contains no matches.
        if process.returncode or diagnostic:
            log.warning("journalctl exited %d: %s", process.returncode, diagnostic)
            raise JournalError(
                "Could not read the service journal. Check the dashboard account's journal access."
            )
