import json
import logging
import os
import subprocess
from datetime import datetime
from tempfile import TemporaryFile
from threading import Timer


log = logging.getLogger(__name__)
JOURNAL_TIMEOUT_SECONDS = 60


class JournalError(Exception):
    """The user journal could not be read."""


def read_journal(start: datetime, end: datetime):
    """Yield journal records for both central services in chronological order."""
    command = [
        "journalctl", "--user", "--no-pager", "--output=json",
        "--user-unit=damnit-combiner.service", "--user-unit=damnit-listener.service",
        "--output-fields=MESSAGE,_SYSTEMD_USER_UNIT,SYSLOG_IDENTIFIER,_PID,_SYSTEMD_INVOCATION_ID",
        f"--since=@{start.timestamp():.6f}", f"--until=@{end.timestamp():.6f}",
    ]
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

        def stop_reader():
            nonlocal timed_out
            timed_out = True
            process.kill()

        with process:
            timer = Timer(JOURNAL_TIMEOUT_SECONDS, stop_reader)
            timer.start()
            try:
                for line in process.stdout:
                    yield json.loads(line)
                process.wait()
            except (ValueError, OSError) as exc:
                read_error = exc
            finally:
                timer.cancel()
                timer.join()
                if process.poll() is None:
                    process.kill()

        if timed_out:
            raise JournalError("The journal query timed out. Try a shorter time range.")
        if read_error:
            raise JournalError("Could not read journalctl JSON output.") from read_error
        errors.seek(0)
        diagnostic = errors.read(8192).decode(errors="replace").strip()
        if process.returncode or diagnostic:
            log.warning("journalctl exited %d: %s", process.returncode, diagnostic)
            raise JournalError(
                "Could not read the service journal. Check the dashboard account's journal access."
            )
