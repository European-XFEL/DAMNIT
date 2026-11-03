"""DESY: scan events from the state of blissdata scans in Redis.

Every scan published to blissdata (by BLISS or Sardana) becomes a `DESYEvent`
with the scan's beamtime ID and scan number. The event is emitted once the
scan reaches its trigger state (CLOSED by default, when the data and the scan
info are final).

Select it for a listener, and optionally configure it, with:

    damnit listen --event-provider blissdata
    damnit listener config blissdata-url redis://<host>:6380
    damnit listener config blissdata-session <session>  # optional filter
"""
import logging
import os
import time
from datetime import datetime

from blissdata.redis_engine.scan import Scan, ScanState
from blissdata.redis_engine.store import DataStore

try:
    from blissdata.exceptions import NoScanAvailable, ScanLoadError, ScanNotFoundError
except ImportError:  # blissdata < 2.4
    from blissdata.redis_engine.exceptions import (
        NoScanAvailable, ScanLoadError, ScanNotFoundError
    )

from ..context import RunData
from ..ctxsupport.desy_scan import find_beamtime
from .events import DESYEvent, EventProvider

log = logging.getLogger(__name__)

DEFAULT_BLISSDATA_URL = "redis://localhost:6380"


class BlissdataEventProvider(EventProvider):
    name = "blissdata"

    def __init__(self, redis_url=DEFAULT_BLISSDATA_URL, session=None,
                 trigger_state=ScanState.CLOSED, only_successful=False,
                 poll_interval=1.0, data_store=None):
        """
        :param redis_url: blissdata Redis URL
        :param session: only scans of this BLISS/Sardana session, None for all
        :param trigger_state: scan state at which the scan is processed
        :param only_successful: skip scans whose end_reason isn't SUCCESS
        :param poll_interval: seconds between checks of unfinished scans
        """
        self.redis_url = redis_url
        self.session = session
        self.trigger_state = ScanState(trigger_state)
        self.only_successful = only_successful
        self.poll_interval = poll_interval
        self._data_store = data_store
        # Scans seen but not yet in the trigger state, by key
        self._pending: dict[str, Scan] = {}

    @classmethod
    def from_settings(cls, settings):
        trigger_state = settings.get("blissdata_trigger_state", "CLOSED")
        return cls(
            redis_url=(settings.get("blissdata_url")
                       or os.environ.get("BLISSDATA_URL", DEFAULT_BLISSDATA_URL)),
            session=settings.get("blissdata_session") or None,
            trigger_state=ScanState[str(trigger_state).upper()],
            only_successful=bool(settings.get("blissdata_only_successful", False)),
        )

    @property
    def data_store(self):
        if self._data_store is None:
            self._data_store = DataStore(self.redis_url)
        return self._data_store

    def official_db_dir(self, proposal):
        try:
            return find_beamtime(proposal) / "processed/_damnit"
        except (FileNotFoundError, ValueError) as e:
            log.warning("Could not find beamtime folder for %s: %s", proposal, e)
            return None

    def events(self):
        log.info("Watching blissdata scans at %s (session: %s)",
                 self.redis_url, self.session or "all")
        # Only scans created from now on, like a Kafka consumer group would
        since = self.data_store.get_last_scan_timetag()
        while True:
            try:
                # Block until the next scan, or the timeout so that the
                # pending scans are still polled regularly.
                since, key = self.data_store.get_next_scan(
                    since=since, timeout=self.poll_interval)
            except NoScanAvailable:
                pass
            else:
                self._add_scan(key)

            yield from self._check_pending()

    def _add_scan(self, key):
        try:
            scan = self.data_store.load_scan(key)
        except ScanNotFoundError:
            return  # Already deleted from Redis
        except ScanLoadError:
            log.warning("Cannot load scan %r", key, exc_info=True)
            return

        if self.session is not None and scan.session != self.session:
            log.debug("Ignoring scan %s from session %r", key, scan.session)
            return

        log.debug("New scan %s (%s, state %s)", key, scan.name, scan.state.name)
        self._pending[key] = scan

    def _check_pending(self):
        for key, scan in list(self._pending.items()):
            try:
                # Consume all the state changes since the last check
                while scan.update(block=False):
                    pass
            except ScanNotFoundError:
                log.warning("Scan %s was deleted before it finished", key)
                del self._pending[key]
                continue
            except Exception:
                log.error("Error updating scan %s, dropping it", key, exc_info=True)
                del self._pending[key]
                continue

            if scan.state >= self.trigger_state:
                del self._pending[key]
                try:
                    event = self.scan_to_event(scan)
                except Exception:
                    log.error("Could not make an event from scan %s", key, exc_info=True)
                    continue
                if event is not None:
                    yield event

    def scan_to_event(self, scan: Scan) -> DESYEvent | None:
        info = scan.info
        end_reason = info.get("end_reason")
        if self.only_successful and end_reason != "SUCCESS":
            log.info("Skipping scan %s which ended with %s", scan.key, end_reason)
            return None

        beamtime_id = info.get("beamtime_id") or getattr(scan, "proposal", None)
        scan_number = info.get("scan_nb", scan.number)
        try:
            beamtime_id, scan_number = int(beamtime_id), int(scan_number)
        except (TypeError, ValueError):
            log.warning("Skipping scan %s (%s): no numeric beamtime ID/scan number "
                        "(got %r, %r)", scan.key, scan.name, beamtime_id, scan_number)
            return None

        return DESYEvent(
            beamtime_id=beamtime_id,
            scan_number=scan_number,
            run_data=RunData.RAW,
            timestamp=_parse_time(info.get("end_time")),
            scan_name=scan.name,
            session=scan.session,
            filename=info.get("filename"),
            end_reason=end_reason,
            scan_key=scan.key,
            blissdata_url=self.redis_url,
            metadata={"state": scan.state.name, "title": info.get("title")},
        )


def _parse_time(iso_time):
    if iso_time:
        try:
            return datetime.fromisoformat(iso_time).timestamp()
        except ValueError:
            pass
    return time.time()
