"""Events announcing that new data is ready to be processed.

There are two separate pieces here:

- The *event model* describes the data in a facility's own terms. At European
  XFEL data is organised by proposal & run (`XFELEvent`), at DESY by beamtime
  ID & scan number (`DESYEvent`). Each model maps its IDs onto the
  (proposal, run) numbers that DAMNIT's database is keyed by.
- The *event provider* is where events come from: Kafka messages
  (`kafka_provider.KafkaEventProvider`) or blissdata scan states in Redis
  (`blissdata_provider.BlissdataEventProvider`).
"""
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..context import RunData


class Event(ABC):
    """Base class of the facility-specific event models"""

    facility: str  # How the data is opened for processing: 'xfel' or 'desy'
    run_data: RunData
    timestamp: float  # Seconds since the epoch
    metadata: dict

    @property
    @abstractmethod
    def damnit_proposal(self) -> int:
        """Number used as the proposal in the DAMNIT database"""

    @property
    @abstractmethod
    def damnit_run(self) -> int:
        """Number used as the run in the DAMNIT database"""

    @abstractmethod
    def describe(self) -> str:
        """Short description for logs, in the facility's own terms"""

    def source_info(self) -> dict:
        """Where to find this run's data, saved in the database so it can be
        processed again later. Empty if it can be found from the IDs alone.
        """
        return {}

    def __str__(self):
        return f"{self.describe()} ({self.run_data.value} data)"


@dataclass(frozen=True)
class XFELEvent(Event):
    """European XFEL: data for a run of a proposal is ready"""
    facility = "xfel"

    proposal: int
    run: int
    run_data: RunData
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict, compare=False)

    @property
    def damnit_proposal(self):
        return self.proposal

    @property
    def damnit_run(self):
        return self.run

    def describe(self):
        return f"p{self.proposal} r{self.run}"


@dataclass(frozen=True)
class DESYEvent(Event):
    """DESY: a scan of a beamtime has finished"""
    facility = "desy"

    beamtime_id: int
    scan_number: int
    run_data: RunData = RunData.RAW
    timestamp: float = field(default_factory=time.time)
    scan_name: str | None = None
    session: str | None = None
    filename: str | None = None  # Where the scan data is written
    end_reason: str | None = None  # e.g. SUCCESS, USER_ABORT, FAILURE
    scan_key: str | None = None  # blissdata key, while the scan is in Redis
    blissdata_url: str | None = None
    metadata: dict = field(default_factory=dict, compare=False)

    @property
    def damnit_proposal(self):
        return self.beamtime_id

    @property
    def damnit_run(self):
        return self.scan_number

    def describe(self):
        return f"beamtime {self.beamtime_id} scan {self.scan_number}"

    def source_info(self):
        return {k: v for k, v in {
            "scan_file": self.filename,
            "scan_key": self.scan_key,
            "blissdata_url": self.blissdata_url,
            "scan_name": self.scan_name,
            "session": self.session,
        }.items() if v is not None}


class EventProvider(ABC):
    """Base class of the sources of events"""

    name = "abstract"

    @abstractmethod
    def events(self) -> Iterator[Event]:
        """Yield events as they arrive, blocking in between.

        Implementations should log and skip malformed messages rather than
        raising, so that one bad message doesn't stop the listener.
        """

    def official_db_dir(self, proposal: int) -> Path | None:
        """Where the official DAMNIT database for a proposal lives, if any.

        Used to add proposals automatically when the listener isn't in static
        mode. Returning None means databases must be added explicitly.
        """
        return None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


EVENT_PROVIDERS = ("kafka", "blissdata")
DEFAULT_EVENT_PROVIDER = "kafka"


def make_event_provider(settings, listener_dir: Path) -> EventProvider:
    """Create the event provider selected by the listener settings.

    Change it with `damnit listen --event-provider blissdata` or
    `damnit listener config event-provider blissdata`.
    """
    kind = settings.get("event_provider", DEFAULT_EVENT_PROVIDER)
    if kind == "kafka":
        from .kafka_provider import KafkaEventProvider
        return KafkaEventProvider(listener_dir)
    elif kind == "blissdata":
        from .blissdata_provider import BlissdataEventProvider
        return BlissdataEventProvider.from_settings(settings)
    else:
        raise ValueError(f"Unknown event_provider {kind!r}, expected one of {EVENT_PROVIDERS}")
