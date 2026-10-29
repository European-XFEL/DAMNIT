"""Opening DESY scans for context file variables.

At DESY, data is organised by beamtime ID & scan number rather than proposal &
run. A beamtime folder looks like:

    /asap3/petra3/gpfs/<beamline>/<year>/data/<beamtime_id>/
        beamtime-metadata-<beamtime_id>.json
        raw/         scan data (e.g. NeXus files written by Sardana/BLISS)
        processed/   processed data, and DAMNIT's database in processed/_damnit
        shared/  scratch_cc/  meta/

This module runs in the context file's Python environment, so it only needs
h5py; blissdata is used if it's installed.
"""
import json
import logging
import os
import time
from datetime import datetime
from glob import glob
from pathlib import Path

log = logging.getLogger(__name__)

# Beamtime folders are found under $DESY_DATA_ROOT/<facility>/gpfs/<beamline>/<year>/data/
DEFAULT_DATA_ROOT = "/asap3"
# On beamline machines, the current beamtime is also mounted here
CURRENT_BEAMTIME = "/gpfs/current"


def _metadata_file(beamtime_dir: Path, beamtime_id):
    return beamtime_dir / f"beamtime-metadata-{beamtime_id}.json"


def find_beamtime(beamtime_id) -> Path:
    """Find the folder of a DESY beamtime.

    Raises FileNotFoundError if it can't be found.
    """
    beamtime_id = str(beamtime_id)

    # The current beamtime on a beamline machine. Its metadata says where the
    # beamtime is stored permanently; prefer that if it's accessible.
    current = Path(os.environ.get("DESY_CURRENT_BEAMTIME", CURRENT_BEAMTIME))
    if (md_file := _metadata_file(current, beamtime_id)).is_file():
        core_path = json.loads(md_file.read_text()).get("corePath")
        if core_path and Path(core_path).is_dir():
            return Path(core_path)
        return current

    root = os.environ.get("DESY_DATA_ROOT", DEFAULT_DATA_ROOT)
    matches = sorted(glob(f"{root}/*/gpfs/*/*/data/{beamtime_id}"))
    if len(matches) == 1:
        return Path(matches[0])
    elif len(matches) > 1:
        raise ValueError(f"Multiple folders found for beamtime {beamtime_id}: {matches}")

    raise FileNotFoundError(f"Couldn't find folder for beamtime {beamtime_id!r}")


def beamtime_metadata(beamtime_id, beamtime_dir=None) -> dict:
    """Load beamtime-metadata-<id>.json (beamline, title, proposalId, ...)"""
    if beamtime_dir is None:
        beamtime_dir = find_beamtime(beamtime_id)
    return json.loads(_metadata_file(Path(beamtime_dir), beamtime_id).read_text())


def _open_when_unlocked(path, attempts=6):
    """Open an HDF5 file read-only, waiting while its writer still has it locked.

    blissdata can mark a scan as closed shortly before the NeXus writer has
    finished with the file.
    """
    import h5py

    for i in range(attempts):
        try:
            return h5py.File(path, "r")
        except BlockingIOError:
            if i == attempts - 1:
                raise
            # File is locked for writing; wait 1, 2, 4, ... seconds
            log.info("%s is locked by its writer, retrying in %d s", path, 2 ** i)
            time.sleep(2 ** i)


class DESYScan:
    """A DESY scan, passed to Variable functions as their first argument.

    - ``nexus``: the scan's NeXus/HDF5 file, opened read-only with h5py
    - ``scan``: the blissdata Scan, while it's still in Redis (else None)
    - ``info``: the scan info (from blissdata, or else from the NeXus file)
    - ``beamtime_path`` / ``beamtime_metadata``: the beamtime folder & its
      metadata JSON
    """

    def __init__(self, beamtime_id: int, scan_number: int, filename=None,
                 scan_key=None, redis_url=None):
        self.beamtime_id = int(beamtime_id)
        self.scan_number = int(scan_number)
        self.filename = Path(filename) if filename else None
        self.scan_key = scan_key
        self.redis_url = redis_url
        self._nexus = None
        self._scan = None
        self._scan_loaded = False

    def __repr__(self):
        return (f"<DESYScan beamtime {self.beamtime_id} scan {self.scan_number}"
                f" file={str(self.filename)!r}>")

    @property
    def nexus(self):
        if self._nexus is None:
            if self.filename is None:
                raise FileNotFoundError(
                    f"No data file known for beamtime {self.beamtime_id} scan {self.scan_number}")
            self._nexus = _open_when_unlocked(self.filename)
        return self._nexus

    @property
    def scan(self):
        """The blissdata Scan, or None if it's unavailable or expired"""
        if not self._scan_loaded:
            self._scan_loaded = True
            if self.scan_key and self.redis_url:
                try:
                    from blissdata.redis_engine.store import DataStore
                    self._scan = DataStore(self.redis_url).load_scan(self.scan_key)
                except ImportError:
                    log.info("blissdata is not installed, scan.scan is unavailable")
                except Exception as e:
                    log.info("Could not load scan %s from blissdata: %s", self.scan_key, e)
        return self._scan

    @property
    def info(self) -> dict:
        if self.scan is not None:
            return self.scan.info
        return self._nexus_info()

    def _nexus_info(self) -> dict:
        if self.filename is None or not self.filename.is_file():
            return {}
        info = {}
        entry = self._nexus_entry()
        for key in ("title", "start_time", "end_time", "end_reason"):
            if key in entry:
                value = entry[key][()]
                info[key] = value.decode() if isinstance(value, bytes) else value
        return info

    def _nexus_entry(self):
        # Sardana writes a single NXentry, e.g. /scan
        for group in self.nexus.values():
            return group
        return {}

    @property
    def beamtime_path(self) -> Path:
        return find_beamtime(self.beamtime_id)

    @property
    def beamtime_metadata(self) -> dict:
        return beamtime_metadata(self.beamtime_id)

    @property
    def start_time(self) -> float | None:
        """Scan start time in seconds since the epoch, if known"""
        start = self.info.get("start_time")
        if start:
            try:
                return datetime.fromisoformat(start).timestamp()
            except ValueError:
                pass
        if self.filename is not None and self.filename.is_file():
            return self.filename.stat().st_mtime
        return None

    def close(self):
        if self._nexus is not None:
            self._nexus.close()
            self._nexus = None
