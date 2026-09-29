"""Processing DESY scans: beamtime ID & scan number instead of proposal & run"""
import json
import sqlite3
import textwrap
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from damnit.backend import initialize_proposal
from damnit.backend.combine import gather_all_fragments
from damnit.backend.db import DamnitDB
from damnit.backend.events import DESYEvent, EventProvider
from damnit.backend.extract_data import RunExtractor
from damnit.backend.extraction_control import available_runs
from damnit.backend.listener import EventProcessor
from damnit.context import RunData
from damnit.ctxsupport.desy_scan import DESYScan, beamtime_metadata, find_beamtime

BEAMTIME = 11024329


@pytest.fixture
def beamtime_dir(tmp_path, monkeypatch):
    """A fake beamtime folder in the DESY layout"""
    root = tmp_path / "asap3"
    bt_dir = root / "petra3/gpfs/p03/2026/data" / str(BEAMTIME)
    (bt_dir / "raw").mkdir(parents=True)
    (bt_dir / "processed").mkdir()
    (bt_dir / f"beamtime-metadata-{BEAMTIME}.json").write_text(json.dumps({
        "beamtimeId": str(BEAMTIME), "beamline": "p03", "proposalId": "20250003",
        "corePath": str(bt_dir), "title": "Test beamtime",
    }))
    monkeypatch.setenv("DESY_DATA_ROOT", str(root))
    monkeypatch.setenv("DESY_CURRENT_BEAMTIME", str(tmp_path / "no-current"))
    return bt_dir


def write_scan_file(path: Path, n_points=3):
    """A NeXus file laid out like the ones Sardana writes"""
    with h5py.File(path, "w") as f:
        entry = f.create_group("scan")
        entry["title"] = "mg_test, ascan exp_dmy01 0.0 1.0 2 0.1"
        entry["start_time"] = "2026-09-29T10:21:40+02:00"
        entry["end_reason"] = "SUCCESS"
        coll = entry.create_group("instrument/collection")
        coll["eh_c01"] = np.arange(n_points, dtype=float) * 10
        coll["exp_dmy01"] = np.linspace(0, 1, n_points)
        entry["data/eh_c01"] = h5py.SoftLink("/scan/instrument/collection/eh_c01")
    return path


def test_find_beamtime(beamtime_dir, tmp_path, monkeypatch):
    assert find_beamtime(BEAMTIME) == beamtime_dir
    assert find_beamtime(str(BEAMTIME)) == beamtime_dir
    assert beamtime_metadata(BEAMTIME)["beamline"] == "p03"

    with pytest.raises(FileNotFoundError):
        find_beamtime(99999999)

    # The current beamtime on a beamline machine, where only the metadata may
    # be there: prefer its corePath when that's accessible.
    current = tmp_path / "current"
    current.mkdir()
    md = {"beamtimeId": "11000001", "corePath": str(beamtime_dir)}
    (current / "beamtime-metadata-11000001.json").write_text(json.dumps(md))
    monkeypatch.setenv("DESY_CURRENT_BEAMTIME", str(current))
    assert find_beamtime(11000001) == beamtime_dir

    md["corePath"] = "/does/not/exist"
    (current / "beamtime-metadata-11000001.json").write_text(json.dumps(md))
    assert find_beamtime(11000001) == current


def test_desy_scan(beamtime_dir):
    scan_file = write_scan_file(beamtime_dir / "raw" / "tst_00007.nxs")
    scan = DESYScan(BEAMTIME, 7, filename=scan_file, scan_key="esrf:scan:gone",
                    redis_url="redis://localhost:1")
    try:
        # blissdata unavailable/expired -> info comes from the NeXus file
        assert scan.scan is None
        assert scan.info["title"].startswith("mg_test")
        assert scan.info["end_reason"] == "SUCCESS"
        assert scan.start_time == pytest.approx(1790671300)
        assert scan.nexus["scan/data/eh_c01"][:].tolist() == [0, 10, 20]
        assert scan.beamtime_path == beamtime_dir
        assert scan.beamtime_metadata["proposalId"] == "20250003"
    finally:
        scan.close()

    with pytest.raises(FileNotFoundError):
        DESYScan(BEAMTIME, 8).nexus


def test_desy_scan_waits_for_writer(beamtime_dir):
    scan_file = write_scan_file(beamtime_dir / "raw" / "tst_00007.nxs")
    real_open = h5py.File
    calls = []

    def locked_once(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise BlockingIOError(11, "unable to lock file")
        return real_open(*args, **kwargs)

    scan = DESYScan(BEAMTIME, 7, filename=scan_file)
    with patch("h5py.File", side_effect=locked_once), patch("time.sleep") as sleep:
        assert scan.nexus["scan/data/eh_c01"].shape == (3,)
    scan.close()
    assert len(calls) == 2
    sleep.assert_called_once_with(1)


def test_desy_event_source_info():
    event = DESYEvent(BEAMTIME, 7, filename="/data/tst_00007.nxs", scan_key="k",
                      blissdata_url="redis://h:6380")
    assert event.facility == "desy"
    assert event.source_info() == {"scan_file": "/data/tst_00007.nxs",
                                   "scan_key": "k", "blissdata_url": "redis://h:6380"}


class StubProvider(EventProvider):
    name = "stub"

    def events(self):
        return iter(())


def test_listener_records_desy_scans(tmp_path, beamtime_dir):
    db_dir = tmp_path / "damnit_db"
    initialize_proposal(db_dir, BEAMTIME)
    db = DamnitDB.from_dir(db_dir)

    processor = EventProcessor(tmp_path, provider=StubProvider())
    processor.db.add_proposal_db(BEAMTIME, db_dir, False)

    event = DESYEvent(BEAMTIME, 7, filename="/data/tst_00007.nxs", scan_key="k")
    with patch("damnit.backend.extraction_control.ExtractionSubmitter.submit") as submit:
        processor.handle_event(event)
    submit.assert_called_once()
    req = submit.call_args[0][0]
    assert (req.proposal, req.run, req.run_data) == (BEAMTIME, 7, RunData.RAW)

    assert db.metameta["facility"] == "desy"
    assert db.get_run_source(BEAMTIME, 7) == {"scan_file": "/data/tst_00007.nxs",
                                               "scan_key": "k"}
    assert db.get_run_source(BEAMTIME, 8) == {}
    # Only the file that exists counts as available for reprocessing
    assert available_runs(db, BEAMTIME) == set()


def test_get_run_source_errors(tmp_path):
    db = DamnitDB.from_dir(tmp_path, create=True)
    assert db.get_run_source(1, 1) == {}  # No run_sources table yet

    # Other database errors aren't hidden
    db.set_run_source(1, 1, {"scan_file": "x"})
    db.conn.execute("ALTER TABLE run_sources RENAME COLUMN info TO other")
    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        db.get_run_source(1, 1)
    db.close()


def test_desy_extraction(mock_db, beamtime_dir, monkeypatch):
    db_dir, db = mock_db
    db.metameta["proposal"] = BEAMTIME
    db.metameta["facility"] = "desy"
    monkeypatch.chdir(db_dir)

    scan_file = write_scan_file(beamtime_dir / "raw" / "tst_00007.nxs")
    db.ensure_run(BEAMTIME, 7)
    db.set_run_source(BEAMTIME, 7, {"scan_file": str(scan_file)})
    assert available_runs(db, BEAMTIME) == {7}
    assert available_runs(db, str(BEAMTIME)) == {7}

    (db_dir / "context.py").write_text(textwrap.dedent("""
    from damnit_ctx import Variable

    @Variable(title="Max counts")
    def max_c01(run):
        return run.nexus["scan/data/eh_c01"][:].max()

    @Variable(title="Scan title")
    def title(run):
        return run.info["title"]

    @Variable(title="Beamline")
    def beamline(run, beamline: "beamtime#beamline"):
        return beamline

    @Variable(title="Raw folder")
    def raw_folder(run, path: "meta#proposal_path", scan_no: "meta#run_number"):
        return f"{path.name}/raw scan {scan_no}"
    """))

    extractor = RunExtractor(BEAMTIME, 7, run_data=RunData.RAW)
    args = extractor._data_source_args()
    assert args == ["--facility", "desy", "--scan-file", str(scan_file)]

    from ctxrunner import main
    # extra_data must not be used to open DESY scans
    with patch("extra_data.open_run", side_effect=AssertionError("open_run called")):
        main(["exec", str(BEAMTIME), "7", "raw", *args])
    gather_all_fragments(db_dir)

    with h5py.File(db_dir / "extracted_data" / f"p{BEAMTIME}_r7.h5") as f:
        assert f["max_c01/data"][()] == 20
        assert f["title/data"].asstr()[()].startswith("mg_test")
        assert f["beamline/data"].asstr()[()] == "p03"
        assert f["raw_folder/data"].asstr()[()] == f"{BEAMTIME}/raw scan 7"
        assert f["start_time/data"][()] == pytest.approx(1790671300)


def test_blissdata_provider_official_db_dir(beamtime_dir):
    bp = pytest.importorskip("damnit.backend.blissdata_provider")
    provider = bp.BlissdataEventProvider(data_store=object())
    assert provider.official_db_dir(BEAMTIME) == beamtime_dir / "processed/_damnit"
    assert provider.official_db_dir(99999999) is None
