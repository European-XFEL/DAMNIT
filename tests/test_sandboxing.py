import os
import subprocess
import sysconfig
from pathlib import Path
from unittest.mock import MagicMock
import sys
from mock import patch

import pytest
from damnit.backend.extract_data import extract_in_subprocess

from damnit.backend.sandboxing import Bubblewrap


@pytest.fixture
def bubblewrap():
    return Bubblewrap()


@pytest.fixture(scope="session")
def mock_proposal_1111(tmp_path_factory):
    proposal_str = "p0001111"
    root: Path = tmp_path_factory.mktemp("root")

    # 'real' directories
    usr = root / "u" / "usr" / proposal_str
    raw = root / "pnfs" / "archive" / proposal_str

    for d in (usr, raw):
        d.mkdir(parents=True)

    # fake gpfs structure - proposal dir with symlinks to usr and raw
    gpfs = root / "gpfs"
    p = gpfs / "p0001111"

    p.mkdir(parents=True)

    usr_link = p / "usr"
    usr_link.symlink_to(usr)

    raw_link = p / "raw"
    raw_link.symlink_to(raw)

    return p


@pytest.mark.parametrize(
    "src,dest,ro,expected",
    [
        (Path("/source"), None, False, ("--bind", "/source", "/source")),
        (Path("/source"), None, True, ("--ro-bind", "/source", "/source")),
        (Path("/source"), Path("/dest"), False, ("--bind", "/source", "/dest")),
    ],
)
def test_add_bind(bubblewrap, src, dest, ro, expected):
    bubblewrap.add_bind(src, dest, ro)

    assert expected in bubblewrap.command_binds


def test_add_bind_proposal(bubblewrap, monkeypatch, mock_proposal_1111):
    proposal_id = 1111

    find_proposal = MagicMock(return_value=str(mock_proposal_1111))
    monkeypatch.setattr("damnit.backend.sandboxing.find_proposal", find_proposal)

    bubblewrap.add_bind_proposal(proposal_id)

    assert find_proposal.call_args == ((f"p{proposal_id:06d}",),)

    binds = [b[1] for b in bubblewrap.command_binds]

    assert str(mock_proposal_1111) in binds

    assert any("u/usr" in b for b in binds)
    assert any("pnfs/archive" in b for b in binds)


def test_add_bind_venv(bubblewrap, monkeypatch):
    python_exec = Path("/path/to/python")

    paths = [
        "/path/venv/lib",
        "/path/venv/include",
        "/path/venv/bin",
    ]

    monkeypatch.setattr(
        "subprocess.check_output",
        MagicMock(return_value="\n".join(paths).encode("utf-8")),
    )

    bubblewrap.add_bind_venv(python_exec)

    assert ("--ro-bind", "/path/venv/lib", "/path/venv/lib") in bubblewrap.command_binds
    assert (
        "--ro-bind",
        "/path/venv/include",
        "/path/venv/include",
    ) in bubblewrap.command_binds
    assert ("--ro-bind", "/path/venv/bin", "/path/venv/bin") in bubblewrap.command_binds


def test_add_bind_venv_with_subprocess(bubblewrap):
    python_exec = Path(sys.executable)
    bubblewrap.add_bind_venv(python_exec)

    for path in sysconfig.get_paths().values():
        assert ("--ro-bind", str(path), str(path)) in bubblewrap.command_binds


def test_write_in_bwrap(bubblewrap, tmp_path):
    bubblewrap.add_bind(tmp_path)

    out = tmp_path / "text"
    cmd = ["touch", str(out.absolute())]
    cmd = bubblewrap.build_command(cmd)

    subprocess.check_call(cmd)

    assert out.exists()

    assert out.stat().st_uid == os.getuid()


def test_write_in_bwrap_ro_fail(bubblewrap, tmp_path):
    bubblewrap.add_bind(tmp_path, ro=True)

    res = subprocess.call(
        bubblewrap.build_command(["touch", str(tmp_path / "hi")]),
        stderr=subprocess.STDOUT,
    )

    assert res != 0


@patch("damnit.backend.extract_data.load_reduced_data")
def test_extract_data_call(bubblewrap, tmp_path):
    out_path = tmp_path / "out"
    out_path.mkdir(parents=True)

    with patch("subprocess.run") as subprocess_run:
        extract_in_subprocess(0, 0, out_path)
        subprocess_run.assert_called()
        args = subprocess_run.call_args[0][0]
        bwrap = args[: args.index("--")]

    out_file = out_path / "touch"
    subprocess.check_call([*bwrap, "--", "touch", str(out_file)])
    assert out_file.exists()

    root = Path(__file__).parent.parent
    subprocess.check_call([*bwrap, "--", "stat", str(root)])
