from datetime import timedelta

import numpy as np

from extra.components import XGM, XrayPulses
from damnit_ctx import Variable


# Run metadata

@Variable(title="Trains")
def n_trains(run):
    return len(run.train_ids)

@Variable(title="Run length")
def run_length(run):
    ts = run.train_timestamps()
    delta = ts[-1] - ts[0]
    delta_s = int(delta / np.timedelta64(1, 's'))
    return str(timedelta(seconds=delta_s))

def run_size_tb(path):
    run_size_bytes = sum(f.stat().st_size for f in path.rglob('*'))
    return run_size_bytes / 1e12

@Variable(title="Raw size [TB]")
def raw_size(run, proposal_path: "meta#proposal_path", run_no: "meta#run_number"):
    run_path = proposal_path / "raw" / f"r{run_no:04}"
    return run_size_tb(run_path)

@Variable(title="Proc size [TB]", data="proc")
def proc_size(run, proposal_path: "meta#proposal_path", run_no: "meta#run_number"):
    run_path = proposal_path / "proc" / f"r{run_no:04}"
    return run_size_tb(run_path)

# Beam properties

@Variable(title="XGM intensity [uJ]", summary="mean")
def xgm_intensity(run):
    return XGM(run).pulse_energy().mean("pulseIndex")

@Variable(title="Pulses", summary="mean")
def pulses(run):
    return XrayPulses(run).pulse_counts().to_xarray()
