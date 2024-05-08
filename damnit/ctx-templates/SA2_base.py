from datetime import timedelta

import numpy as np

from damnit_ctx import Variable


# If you have the Karabo device name of the XGM you're interested in, you can
# set the variable below and the 'xgm_intensity' and 'pulses' Variable's will
# use it. By default it's set to the XGM for SASE 2.
xgm_name = "SA2_XTD1_XGM/XGM/DOOCS"

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
    xgm = run[f"{xgm_name}:output", 'data.intensityTD'].xarray()
    return xgm[:, np.where(xgm[0] > 1)[0]].mean(axis=1)

@Variable(title="Pulses", summary="mean")
def pulses(run):
    return run[xgm_name, 'pulseEnergy.numberOfBunchesActual'].xarray()
