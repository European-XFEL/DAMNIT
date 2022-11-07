import os
from pathlib import Path

import numpy as np

from damnit_ctx import Variable


# If you have the Karabo device name of the XGM you're interested in, you can
# set the variable below and the 'xgm_intensity' and 'pulses' Variable's will
# use it. By default it's set to the XGM for SASE 2.
xgm_name = "SA2_XTD1_XGM/XGM/DOOCS"

@Variable(title="XGM intensity", summary="mean")
def xgm_intensity(run):
    xgm = run[f"{xgm_name}:output", 'data.intensityTD'].xarray()
    return xgm[:, np.where(xgm[0] > 1)[0]].sum(axis=1)

@Variable(title="Pulses")
def pulses(run):
    return int(run[xgm_name, 'pulseEnergy.numberOfBunchesActual'][0].ndarray()[0])

@Variable(title="Trains")
def n_trains(run):
    return len(run.train_ids)

@Variable(title="Run size (TB)")
def raw_size(run):
    # If the run only has one file then it's most likely the virtual overview
    # file, which has negligible size. To get the actual run size we find the
    # run directory and get it's size directly.
    if len(run.files) == 1:
        voview_path = Path(run.files[0].filename)
        run_dir = voview_path.parts[-1].split("-")[1].lower()
        proposal_path = voview_path.parents[2]

        run_path = proposal_path / "raw" / run_dir
        run_size_bytes = sum(f.stat().st_size for f in run_path.rglob('*'))
    else:
        # Otherwise, we just use the files opened by extra-data
        run_size_bytes = 0

        for f in run.files:
            run_size_bytes += os.path.getsize(f.filename)

    return run_size_bytes / 1e12
