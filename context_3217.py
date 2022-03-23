import os

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

import pasha
from extra_geom import AGIPD_1MGeometry
from extra_data import stack_detector_data
from extra_data.components import AGIPD1M

from amore_mid_prototype.context import Variable

@Variable(title='XGM beam intensity', summary='mean')
def xgm_intensity(run):
    return run['SA2_XTD1_XGM/XGM/DOOCS:output', 'data.intensityTD'].xarray(roi=(0,))

@Variable(title="ROI intensity", summary="mean")
def roi_intensity(run):
    # Create geometry
    q1m1_src = ("MID_EXP_AGIPD1M/MOTOR/Q1M1", "actualPosition")
    q2m1_src = ("MID_EXP_AGIPD1M/MOTOR/Q2M1", "actualPosition")
    q3m1_src = ("MID_EXP_AGIPD1M/MOTOR/Q3M1", "actualPosition")
    q4m1_src = ("MID_EXP_AGIPD1M/MOTOR/Q4M1", "actualPosition")

    q1m1 = run[q1m1_src].as_single_value()
    q2m1 = run[q2m1_src].as_single_value()
    q3m1 = run[q3m1_src].as_single_value()
    q4m1 = run[q4m1_src].as_single_value()

    q1_x = -542 + 0 * q1m1
    q1_y = 660 + q1m1 / (-0.2)

    q2_x = -608 + 0 * q2m1
    q2_y = -35 + q2m1 / 0.2

    q3_x = 534 + 0 * q3m1
    q3_y = -221 + q3m1 / 0.2

    q4_x = 588 + 0 * q4m1
    q4_y = 474 + q4m1 / (-0.2)

    geom = AGIPD_1MGeometry.from_quad_positions([
        (q1_x, q1_y),
        (q2_x, q2_y),
        (q3_x, q3_y),
        (q4_x, q4_y)
    ])

    # Find number of pulses
    # litfrm_src = ("MID_EXP_AGIPD1M1/REDU/LITFRM", )
    # run[litfrm_src[0], "dataFrames"].ndarray()[0]
    pulses = [1, 2]

    # Load the first train to find a ROI
    agipd = AGIPD1M(run).select_trains(np.s_[:1])
    train = agipd.get_array("image.data", pulses=pulses).data.squeeze()

    # Smooth the image to reduce noise
    train_mean = np.mean(train, axis=1)
    smoothed = np.empty_like(train_mean)
    for i in range(train_mean.shape[0]):
        smoothed[i] = gaussian_filter(train_mean[i], sigma=10)

    image, image_center = geom.position_modules(smoothed)

    # Take profiles to find the peak center
    x_lineout = np.nanmean(image, axis=0)
    y_lineout = np.nanmean(image, axis=1)

    # Get the array coordinates of the ROI
    peak_x = np.nanargmax(x_lineout)
    peak_y = np.nanargmax(y_lineout)

    roi_size = 200
    roi_half_size = roi_size // 2
    roi_xmin = peak_x - roi_half_size
    roi_xmax = peak_x + roi_half_size
    roi_ymin = peak_y - roi_half_size
    roi_ymax = peak_y + roi_half_size

    # Convert to physical coordinates
    roi_phys_xmin = (roi_xmin - image_center[1]) * geom.pixel_size
    roi_phys_xmax = (roi_xmax - image_center[1]) * geom.pixel_size
    roi_phys_ymin = (roi_ymin - image_center[0]) * geom.pixel_size
    roi_phys_ymax = (roi_ymax - image_center[0]) * geom.pixel_size

    # Create mask
    pixpos = geom.get_pixel_positions()
    px, py, pz = np.moveaxis(pixpos, -1, 0)
    roi_mask = (roi_phys_xmin < px) & (px < roi_phys_xmax) & (roi_phys_ymin < py) & (py < roi_phys_ymax)
    roi_indices = np.flatnonzero(roi_mask)

    pasha.set_default_context("processes", num_workers=os.cpu_count() // 2)
    intensities = pasha.alloc((len(run.train_ids),), dtype=np.float32)

    def compute_intensity(worker_id, index, train_id, data):
        agipd_train = stack_detector_data(data, "image.data")[pulses]
        agipd_mean = np.mean(agipd_train, axis=0)
        roi = agipd_mean.flat[roi_indices]
        intensities[index] = np.sum(roi)

    run_selection = run.select([("*/DET/*", "image.data")])
    pasha.map(compute_intensity, run_selection)

    return xr.DataArray(intensities, dims=("trainId",), coords={"trainId": run.train_ids})
