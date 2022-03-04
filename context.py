import numpy as np

from amore_mid_prototype.context import Variable

@Variable(summary='mean')
def xgm_intensity(run):
    return run['SA2_XTD1_XGM/XGM/DOOCS:output', 'data.intensityTD'].xarray(roi=(0,))


@Variable(summary='mean')
def popin_intensity(run):
    return run['MID_XTD6_IMGPI/PROC/BEAMVIEW', 'regionMean'].xarray()

@Variable()
def popin_region(run):
    return run[
        'MID_XTD6_IMGPI/PROC/BEAMVIEW', 'integrationRegion'
    ].as_single_value().astype(np.int32)

@Variable(summary='mean')
def popin_beam_x(run):
    return run['MID_XTD6_IMGPI/SPROC/BEAMVIEW', 'positionX'].xarray()

@Variable(summary='mean')
def popin_beam_y(run):
    return run['MID_XTD6_IMGPI/SPROC/BEAMVIEW', 'positionX'].xarray()

@Variable(summary='mean')
def diamond_des(run):
    return run['MID_EXP_FASTADC/ADC/DESTEST:channel_2.output', 'data.peaks'].xarray(roi=(0,))

@Variable(summary='mean')
def diamond_sample(run):
    return run['MID_EXP_FASTADC/ADC/DESTEST:channel_3.output', 'data.peaks'].xarray(roi=(0,))

@Variable(summary='mean')
def diamond_sdl_lower(run):
    return run['MID_EXP_FASTADC/ADC/DESTEST:channel_5.output', 'data.peaks'].xarray(roi=(0,))

@Variable(summary='mean')
def sample_x(run):
    return run['MID_EXP_UPP/MOTOR/R3', 'actualPosition'].xarray()

@Variable(summary='mean')
def sample_y(run):
    return run['MID_EXP_UPP/MOTOR/R1', 'actualPosition'].xarray()

@Variable(summary='mean')
def sample_z(run):
    return run['MID_EXP_UPP/MOTOR/R2', 'actualPosition'].xarray()
