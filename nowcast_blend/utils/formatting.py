import os
from datetime import datetime
import numpy.typing as npt
import numpy as np
import xarray as xr

from nowcast_blend.preprocess.preprocess_radar import convert_input_to_xarray_dataset


def convert_npy_to_nc_file(path_blend, path_nowcast,  metadata_blend, metadata_nowcast, path_nwp=None, metadata_nwp=None):

    blended_forecast = convert_input_to_xarray_dataset(
        precip=np.load(path_blend),
        quality=None,
        metadata=metadata_blend,
        startdate=metadata_blend["timestamps"][0],
        timestep=10,
    )
    radar_nowcast = convert_input_to_xarray_dataset(
        precip=np.load(path_nowcast),
        quality=None,
        metadata=metadata_nowcast,
        startdate=metadata_nowcast["timestamps"][0],
        timestep=10,
    )

    radar_nowcast.precip_intensity.attrs["transform"] = "No"
    blended_forecast.precip_intensity.attrs["transform"] = "No"

    blended_forecast.to_netcdf(path_blend[:-3] + 'nc')
    radar_nowcast.to_netcdf(path_nowcast[:-3] + 'nc')
    
    if path_nwp!= None:
        nwp_forecast = convert_input_to_xarray_dataset(
            precip=np.load(path_nwp),
            quality=None,
            metadata=metadata_nwp,
            startdate=metadata_nwp["timestamps"][0],
            timestep=10,
        )
        nwp_forecast.precip_intensity.attrs["transform"] = "No"
        nwp_forecast.to_netcdf(blended_forecast[:-3] + '.nc')