# -*- coding: utf-8 -*-
"""
Blended forecast
====================

This tutorial shows how to construct a blended forecast from an ensemble nowcast
using the STEPS approach and a Numerical Weather Prediction (NWP) rainfall
forecast. The used datasets are from the Bureau of Meteorology, Australia.
"""

import time


import os
from datetime import datetime
import numpy.typing as npt
import cv2
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt


import pysteps

from pysteps import io, rcparams, blending, motion
from pysteps.visualization import plot_precip_field

from pysteps import utils
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.cascade import decomposition
from pysteps.utils import conversion, transformation, reproject_grids
from pysteps.downscaling import rainfarm
from pysteps.blending import steps
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata
import requests
import sys

sys.path.insert(0, "/home/joep/git/wi-research/p111_ecmwf_destine/")


import dgmr_for_blending

import download_destinE_data

from datetime import datetime, timedelta
import shutil
import warnings
import pyproj


# load functions
def cf_parameters_from_unit(unit: str) -> tuple[str, dict[str, str | None]]:
    if unit == "mm/h":
        var_name = "precip_intensity"
        var_standard_name = "instantaneous_precipitation_rate"
        var_long_name = "instantaneous precipitation rate"
        var_unit = "mm/h"
    elif unit == "mm":
        var_name = "precip_accum"
        var_standard_name = "accumulated_precipitation"
        var_long_name = "accumulated precipitation"
        var_unit = "mm"
    elif unit == "dBZ":
        var_name = "reflectivity"
        var_long_name = "equivalent reflectivity factor"
        var_standard_name = "equivalent_reflectivity_factor"
        var_unit = "dBZ"
    else:
        raise ValueError(f"unknown unit {unit}")

    return var_name, {
        "standard_name": var_standard_name,
        "long_name": var_long_name,
        "units": var_unit,
    }


def _convert_proj4_to_grid_mapping(proj4str):
    tokens = proj4str.split("+")

    d = {}
    for t in tokens[1:]:
        t = t.split("=")
        if len(t) > 1:
            d[t[0]] = t[1].strip()

    params = {}
    # TODO(exporters): implement more projection types here
    if d["proj"] == "stere":
        grid_mapping_var_name = "polar_stereographic"
        grid_mapping_name = "polar_stereographic"
        v = d["lon_0"] if d["lon_0"][-1] not in ["E", "W"] else d["lon_0"][:-1]
        params["straight_vertical_longitude_from_pole"] = float(v)
        v = d["lat_0"] if d["lat_0"][-1] not in ["N", "S"] else d["lat_0"][:-1]
        params["latitude_of_projection_origin"] = float(v)
        if "lat_ts" in list(d.keys()):
            params["standard_parallel"] = float(d["lat_ts"])
        elif "k_0" in list(d.keys()):
            params["scale_factor_at_projection_origin"] = float(d["k_0"])
        params["false_easting"] = float(d["x_0"])
        params["false_northing"] = float(d["y_0"])
    elif d["proj"] == "aea":  # Albers Conical Equal Area
        grid_mapping_var_name = "proj"
        grid_mapping_name = "albers_conical_equal_area"
        params["false_easting"] = float(d["x_0"]) if "x_0" in d else float(0)
        params["false_northing"] = float(d["y_0"]) if "y_0" in d else float(0)
        v = d["lon_0"] if "lon_0" in d else float(0)
        params["longitude_of_central_meridian"] = float(v)
        v = d["lat_0"] if "lat_0" in d else float(0)
        params["latitude_of_projection_origin"] = float(v)
        v1 = d["lat_1"] if "lat_1" in d else float(0)
        v2 = d["lat_2"] if "lat_2" in d else float(0)
        params["standard_parallel"] = (float(v1), float(v2))
    else:
        print("unknown projection", d["proj"])
        return None, None, None

    return grid_mapping_var_name, grid_mapping_name, params


def compute_lat_lon(
    x_r: npt.ArrayLike, y_r: npt.ArrayLike, projection: str
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    x_2d, y_2d = np.meshgrid(x_r, y_r)
    pr = pyproj.Proj(projection)
    lon, lat = pr(x_2d.flatten(), y_2d.flatten(), inverse=True)
    return lat.reshape(x_2d.shape), lon.reshape(x_2d.shape)


# Here y_r is being flipped in the original code?? why??
def convert_input_to_xarray_dataset(
    precip: np.ndarray,
    quality: np.ndarray | None,
    metadata: dict[str, str | float | None],
    startdate: datetime | None = None,
    timestep: int | None = None,
) -> xr.Dataset:
    """
    Read a precip, quality, metadata tuple as returned by the importers
    (:py:mod:`pysteps.io.importers`) and return an xarray dataset containing
    this data.

    Parameters
    ----------
    precip: array
        ND array containing imported precipitation data.
    quality: array, None
        ND array containing the quality values of the imported precipitation
        data, can be None.
    metadata: dict
        Metadata dictionary containing the attributes described in the
        documentation of :py:mod:`pysteps.io.importers`.
    startdate: datetime, None
        Datetime object containing the start date and time for the nowcast
    timestep: int, None
        The timestep in seconds between 2 consecutive fields, mandatory if
        the precip has 3 or more dimensions

    Returns
    -------
    out: Dataset
        A CF compliant xarray dataset, which contains all data and metadata.

    """
    var_name, attrs = cf_parameters_from_unit(metadata["unit"])

    dims = None
    timesteps = None
    ens_number = None

    if precip.ndim == 4:
        ens_number, timesteps, h, w = precip.shape
        dims = ["ens_number", "time", "y", "x"]

        if startdate is None:
            raise Exception("startdate missing")
        if timestep is None:
            raise Exception("timestep missing")

    elif precip.ndim == 3:
        timesteps, h, w = precip.shape
        dims = ["time", "y", "x"]

        if startdate is None:
            raise Exception("startdate missing")
        if timestep is None:
            raise Exception("timestep missing")

    elif precip.ndim == 2:
        h, w = precip.shape
        dims = ["y", "x"]
    else:
        raise Exception(f"Precip field shape: {precip.shape} not supported")

    x_r = np.linspace(metadata["x1"], metadata["x2"], w + 1)[:-1]
    x_r += 0.5 * (x_r[1] - x_r[0])
    y_r = np.linspace(metadata["y1"], metadata["y2"], h + 1)[:-1]
    y_r += 0.5 * (y_r[1] - y_r[0])

    if "xpixelsize" in metadata:
        xpixelsize = metadata["xpixelsize"]
    else:
        xpixelsize = x_r[1] - x_r[0]

    if "ypixelsize" in metadata:
        ypixelsize = metadata["ypixelsize"]
    else:
        ypixelsize = y_r[1] - y_r[0]

    if x_r[1] - x_r[0] != xpixelsize:
        # XR: This should be an error, but the importers don't always provide correct pixelsizes
        warnings.warn(
            "xpixelsize does not match x1, x2 and array shape, using xpixelsize for pixel size"
        )
    if y_r[1] - y_r[0] != ypixelsize:
        # XR: This should be an error, but the importers don't always provide correct pixelsizes
        warnings.warn(
            "ypixelsize does not match y1, y2 and array shape, using ypixelsize for pixel size"
        )

    # flip yr vector if yorigin is upper
    if metadata["yorigin"] == "upper":
        # y_r = np.flip(y_r)
        b = 1

    lat, lon = compute_lat_lon(x_r, y_r, metadata["projection"])

    (
        grid_mapping_var_name,
        grid_mapping_name,
        grid_mapping_params,
    ) = _convert_proj4_to_grid_mapping(metadata["projection"])

    data_vars = {
        var_name: (
            dims,
            precip,
            {
                "units": attrs["units"],
                "standard_name": attrs["standard_name"],
                "long_name": attrs["long_name"],
                "grid_mapping": "projection",
            },
        )
    }

    # XR: accutime vs timestep, what should be optional and what required?
    optional_metadata_keys = ["transform", "accutime", "zr_a", "zr_b"]

    required_metadata_keys = ["threshold", "zerovalue"]

    for metadata_field in optional_metadata_keys:
        if metadata_field in metadata:
            data_vars[var_name][2][metadata_field] = metadata[metadata_field]

    for metadata_field in required_metadata_keys:
        data_vars[var_name][2][metadata_field] = metadata[metadata_field]

    if quality is not None:
        data_vars["quality"] = (
            dims,
            quality,
            {
                "units": "1",
                "standard_name": "quality_flag",
                "grid_mapping": "projection",
            },
        )
    coords = {
        "y": (
            ["y"],
            y_r,
            {
                "axis": "Y",
                "long_name": "y-coordinate in Cartesian system",
                "standard_name": "projection_y_coordinate",
                "units": metadata["cartesian_unit"],
                "stepsize": ypixelsize,
            },
        ),
        "x": (
            ["x"],
            x_r,
            {
                "axis": "X",
                "long_name": "x-coordinate in Cartesian system",
                "standard_name": "projection_x_coordinate",
                "units": metadata["cartesian_unit"],
                "stepsize": xpixelsize,
            },
        ),
        "lon": (
            ["y", "x"],
            lon,
            {
                "long_name": "longitude coordinate",
                "standard_name": "longitude",
                "units": "degrees_east",
            },
        ),
        "lat": (
            ["y", "x"],
            lat,
            {
                "long_name": "latitude coordinate",
                "standard_name": "latitude",
                "units": "degrees_north",
            },
        ),
    }

    if ens_number is not None:
        coords["ens_number"] = (
            ["ens_number"],
            list(range(1, ens_number + 1, 1)),
            {
                "long_name": "ensemble member",
                "standard_name": "realization",
                "units": "",
            },
        )

    if timesteps is not None:
        startdate_str = datetime.strftime(startdate, "%Y-%m-%d %H:%M:%S")

        coords["time"] = (
            ["time"],
            [
                startdate + timedelta(seconds=float(second))
                for second in np.arange(timesteps) * timestep
            ],
            {"long_name": "forecast time", "stepsize": timestep},
            {"units": "seconds since %s" % startdate_str},
        )
    if grid_mapping_var_name is not None:
        coords[grid_mapping_name] = (
            [],
            None,
            {"grid_mapping_name": grid_mapping_name, **grid_mapping_params},
        )
    attrs = {
        "Conventions": "CF-1.7",
        "institution": metadata["institution"],
        "projection": metadata["projection"],
        "precip_var": var_name,
    }
    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    return dataset.sortby(dims)


def download_radar_knmi(gauge_adjusted, last_hour, date, input_dir):
    if gauge_adjusted:
        url = "https://api.dataplatform.knmi.nl/open-data/v1/datasets/nl_rdr_data_rtcor_5m/versions/1.0/files"
        lastfile = last_hour.strftime("RAD_NL25_RAC_RT_%Y%m%d%H%M.h5")
    else:
        url = "https://api.dataplatform.knmi.nl/open-data/datasets/radar_reflectivity_composites/versions/2.0/files"
        lastfile = last_hour.strftime("RAD_NL25_PCP_NA_%Y%m%d%H%M.h5")

    api_key = "5e554e19274a9600012a3eb10174be35b75442a7a5e2ba066642a279"

    file_list = requests.get(
        url,
        headers={"Authorization": api_key},
        params={"startAfterFilename": lastfile, "maxKeys": 12},
    )

    file_list = file_list.json().get("files")

    # Download the last 3 available files
    for ii in range(len(file_list) - 4, len(file_list)):
        fn = file_list[ii]["filename"]
        print(fn)

        yr = fn[16:20]
        mnth = fn[20:22]
        day = fn[22:24]
        hour = fn[24:26]
        minute = fn[26:28]

        local_folder_today = input_dir + "/{}/{}/{}/".format(yr, mnth, day)

        for folder in [local_folder_today]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        direc = local_folder_today

        if not os.path.exists(direc + fn):

            get_file_response = requests.get(
                url + "/" + fn + "/url", headers={"Authorization": api_key}
            )

            download_url = get_file_response.json().get("temporaryDownloadUrl")

            dataset_file = requests.get(download_url, stream=True)

            if dataset_file.status_code == 200:
                with open(direc + fn, "wb") as f:
                    dataset_file.raw.decode_content = True
                    shutil.copyfileobj(dataset_file.raw, f)
    if gauge_adjusted:
        fns = io.find_by_date(
            date,
            input_dir,
            "%Y/%m/%d",
            "RAD_NL25_RAC_RT_%Y%m%d%H%M",
            "h5",
            5,
            num_prev_files=3,
        )
    else:
        fns = io.find_by_date(
            date,
            input_dir,
            "%Y/%m/%d",
            "RAD_NL25_PCP_NA_%Y%m%d%H%M",
            "h5",
            5,
            num_prev_files=3,
        )

        assert (
            len(fns[0]) == 4
        ), f"fns does not contain enough radar images for DGMR (needs 4, contains {len(fns[0])})"
    return fns


def cdo_to_netcdf(
    date,
    destinE_data,
    destinE_data_cut,
    numerical_data,
    destineE_datafolder,
    filename,
    freq,
    historical_destine,
):
    if historical_destine != True:
        lat_min, lat_max = (
            destinE_data_cut.lat.values.min(),
            destinE_data_cut.lat.values.max(),
        )
        lon_min, lon_max = (
            destinE_data_cut.lon.values.min(),
            destinE_data_cut.lon.values.max(),
        )
        new_times = pd.date_range(
            destinE_data_cut.time.values[0], destinE_data_cut.time.values[-1], freq=freq
        )
    else:
        lat_min, lat_max = (
            destinE_data_cut.latitude.values.min(),
            destinE_data_cut.latitude.values.max(),
        )
        lon_min, lon_max = (
            destinE_data_cut.longitude.values.min(),
            destinE_data_cut.longitude.values.max(),
        )
        new_times = pd.date_range(
            destinE_data_cut.step.values[0], destinE_data_cut.step.values[-1], freq=freq
        )

    # Interpolate to new time grid
    lat_new = np.linspace(lat_min, lat_max, len(numerical_data[0]))
    lon_new = np.linspace(lon_min, lon_max, len(numerical_data[0][0]))

    destinE_data_radar_scale_xr = xr.DataArray(
        numerical_data,
        dims=("time", "lat", "lon"),
        coords={
            "time": new_times,
            "lat": lat_new,
            "lon": lon_new,
        },
        name=list(destinE_data.data_vars)[-1],  # reuse variable name
    )

    # Copy variable attributes
    destinE_data_radar_scale_xr.attrs = destinE_data[
        list(destinE_data.data_vars)[0]
    ].attrs

    yr = date.year
    mnth = date.month
    date_str = date.strftime("%Y%m%d")

    # Make sure local folder exists
    local_folder_today = destineE_datafolder + "{}/{}/".format(yr, mnth)
    for folder in [local_folder_today]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Wrap in dataset and copy global attrs
    ds_destinE_data_radar_scale_xr = xr.Dataset(
        {destinE_data_radar_scale_xr.name: destinE_data_radar_scale_xr}
    )
    ds_destinE_data_radar_scale_xr.attrs = destinE_data.attrs

    ds_destinE_data_radar_scale_xr["lat"].attrs["units"] = "degrees_north"
    ds_destinE_data_radar_scale_xr["lon"].attrs["units"] = "degrees_east"

    ds_destinE_data_radar_scale_xr.to_netcdf(filename)


def advection_correction_backward(R, T=5, t=1):
    """
    R = np.array([qpe_previous, qpe_current])
    T = time between two observations (5 min)
    t = interpolation timestep (1 min)
    """

    # Evaluate advection
    oflow_method = motion.get_method("LK")
    fd_kwargs = {"buffer_mask": 10}  # avoid edge effects
    V = oflow_method(np.log(R), fd_kwargs=fd_kwargs)

    # Perform temporal interpolation
    x, y = np.meshgrid(
        np.arange(R[0].shape[1], dtype=float), np.arange(R[0].shape[0], dtype=float)
    )
    ny, nx = R[0].shape
    n_steps = T // t
    sequence = np.zeros((n_steps, ny, nx))

    for idx, i in enumerate(range(t, T + t, t)):
        # pos1 = (y - i / T * V[1], x - i / T * V[0])
        # R1 = map_coordinates(R[0], pos1, order=1)

        pos2 = (y + (T - i) / T * V[1], x + (T - i) / T * V[0])
        R2 = map_coordinates(R[1], pos2, order=1)

        # Blend fields ?? check this?
        sequence[idx, :, :] = R2
        # Rd += (T - i) * R1 + i * R2
    return sequence


def advection_correction(R, T=5, t=1):
    """
    R = np.array([qpe_previous, qpe_current])
    T = time between two observations (5 min)
    t = interpolation timestep (1 min)
    """

    # Evaluate advection
    oflow_method = motion.get_method("LK")
    fd_kwargs = {"buffer_mask": 10}  # avoid edge effects
    V = oflow_method(np.log(R), fd_kwargs=fd_kwargs)

    # Perform temporal interpolation
    Rd = np.zeros((R[0].shape))
    x, y = np.meshgrid(
        np.arange(R[0].shape[1], dtype=float), np.arange(R[0].shape[0], dtype=float)
    )
    ny, nx = R[0].shape
    n_steps = T // t
    sequence = np.zeros((n_steps, ny, nx))

    for idx, i in enumerate(range(t, T + t, t)):
        pos1 = (y - i / T * V[1], x - i / T * V[0])
        R1 = map_coordinates(R[0], pos1, order=1)

        pos2 = (y + (T - i) / T * V[1], x + (T - i) / T * V[0])
        R2 = map_coordinates(R[1], pos2, order=1)

        # Blend fields ?? check this?
        sequence[idx, :, :] = (1 - i / T) * R1 + i / T * R2
        # Rd += (T - i) * R1 + i * R2
    return sequence
    # return t / T**2 * Rd


###############################################
# LOAD KNMI RADAR DATA
###############################################


def run_blending_operational(
    date,
    historical_destine,
    knmi_input_dir,
    destineE_datafolder,
    timesteps,
    timestep_interval,
    n_ens_members,
    n_ens_members_dgmr,
    weights_method,
    custom_weights=None,
    return_weights=False,
    re_do_blending=False,
):

    start_time = time.time()
    gauge_adjusted = True
    # If before 18 Nov. 2024 do more fileterin?

    # inset a date and time (in utc)
    last_hour = date + timedelta(hours=-1)

    def round_to_5min(dt):
        minutes = dt.minute
        rounded = int(round(minutes / 5.0) * 5)
        diff = rounded - minutes
        return (dt + timedelta(minutes=diff)).replace(second=0, microsecond=0)

    # date_5min = round_to_5min(date) - timedelta(minutes=5) #round to 5 minutes, then substract 5 minutes so that DGMR is initialised on the hour exactly
    date_5min = round_to_5min(
        date
    )  # Currently running DGMR on 5 past the hour, but including last radar image -> gives 6hours +5 minutes which is needed for blending

    last_hour_5min = round_to_5min(
        last_hour
    )  # - timedelta(minutes=5) see reason above for not using this

    # check if data exists, otherwise download
    fns = None
    try:
        if gauge_adjusted:
            fns = io.find_by_date(
                date_5min,
                knmi_input_dir,
                "%Y/%m/%d",
                "RAD_NL25_RAC_RT_%Y%m%d%H%M",
                "h5",
                5,
                num_prev_files=3,
            )
            assert (
                len(fns[0]) == 4
            ), f"fns does not contain enough radar images for DGMR (needs 4, contains {len(fns[0])})"

        else:
            fns = io.find_by_date(
                date_5min,
                knmi_input_dir,
                "%Y/%m/%d",
                "RAD_NL25_PCP_NA_%Y%m%d%H%M",
                "h5",
                5,
                num_prev_files=3,
            )
            assert (
                len(fns[0]) == 4
            ), f"fns does not contain enough radar images for DGMR (needs 4, contains {len(fns[0])})"

    except:
        fns = download_radar_knmi(
            gauge_adjusted, last_hour_5min, date_5min, knmi_input_dir
        )

    # start to unpack the radar files
    # load Radar files
    importer_kwargs = {"accutime": 5, "qty": "ACRR", "pixelsize": 1000.0}

    # Read the data from the archive
    try:
        importer = io.get_method("knmi_hdf5", "importer")
        R, _, metadata_radar = io.read_timeseries(fns, importer, **importer_kwargs)
    except:
        print("Input data unreadable. Abort script.")

    # Convert to rain rate
    R, metadata_radar = conversion.to_rainrate(R, metadata_radar)
    del metadata_radar["transform"]
    R[np.isnan(R)] = 0

    import warnings
    import pyproj

    R_xr = convert_input_to_xarray_dataset(R, None, metadata_radar, fns[1][0], 5 * 60)
    # R_xr.to_netcdf(f"{input_dir}knmi_radar_template.nc")

    ###############################################
    # LOAD DESTINE DATA
    ###############################################
    # in case cdo_to netcdf is not run
    yr = date.year
    mnth = date.month
    date_str = date.strftime("%Y%m%d%H")
    print(f"Running blend for:{date_str}")

    # Make sure local folder exists
    local_folder_today = destineE_datafolder + "{}/{}/".format(yr, mnth)

    if historical_destine == True:
        param = "218.228-219.228-228.128"
        extention = "grib"
    else:
        param = "228_regrid_nl"
        extention = "nc"

    try:
        files = io.find_by_date(
            date,
            destineE_datafolder,
            "%Y/%m/",
            f"DestinE_ExtremesDT_%Y%m%d_{param}",
            extention,
            1,
            num_prev_files=0,
        )[0][0]
    # Download DestinE data
    except:
        files = download_destinE_data.download_destine(date, destineE_datafolder)
        param = "228_regrid_nl"
        extention = "nc"

    try:
        # files_pre_processed = io.find_by_date(
        #         date, destineE_datafolder, "%Y/%m", f'DestinE_ExtremesDT_%Y%m%d_{param}_hres_interp_nlgrid_{timestep_interval}_{timesteps}', "nc", 1, num_prev_files=0
        #     )
        # destinE_nlgrid = xr.open_dataset(files_pre_processed[0][0])
        destinE_nlgrid = xr.open_dataset(
            local_folder_today
            + f"/DestinE_ExtremesDT_{date_str}_{param}_hres_interp_nlgrid_{timestep_interval}_{timesteps}.nc"
        )

        # slice the timesteps to match the radar timesteps
        time_slice = slice(
            pd.to_datetime(R_xr["time"][-1].values) + timedelta(minutes=5),
            pd.to_datetime(R_xr["time"][-1].values)
            + timedelta(minutes=timestep_interval * (timesteps + 1)),
        )

        destinE_nlgrid_blend = destinE_nlgrid.sel(time=time_slice)

        print(f"destinE_nlgrid_blend time range: {time_slice}")

        len_nwp = len(destinE_nlgrid_blend["time"])
        assert len_nwp == (
            timesteps + 1
        ), f"Not the correct length timesteps in destine file, length is currently: {len_nwp} while it should be {(timesteps + 1)} "

    # Mkae two pre-processing scripts that work with 2023 and 2025 data
    except:
        print("Pre-processing Destine file:", files)
        destinE_data = xr.open_dataset(files)
        destinE_data_np = destinE_data.tp.values

        destinE_data["tp"].attrs = {
            "long_name": "Accumulated precipitation",
            "units": "mm/h",
            "param": "193.1.0",
        }

        accum_prcp = destinE_data["tp"] * 1000

        try:
            accum_prcp_subset = accum_prcp.sel(
                latitude=slice(56.4, 48.4), longitude=slice(-1, 11.87)
            )
            accum_prcp_subset = accum_prcp_subset.assign_coords(
                step=[
                    pd.to_datetime(destinE_data["time"].values) + timedelta(hours=i)
                    for i in range(len(destinE_data["step"]))
                ]
            )
            precipitation = accum_prcp_subset - accum_prcp_subset.shift({"step": 1})
            precipitation = precipitation.dropna("step", how="all")

        except:
            # accum_prcp_subset = accum_prcp.sel(lat=slice(56.4, 48.4),lon=slice(-1, 11.87))
            precipitation = accum_prcp - accum_prcp.shift({"time": 1})
            precipitation = precipitation.dropna("time", how="all")

            # precipitation_0 = accum_prcp_subset[0]
            # precipitation_0 = precipitation_0.expand_dims('time')

            # precipitation_non0 = accum_prcp_subset - accum_prcp_subset.shift({'time': 1})
            # precipitation_non0 = precipitation_non0.dropna('time', how="all")
            # precipitation = xr.concat([precipitation_0, precipitation_non0], dim='time')

            # DOWNSCALE DESTINE DATA WITH RAINFARM TO RADAR RESOLUTION
        destinE_data_radar_scale = []
        for i in range(precipitation.shape[0]):
            destinE_data_radar_scale.append(
                rainfarm.downscale(
                    precipitation[i], ds_factor=4, kernel_type="gaussian"
                )
            )

        destinE_data_radar_scale = np.array(destinE_data_radar_scale)

        # TODO: APPLY ADVECTION CORRECTION TO DESTINE DATA -> discuss with kyrie -> use something else?
        all_steps = [destinE_data_radar_scale[0:1]]  # keep first slice as (1, ny, nx)

        for i in range(destinE_data_radar_scale.shape[0] - 1):
            steps = advection_correction_backward(
                destinE_data_radar_scale[i : i + 2], T=60, t=timestep_interval
            )
            all_steps.append(steps)
        # Concatenate along time
        destinE_nlgrid_hres_advected = np.concatenate(all_steps, axis=0)

        # Write to netcdf so cdo can use the data
        cdo_to_netcdf(
            date,
            destinE_data,
            precipitation,
            destinE_nlgrid_hres_advected,
            destineE_datafolder,
            local_folder_today
            + f"DestinE_ExtremesDT_{date_str}_{param}_hres_advect_xr_{timestep_interval}_{timesteps}.nc",
            f"{timestep_interval}min",
            historical_destine,
        )

        # REGRID DESTINE DATA TO KNMI RADAR GRID
        import cdo
        from cdo import Cdo

        cdo = Cdo()
        cdo.remapnn(
            f"{knmi_input_dir}knmi_grid.txt",  # target grid
            input=local_folder_today
            + f"DestinE_ExtremesDT_{date_str}_{param}_hres_advect_xr_{timestep_interval}_{timesteps}.nc",  # source file
            output=local_folder_today
            + f"DestinE_ExtremesDT_{date_str}_{param}_hres_interp_nlgrid_{timestep_interval}_{timesteps}.nc",  # output file
        )

        # Open created grid
        destinE_nlgrid = xr.open_dataset(
            local_folder_today
            + f"DestinE_ExtremesDT_{date_str}_{param}_hres_interp_nlgrid_{timestep_interval}_{timesteps}.nc"
        )

        time_slice = slice(
            pd.to_datetime(R_xr["time"][-1].values) + timedelta(minutes=5),
            pd.to_datetime(R_xr["time"][-1].values)
            + timedelta(minutes=timestep_interval * (timesteps + 1)),
        )

        # slice the timesteps to match the radar timesteps
        destinE_nlgrid_blend = destinE_nlgrid.sel(time=time_slice)

        print(f"destinE_nlgrid_blend time range: {time_slice}")

        len_nwp = len(destinE_nlgrid_blend["time"])
        assert len_nwp == (
            timesteps + 1
        ), f"Not the correct length timesteps in destine file, length is currently: {len_nwp} while it should be {(timesteps + 1)} "

    destinE_nlgrid_blend_metadata = metadata_radar
    destinE_nlgrid_blend_metadata["timestamps"] = destinE_nlgrid_blend.time.values
    destinE_nlgrid_blend_metadata["institution"] = destinE_nlgrid_blend.institution
    destinE_nlgrid_blend_metadata["unit"] = "mm/h"
    destinE_nlgrid_blend_metadata["threshold"] = float(0.1)

    # TODO no option to change the timlength yet for DGMR (now only works if it accounts to 6 hours)
    path_DGRMR = (
        local_folder_today
        + f"/DGMR_{date_str}_step_min_{timestep_interval}_len_{timesteps}_ens_{n_ens_members_dgmr}.npy"
    )
    if not os.path.exists(path_DGRMR):
        DGMR_det_long = dgmr_for_blending.run_dgmr_ensemble(
            R_xr.precip_intensity.values, ens_members=n_ens_members_dgmr
        )
        np.save(path_DGRMR, DGMR_det_long, allow_pickle=True)
    else:
        DGMR_det_long = np.load(path_DGRMR, allow_pickle=True)

    # Select only the relevant files
    if timestep_interval != 5:
        step = timestep_interval // 5
        DGMR_det = DGMR_det_long[:, ::step]
    else:
        DGMR_det = DGMR_det_long

    assert len(DGMR_det[0]) == (
        timesteps + 1
    ), "length of DGMR output is not the same as the timesteps value!"
    ############FINISHED TILL HERE

    ###############################################################################
    # Load the data from the archive
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    metadata_radar["transform"] = None

    # Log-transform the data
    metadata_radar["timestamps"] = destinE_nlgrid_blend_metadata["timestamps"]
    DGMR_det_db, metadata_radar_db = transformation.dB_transform(
        DGMR_det, metadata_radar, threshold=0.1, zerovalue=-15.0
    )

    if DGMR_det_db.ndim == 3:
        DGMR_det_db = DGMR_det_db[None, :]

    converter = pysteps.utils.get_method("mm/h")
    radar_precip, metadata_radar = converter(
        R_xr.precip_intensity.values, metadata_radar
    )
    destinE_nlgrid_blend_val, destinE_nlgrid_blend_metadata = converter(
        destinE_nlgrid_blend.tp.values, destinE_nlgrid_blend_metadata
    )

    # Threshold the data
    radar_precip[radar_precip < 0.1] = 0.0
    destinE_nlgrid_blend_val[destinE_nlgrid_blend_val < 0.1] = 0.0

    # transform the data to dB
    transformer = pysteps.utils.get_method("dB")
    radar_precip, radar_metadata = transformer(
        radar_precip, metadata_radar, threshold=0.1
    )
    nwp_precip, nwp_metadata = transformer(
        destinE_nlgrid_blend_val, destinE_nlgrid_blend_metadata, threshold=0.1
    )

    # r_nwp has to be four dimentional (n_models, time, y, x).
    # If we only use one model:
    if nwp_precip.ndim == 3:
        nwp_precip = nwp_precip[None, :]

        ###############################################################################
        # For the initial time step (t=0), the NWP rainfall forecast is not that different
        # from the observed radar rainfall, but it misses some of the locations and
        # shapes of the observed rainfall fields. Therefore, the NWP rainfall forecast will
        # initially get a low weight in the blending process.
        #
        # Determine the velocity fields
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    path_blend = (
        local_folder_today
        + f"/Blended_forecast_{date_str}_step_min_{timestep_interval}_len_{timesteps}_ens_dgmr_{n_ens_members_dgmr}_ens_{n_ens_members}.npy"
    )
    if not os.path.exists(path_blend) or re_do_blending is True:
        oflow_method = pysteps.motion.get_method("lucaskanade")

        # First for the radar images
        velocity_radar = oflow_method(radar_precip)

        # Then for the NWP forecast
        velocity_nwp = []
        # Loop through the models
        for n_model in range(nwp_precip.shape[0]):
            # Loop through the timesteps. We need two images to construct a motion
            # field, so we can start from timestep 1. Timestep 0 will be the same
            # as timestep 1.
            _v_nwp_ = []
            for t in range(1, nwp_precip.shape[1]):
                v_nwp_ = oflow_method(nwp_precip[n_model, t - 1 : t + 1, :])
                _v_nwp_.append(v_nwp_)
                v_nwp_ = None
            # Add the velocity field at time step 1 to time step 0.
            _v_nwp_ = np.insert(_v_nwp_, 0, _v_nwp_[0], axis=0)
            velocity_nwp.append(_v_nwp_)

        velocity_nwp = np.stack(velocity_nwp)

        ################################################################################
        # The blended forecast
        # --------------------
        # timestep_interval = 5
        try:
            precip_forecast_stacked = blending.steps.forecast(
                precip=radar_precip,
                precip_nowcast=DGMR_det_db,
                nowcasting_method="external_nowcast",
                mask_method=None,
                precip_models=nwp_precip,
                velocity=velocity_radar,
                velocity_models=velocity_nwp,
                timesteps=timesteps,
                timestep=timestep_interval,
                issuetime=pd.to_datetime(radar_metadata["timestamps"][-1]),
                n_ens_members=n_ens_members,
                # resample_distribution=False,
                precip_thr=radar_metadata["threshold"],
                kmperpixel=radar_metadata["xpixelsize"] / 1000.0,
                # noise_stddev_adj=None,
                # noise_method=None,
                weights_method=weights_method,
                custom_weights=custom_weights,
                return_weights=return_weights,
                probmatching_method="cdf",
                vel_pert_method=None,
            )
        except:  # If the blending fails, it is likely due to a an error with x= non-finite number in Gamma determination. Use climatological weights instead.
            print(
                "Error in blending with weights method:",
                weights_method,
                " - switching to custom climatological weights",
            )
            GAMMA = np.array(
                [
                    [0.99805, 0.9933],
                    [0.9925, 0.923],
                    [0.9776, 0.975],
                    [0.9297, 0.750],
                    [0.796, 0.367],
                    [0.482, 0.069],
                ]
            )
            regr_pars = np.array(
                [
                    [130.0, 165.0, 120.0, 55.0, 50.0, 15.0],
                    [155.0, 220.0, 200.0, 75.0, 10e4, 10e4],
                ]
            )
            clim_cor_values = np.array([0.848, 0.537, 0.237, 0.065, 0.02, 0.0044])
            custom_weights = {
                "GAMMA": GAMMA,
                "regr_pars": regr_pars,
                "clim_cor_values": clim_cor_values,
            }
            precip_forecast_stacked = blending.steps.forecast(
                precip=radar_precip,
                precip_nowcast=DGMR_det_db,
                nowcasting_method="external_nowcast",
                mask_method=None,
                precip_models=nwp_precip,
                velocity=velocity_radar,
                velocity_models=velocity_nwp,
                timesteps=timesteps,
                timestep=timestep_interval,
                issuetime=pd.to_datetime(radar_metadata["timestamps"][-1]),
                n_ens_members=n_ens_members,
                # resample_distribution=False,
                precip_thr=radar_metadata["threshold"],
                kmperpixel=radar_metadata["xpixelsize"] / 1000.0,
                # noise_stddev_adj=None,
                # noise_method=None,
                weights_method="custom",
                custom_weights=custom_weights,
                return_weights=return_weights,
                probmatching_method="cdf",
                vel_pert_method=None,
            )

        if return_weights:
            precip_forecast_stacked, weights = precip_forecast_stacked

        precip_forecast_mm, _ = converter(precip_forecast_stacked, radar_metadata)
        np.save(path_blend, precip_forecast_mm, allow_pickle=True)
    else:
        precip_forecast_mm = np.load(path_blend, allow_pickle=True)

    converter = pysteps.utils.get_method("mm/h")
    radar_precip_mm, _ = converter(DGMR_det_db, radar_metadata)
    nwp_precip_mm, _ = converter(nwp_precip, nwp_metadata)
    print((time.time() - start_time), "seconds")

    if return_weights:
        return precip_forecast_mm, radar_precip_mm, nwp_precip_mm, weights
    else:
        return precip_forecast_mm, radar_precip_mm, nwp_precip_mm


if __name__ == "__main__":
    # date = datetime.now()
    year = 2023
    month = 11
    day = 1
    hour = 5
    date = datetime(year, month, day, hour)
    timesteps = 36
    timestep_interval = 10
    n_ens_members = 5
    n_ens_members_dgmr = 1
    weights_method = "bps"
    gauge_adjusted = True
    if gauge_adjusted:
        knmi_input_dir = "/srv/data/nas/input_general/knmi_radar_gauge_adj/"
    else:
        knmi_input_dir = "/srv/data/nas/input_general/knmi_radar/"
    destineE_datafolder = "/srv/data/nas/project_data/p111_ecmwf_destine/"
    historical_destine = True
    precip_blended = run_blending_operational(
        date,
        historical_destine,
        knmi_input_dir,
        destineE_datafolder,
        timesteps,
        timestep_interval,
        n_ens_members,
        n_ens_members_dgmr,
        weights_method,
    )
