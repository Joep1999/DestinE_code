import optuna
import numpy as np
from datetime import datetime
from pysteps import io, rcparams
from pysteps.utils import conversion
from datetime import datetime, timedelta
import requests
import os
import shutil
import numpy as np
import xarray as xr
import numpy.typing as npt
import pandas as pd
import time
import sys

sys.path.insert(0, "/home/joep/git/wi-research/p111_ecmwf_destine/")
import blending_operational
import warnings
import pyproj
import scoringrules as sr

# input /output directories
gauge_adjusted = True
if gauge_adjusted:
    knmi_input_dir = "/srv/data/nas/input_general/knmi_radar_gauge_adj/"
else:
    knmi_input_dir = "/srv/data/nas/input_general/knmi_radar/"

destineE_datafolder = "/srv/data/nas/project_data/p111_ecmwf_destine/"


def download_radar_knmi_check(
    gauge_adjusted, start_date, input_dir, time_interval, timesteps
):
    before_start_date = start_date + timedelta(minutes=-10)
    if gauge_adjusted:
        url = "https://api.dataplatform.knmi.nl/open-data/v1/datasets/nl_rdr_data_rtcor_5m/versions/1.0/files"
        startfile = before_start_date.strftime("RAD_NL25_RAC_RT_%Y%m%d%H%M.h5")
    else:
        url = "https://api.dataplatform.knmi.nl/open-data/datasets/radar_reflectivity_composites/versions/2.0/files"
        startfile = before_start_date.strftime("RAD_NL25_PCP_NA_%Y%m%d%H00.h5")

    api_key = "5e554e19274a9600012a3eb10174be35b75442a7a5e2ba066642a279"

    interval_factor = time_interval // 5

    file_list = requests.get(
        url,
        headers={"Authorization": api_key},
        params={
            "startAfterFilename": startfile,
            "maxKeys": int(interval_factor * (timesteps + 1)),
        },
    )  # add a buffer of 1 to be sure to have all the files, correct files are selected later

    file_list = file_list.json().get("files")

    # Parse timestamps from filenames
    def extract_datetime(filename):
        # Extract the 14-digit timestamp from the filename
        date_str = filename.split("_")[-1].split(".")[0]
        return datetime.strptime(date_str, "%Y%m%d%H%M")

    # Find the index of the dictionary with that timestamp
    index_start = next(
        (
            i
            for i, d in enumerate(file_list)
            if extract_datetime(d["filename"]) == start_date
        ),
        None,  # returns None if not found
    )

    end_date = start_date + timedelta(minutes=timesteps * time_interval)
    print("last timestep is: ", end_date)

    # Find the index of the dictionary with that timestamp
    index_end = next(
        (
            i
            for i, d in enumerate(file_list)
            if extract_datetime(d["filename"]) == end_date
        ),
        None,  # returns None if not found
    )

    file_list = file_list[index_start : index_end + 1]

    # Download the last 3 available files
    for ii in range(len(file_list)):
        fn = file_list[ii]["filename"]
        print("Downloading: ", fn)

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

    fns = io.find_by_date(
        start_date,
        input_dir,
        "%Y/%m/%d",
        "RAD_NL25_RAC_RT_%Y%m%d%H%M",
        "h5",
        time_interval,
        num_next_files=timesteps,
    )
    return fns


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


import numpy as np
from pysteps.verification import probscores, spatialscores


def skill_score(
    date,
    histrorical_destine,
    knmi_input_dir,
    destineE_datafolder,
    timesteps,
    timestep_interval,
    n_ens_members,
    n_ens_members_dgmr,
    weights_method,
    custom_weights=None,
    return_weights=True,
    re_do_blending=True,
):

    # check if radar images exist
    gauge_adjusted = True

    # def round_to_5min(dt):
    #     minutes = dt.minute
    #     rounded = int(round(minutes / 5.0) * 5)
    #     diff = rounded - minutes
    #     return (dt + timedelta(minutes=diff)).replace(second=0, microsecond=0)
    # date_5min = round_to_5min(date)
    # last_hour_5min = round_to_5min(last_hour)

    # check if data exists, otherwise download
    fns = None
    try:
        if gauge_adjusted:
            fns = io.find_by_date(
                date,
                knmi_input_dir,
                "%Y/%m/%d",
                "RAD_NL25_RAC_RT_%Y%m%d%H%M",
                "h5",
                timestep_interval,
                num_next_files=timesteps,
            )
        else:
            fns = io.find_by_date(
                date,
                knmi_input_dir,
                "%Y/%m/%d",
                "RAD_NL25_PCP_NA_%Y%m%d%H%M",
                "h5",
                timestep_interval,
                num_next_files=timesteps,
            )
    except:
        fns = download_radar_knmi_check(
            gauge_adjusted, date, knmi_input_dir, timestep_interval, timesteps
        )

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

    R_xr = convert_input_to_xarray_dataset(
        R, None, metadata_radar, fns[1][0], timestep_interval * 60
    )

    # TODO run this

    # Fix it so pre-processing does not always have to happen.
    blend_precip, nowcast_precip, nwp_precip, weights = (
        blending_operational.run_blending_operational(
            date,
            histrorical_destine,
            knmi_input_dir,
            destineE_datafolder,
            timesteps,
            timestep_interval,
            n_ens_members,
            n_ens_members_dgmr,
            weights_method=weights_method,
            custom_weights=custom_weights,
            return_weights=return_weights,
            re_do_blending=re_do_blending,
        )
    )

    return (
        blend_precip,
        nowcast_precip,
        nwp_precip,
        R_xr.precip_intensity.values[:-1],
        weights,
    )


# for full dataset
# years = [2023,2024,2025]
# months = np.arange(1,13,1)
# days = np.arange(1,32,1)
# hours = [6,12,18, 24, 30]

# for single case:
years = [2024]
months = [5]
days = [1]  # np.arange(1,32)
hours = [17]

# IF using data downloaded from ATOS, use TRUE
historical_destine = True

# Model settings -> currently only works for hour = 6 becasue of DGMR
hour_length = 6
timestep_interval = 20
timesteps = int(hour_length * (60 / timestep_interval))
n_ens_members = 5
n_ens_members_dgmr = 5
weights_method = "bps"

blend_precip_total = []
nowcast_precip_total = []
nwp_precip_total = []
radar_data = []
for yr in years:
    for mnth in months:
        for dy in days:
            for hr in hours:

                date = datetime(yr, mnth, dy, hr)
                date_str = date.strftime("%Y%m%d%H")
                # ============================================================
                # 1️⃣ Base parameters (climatological and dynamic)
                # ============================================================

                GAMMA_base = np.array(
                    [
                        [0.99805, 0.9933],
                        [0.9925, 0.923],
                        [0.9776, 0.975],
                        [0.9297, 0.750],
                        [0.796, 0.367],
                        [0.482, 0.069],
                    ]
                )

                regr_pars_base = np.array(
                    [
                        [130.0, 165.0, 120.0, 55.0, 50.0, 15.0],
                        [155.0, 220.0, 200.0, 75.0, 1e4, 1e4],
                    ]
                )

                clim_cor_values_base = np.array(
                    [0.848, 0.537, 0.237, 0.065, 0.02, 0.0044]
                )

                custom_weights = {
                    "GAMMA": GAMMA_base,
                    "regr_pars": regr_pars_base,
                    "clim_cor_values": clim_cor_values_base,
                }

                # Run model twice (climatological and dynamic)
                # blend_precip_clim, nowcast_precip, nwp_precip, radar_precip, weights_clim = skill_score(date,historical_destine, knmi_input_dir, destineE_datafolder, timesteps, timestep_interval, n_ens_members,n_ens_members_dgmr, weights_method = 'custom', custom_weights = custom_weights, return_weights = True, re_do_blending  =True)
                (
                    blend_precip_dynamic,
                    nowcast_precip,
                    nwp_precip,
                    radar_precip,
                    weights_dynamic,
                ) = skill_score(
                    date,
                    historical_destine,
                    knmi_input_dir,
                    destineE_datafolder,
                    timesteps,
                    timestep_interval,
                    n_ens_members,
                    n_ens_members_dgmr,
                    weights_method="bps",
                    custom_weights=None,
                    return_weights=True,
                    re_do_blending=True,
                )

                blend_precip_clim_swp = np.moveaxis(blend_precip_clim, 0, 3)
                blend_precip_dynamic_swp = np.moveaxis(blend_precip_dynamic, 0, 3)
                blend_precip_clim_swp = np.moveaxis(blend_precip_clim_swp, 0, 2)
                blend_precip_dynamic_swp = np.moveaxis(blend_precip_dynamic_swp, 0, 2)
                blend_precip_clim_swp = np.moveaxis(blend_precip_clim_swp, 0, 1)
                blend_precip_dynamic_swp = np.moveaxis(blend_precip_dynamic_swp, 0, 1)
                radar_precip_swp = np.moveaxis(radar_precip, 0, 2)

                # calculate CRPS for both
                # crps_clim = sr.crps_ensemble(radar_precip_swp, blend_precip_clim_swp)
                # crps_dynamic = sr.crps_ensemble(radar_precip_swp, blend_precip_dynamic_swp)

                # Example dynamic values (replace with your real second run)
                GAMMA_dynamic = weights_dynamic["GAMMA"]
                regr_pars_dynamic = weights_dynamic["regr_pars"]
                clim_cor_values_dynamic = weights_dynamic["clim_cor"]

                GAMMA_dynamic = np.array(
                    [
                        [0.99477983, 0.98303498],
                        [0.94658293, 0.84849088],
                        [0.75459812, 0.53982899],
                        [0.35159685, 0.13695139],
                        [0.16055702, 0.13328882],
                        [0.13879112, 0.15488655],
                    ]
                )
                clim_cor_values_dynamic = np.array(
                    [0.848, 0.537, 0.237, 0.065, 0.02, 0.0044]
                )
                regr_pars_dynamic = np.array(
                    [
                        [1.30e02, 1.65e02, 1.20e02, 5.50e01, 5.00e01, 1.50e01],
                        [1.55e02, 2.20e02, 2.00e02, 7.50e01, 1.00e05, 1.00e05],
                    ]
                )

                # ============================================================
                # 2️⃣ Helper to create sampling bounds
                # ============================================================
                def make_range(base, dynamic):
                    diff = np.abs(dynamic - base)
                    lower = np.minimum(base, dynamic) - 1 * diff
                    upper = np.maximum(base, dynamic) + 1 * diff
                    return np.clip(lower, 0, 1), np.clip(upper, 0, 1)

                # ============================================================
                # 3️⃣ Objective function (Optuna trial)
                # ============================================================
                def objective(trial):
                    # --- Sample GAMMA (monotonic decreasing constraint)
                    GAMMA = np.zeros_like(GAMMA_base)
                    for j in range(2):
                        lower, upper = make_range(GAMMA_base[:, j], GAMMA_dynamic[:, j])
                        prev_val = 1.0
                        for i in range(6):
                            val = trial.suggest_float(
                                f"GAMMA_{i}_{j}", lower[i], upper[i]
                            )
                            val = min(val, prev_val)  # enforce decreasing
                            GAMMA[i, j] = val
                            prev_val = val

                    # --- Sample regr_pars (no monotonic constraint)
                    regr_pars = np.zeros_like(regr_pars_base)
                    for j in range(2):
                        lower, upper = make_range(
                            regr_pars_base[j, :], regr_pars_dynamic[j, :]
                        )
                        for i in range(6):
                            regr_pars[j, i] = trial.suggest_float(
                                f"regr_{j}_{i}", lower[i], upper[i]
                            )

                    # --- Sample clim_cor_values (decreasing constraint)
                    lower, upper = make_range(
                        clim_cor_values_base, clim_cor_values_dynamic
                    )
                    clim_cor_values = np.zeros_like(clim_cor_values_base)
                    prev_val = 1.0
                    for i in range(6):
                        val = trial.suggest_float(f"clim_{i}", lower[i], upper[i])
                        val = min(val, prev_val)
                        clim_cor_values[i] = val
                        prev_val = val

                    # --- Prepare parameter dictionary
                    custom_weights = {
                        "GAMMA": GAMMA,
                        "regr_pars": regr_pars,
                        "clim_cor_values": clim_cor_values,
                    }

                    # ========================================================
                    #  Run your blending model here
                    # ========================================================
                    #
                    # Example:
                    # crps_score = run_blending_and_compute_CRPS(custom_weights)
                    (
                        blend_precip_optuna,
                        nowcast_precip,
                        nwp_precip,
                        radar_precip,
                        weights_optuna,
                    ) = skill_score(
                        date,
                        historical_destine,
                        knmi_input_dir,
                        destineE_datafolder,
                        timesteps,
                        timestep_interval,
                        n_ens_members,
                        n_ens_members_dgmr,
                        weights_method="bps",
                        custom_weights=custom_weights,
                        return_weights=True,
                        re_do_blending=True,
                    )

                    blend_precip_optuna_swp = np.moveaxis(blend_precip_optuna, 0, 3)
                    blend_precip_optuna_swp = np.moveaxis(blend_precip_optuna_swp, 0, 2)
                    blend_precip_optuna_swp = np.moveaxis(blend_precip_optuna_swp, 0, 1)
                    radar_precip_swp = np.moveaxis(radar_precip, 0, 2)

                    crps_score = sr.crps_ensemble(
                        radar_precip_swp, blend_precip_optuna_swp
                    )
                    crps_score_mean = np.nanmean(crps_score)

                    return crps_score_mean  # CRPS → minimize

                # ============================================================
                # 4️⃣ Study setup with persistent SQLite storage
                # ============================================================

                # Create timestamped study name
                study_name = f"pysteps_blending_{date_str}"

                # SQLite DB file (will be created automatically)
                storage = f"sqlite:///optuna_blending_study.db"

                # Create or load the study
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage,
                    direction="minimize",
                    load_if_exists=True,
                )

                # ============================================================
                # 5️⃣ Warm-start the study with two baseline trials
                # ============================================================
                def flatten_params(GAMMA, regr_pars, clim_cor_values):
                    params = {}
                    for j in range(2):
                        for i in range(6):
                            params[f"GAMMA_{i}_{j}"] = float(GAMMA[i, j])
                    for j in range(2):
                        for i in range(6):
                            params[f"regr_{j}_{i}"] = float(regr_pars[j, i])
                    for i in range(6):
                        params[f"clim_{i}"] = float(clim_cor_values[i])
                    return params

                # Flatten the parameter sets
                climatological_params = flatten_params(
                    GAMMA_base, regr_pars_base, clim_cor_values_base
                )
                operational_params = flatten_params(
                    GAMMA_dynamic, regr_pars_dynamic, clim_cor_values_dynamic
                )

                # Add them to the queue (these will be the first two runs)
                study.enqueue_trial(climatological_params)
                study.enqueue_trial(operational_params)

                print("🧊 Enqueued climatological and operational baseline runs.")

                # ============================================================
                # 6️⃣ Run the optimization
                # ============================================================
                print(f"🚀 Starting optimization for study: {study_name}")
                study.optimize(
                    objective,
                    n_trials=30,  # adjust to available time (e.g. 10–15 mins each)
                    n_jobs=1,  # sequential, since runs are long
                    timeout=6 * 3600,  # 6 hours max runtime
                )

                # ============================================================
                #  Save and summarize results
                # ============================================================
                print("✅ Optimization complete.")
                print(f"Best CRPS: {study.best_value:.4f}")
                print("Best parameters:")
                for key, val in study.best_trial.params.items():
                    print(f"  {key}: {val}")
