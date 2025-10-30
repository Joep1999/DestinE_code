# -*- coding: utf-8 -*-
"""
Blended forecast
====================

This tutorial shows how to construct a blended forecast from an ensemble nowcast
using the STEPS approach and a Numerical Weather Prediction (NWP) rainfall
forecast. The used datasets are from the Bureau of Meteorology, Australia.

"""

import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(1, "C:/Users/Joep.Bosdijk/git/DestinE_code/pysteps_destine")
import pysteps

print(pysteps.__file__)

from pysteps import io, rcparams, blending
from pysteps.visualization import plot_precip_field

from pysteps import utils
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.cascade import decomposition
from pysteps.utils import conversion, transformation

from pysteps.blending import steps

import sys

# caution: path[0] is reserved for script path (or '' in REPL)
# sys.path.insert(1, "C:/Users/Joep.Bosdijk/git/DestinE_code")

# import dgmr_operational

# from git.destinE_code import dgmr_operational

################################################################################
# Read the radar images and the NWP forecast
# ------------------------------------------
#
# First, we import a sequence of 3 images of 10-minute radar composites
# and the corresponding NWP rainfall forecast that was available at that time.
#
# You need the pysteps-data archive downloaded and the pystepsrc file
# configured with the data_source paths pointing to data folders.
# Additionally, the pysteps-nwp-importers plugin needs to be installed, see
# https://github.com/pySTEPS/pysteps-nwp-importers.

# Selected case
date_radar = datetime.strptime("202010310400", "%Y%m%d%H%M")
date_radar_nowcast = datetime.strptime("202010310700", "%Y%m%d%H%M")
# The last NWP forecast was issued at 00:00
date_nwp = datetime.strptime("202010310000", "%Y%m%d%H%M")
radar_data_source = rcparams.data_sources["bom"]
nwp_data_source = rcparams.data_sources["bom_nwp"]

###############################################################################
# Load the data from the archive
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

root_path = radar_data_source["root_path"]
root_path_nwp = "../../pysteps-data/nwp/bom"
root_path = "../../pysteps-data/radar/bom"
path_fmt = "prcp-c10/66/%Y/%m/%d"
fn_pattern = "66_%Y%m%d_%H%M00.prcp-c10"
fn_ext = radar_data_source["fn_ext"]
importer_name = radar_data_source["importer"]
importer_kwargs = radar_data_source["importer_kwargs"]
timestep = 10.0


# Find the radar files in the archive
fns = io.find_by_date(
    date_radar, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=2
)

fns_radarnowcast = io.find_by_date(
    date_radar_nowcast,
    root_path,
    path_fmt,
    fn_pattern,
    fn_ext,
    timestep,
    num_prev_files=18,
)

# Read the radar composites
importer = io.get_method(importer_name, "importer")
radar_precip, _, radar_metadata = io.read_timeseries(fns, importer, **importer_kwargs)
radar_precip_nowcast, _, radar_metadata_nowcast = io.read_timeseries(
    fns_radarnowcast, importer, **importer_kwargs
)

# Import the NWP data
filename = os.path.join(
    root_path_nwp,
    datetime.strftime(date_nwp, nwp_data_source["path_fmt"]),
    datetime.strftime(date_nwp, nwp_data_source["fn_pattern"])
    + "."
    + nwp_data_source["fn_ext"],
)

nwp_importer = io.get_method("bom_nwp", "importer")
nwp_precip, _, nwp_metadata = nwp_importer(filename)

# Only keep the NWP forecasts from the last radar observation time (2020-10-31 04:00)
# onwards

nwp_precip = nwp_precip[24:43, :, :]


# Convert to rain rate
radar_precip_nowcast, radar_metadata_nowcast = conversion.to_rainrate(
    radar_precip_nowcast, radar_metadata_nowcast
)

# Nicely print the metadata
print(radar_metadata_nowcast)

# Plot the rainfall field
# plot_precip_field(radar_precip_nowcast[0], geodata=radar_metadata_nowcast)
# plt.show()

# Log-transform the data
radar_precip_nowcast, radar_metadata_nowcast = transformation.dB_transform(
    radar_precip_nowcast, radar_metadata_nowcast, threshold=0.1, zerovalue=-15.0
)

filter = filter_gaussian(radar_precip_nowcast[0].shape, 6)

if radar_precip_nowcast.ndim == 3:
    radar_precip_nowcast = radar_precip_nowcast[None, :]

# Simulate like it has ensembles
radar_precip_nowcast = np.repeat(radar_precip_nowcast, 5, axis=0)

from pysteps.nowcasts import utils as nowcast_utils

# cascade_connected_ens = []
# for j in range(radar_precip_nowcast.shape[0]):
#     cascade_connected = []
#     for i in range(radar_precip_nowcast.shape[1]):
#         cascades = decomposition.decomposition_fft(
#             radar_precip_nowcast[j][i],
#             bp_filter=filter,
#             n_levels=6,
#             method="fft",
#             fft_method="numpy",
#             input_domain="spatial",
#             output_domain="spatial",
#             compute_stats=True,
#             normalize=True,
#             compact_output=True,
#         )

#         # cascade_connected.append(cascades['cascade_levels'])
#         cascade_connected.append(cascades)

#     cascade_connected_stacked = nowcast_utils.stack_cascades(cascade_connected, 6)
#     cascade_connected_ens.append(cascade_connected_stacked)

# cascade_connected_stacked = nowcast_utils.stack_cascades(cascade_connected, 6)

################################################################################
# Pre-processing steps
# --------------------

# Make sure the units are in mm/h
converter = pysteps.utils.get_method("mm/h")
radar_precip, radar_metadata = converter(radar_precip, radar_metadata)
nwp_precip, nwp_metadata = converter(nwp_precip, nwp_metadata)

# Threshold the data
radar_precip[radar_precip < 0.1] = 0.0
nwp_precip[nwp_precip < 0.1] = 0.0

# Plot the radar rainfall field and the first time step of the NWP forecast.
# date_str = datetime.strftime(date_radar, "%Y-%m-%d %H:%M")
# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# plot_precip_field(
#     radar_precip[-1, :, :],
#     geodata=radar_metadata,
#     title=f"Radar observation at {date_str}",
#     colorscale="STEPS-NL",
# )
# plt.subplot(122)
# plot_precip_field(
#     nwp_precip[0, :, :],
#     geodata=nwp_metadata,
#     title=f"NWP forecast at {date_str}",
#     colorscale="STEPS-NL",
# )
# plt.tight_layout()
# plt.show()

# transform the data to dB
transformer = pysteps.utils.get_method("dB")
radar_precip, radar_metadata = transformer(radar_precip, radar_metadata, threshold=0.1)
nwp_precip, nwp_metadata = transformer(nwp_precip, nwp_metadata, threshold=0.1)

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
rounded_arr = np.round(radar_precip_nowcast, 1)
values, counts = np.unique(rounded_arr, return_counts=True)

# Find the most common value
most_common = values[np.argmax(counts)]

print("Most common number second var:", most_common)
print("Count of this number second var:", counts.max())

GAMMA = np.array(
    [
        [0.99805, 0.9933],
        [0.9925, 0.923],
        [0.9776, 0.975],
        [0.9297, 0.750],
        [0.796, 0.367],
        [0.482, 0.069],
        [0.079, 0.0018],
        [0.0006, 0.0014],
    ]
)


regr_pars = np.array(
    [
        [130.0, 165.0, 120.0, 55.0, 50.0, 15.0, 15.0, 10.0],
        [155.0, 220.0, 200.0, 75.0, 10e4, 10e4, 10e4, 10e4],
    ]
)
from pysteps.blending import clim

clim_cor_values = np.array([0.848, 0.537, 0.237, 0.065, 0.02, 0.0044])

custom_weights = {
    "GAMMA": GAMMA,
    "regr_pars": regr_pars,
    "clim_cor_values": clim_cor_values,
}

precip_forecast_stacked = blending.steps.forecast(
    precip=radar_precip,
    precip_nowcast=radar_precip_nowcast,
    nowcasting_method="external_nowcast",
    mask_method=None,
    precip_models=nwp_precip,
    velocity=velocity_radar,
    velocity_models=velocity_nwp,
    timesteps=18,
    timestep=timestep,
    issuetime=date_radar,
    n_ens_members=25,
    # resample_distribution=False,
    precip_thr=radar_metadata["threshold"],
    kmperpixel=radar_metadata["xpixelsize"] / 1000.0,
    weights_method="custom",
    custom_weights=custom_weights,
    # noise_stddev_adj=None,
    # noise_method=None,
    probmatching_method="cdf",
    vel_pert_method=None,
)


phi_extra = [
    [1.6206979, -0.62920261, 0.07931322],
    [1.3792547, -0.45708808, 0.28680326],
    [0.80645367, -0.06871943, 0.65463602],
    [0.34624854, 0.01521149, 0.9360432],
    [0.14283874, 0.11035506, 0.98099803],
    [0.11959806, 0.1382874, 0.98080681],
]


# Transform the data back into mm/h
precip_forecast_mm, _ = converter(precip_forecast_stacked, radar_metadata)
radar_precip_mm, _ = converter(radar_precip_nowcast, radar_metadata)
nwp_precip_mm, _ = converter(nwp_precip, nwp_metadata)


def plot_hotspot_and_timeseries(ens_fcst_all, case_index=0):
    """
    Identify and plot the grid cell with the highest ensemble precipitation
    for a given forecast case, and show its ensemble time series.

    Parameters
    ----------
    obs_all : np.ndarray
        Observations [time, lat, lon]
    ens_fcst_all : np.ndarray
        Ensemble forecasts [time, member, lat, lon]
    case_index : int
        Index of the forecast case to inspect (default=0)
    lats, lons : np.ndarray, optional
        Latitude/longitude arrays for plotting
    """
    # --- Identify the hotspot grid cell
    fcst_case = ens_fcst_all[case_index]  # [member, lat, lon]
    ens_mean = np.mean(fcst_case, axis=0)
    ens_max = np.max(fcst_case, axis=0)

    # Find gridcell with maximum ensemble precipitation
    hotspot_idx = np.unravel_index(np.argmax(ens_max), ens_max.shape)
    lat_idx, lon_idx = hotspot_idx
    hotspot_value = ens_max[lat_idx, lon_idx]

    # --- Plot the spatial field with hotspot
    # plt.figure(figsize=(7, 5))
    # plt.imshow(ens_max, cmap="Blues", origin="lower")
    # plt.scatter(lon_idx, lat_idx, color="red", s=80, label="Max gridcell")
    # plt.colorbar(label="Max ensemble precipitation [mm]")
    # plt.title(f"Ensemble Max Precipitation (case {case_index})\nMax={hotspot_value:.1f} mm")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # --- Plot the time series for that grid cell
    cell_series = ens_fcst_all[:, :, lat_idx, lon_idx]  # [time, member]
    mean_series = np.mean(cell_series, axis=1)

    cell_series_accum = np.cumsum(cell_series, axis=0)
    mean_series_accum = np.cumsum(mean_series, axis=0)

    plt.figure(figsize=(7, 4))
    plt.plot(mean_series_accum, "k-", lw=2, label="Ensemble mean")
    plt.plot(cell_series_accum, lw=0.8, alpha=0.5)
    plt.xlabel("Model timestep (10 minutes per timestep)")
    plt.ylabel("Precipitation [mm]")
    plt.title(f"Time Series at Max-Precip Grid Cell (lat={lat_idx}, lon={lon_idx})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(
    # destineE_datafolder + 'verification/ensemble_spread_hotspot_' + str(date_str) + '.png', dpi=300
    # )

    return {
        "hotspot_index": (lat_idx, lon_idx),
        "hotspot_value": hotspot_value,
        "cell_series": cell_series,
        "mean_series": mean_series,
    }


precip_forecast_mm_accum = precip_forecast_mm / 6

plot_hotspot_and_timeseries(precip_forecast_mm_accum)

precip_forecast_stacked

################################################################################
# Visualize the output
# ~~~~~~~~~~~~~~~~~~~~
#
# The NWP rainfall forecast has a lower weight than the radar-based extrapolation
# forecast at the issue time of the forecast (+0 min). Therefore, the first time
# steps consist mostly of the extrapolation.
# However, near the end of the forecast (+180 min), the NWP share in the blended
# forecast has become more important and the forecast starts to resemble the
# NWP forecast more.


if radar_precip_nowcast.ndim == 3:
    radar_precip_nowcast = radar_precip_nowcast[None, :]
radar_precip_mm, _ = converter(radar_precip_nowcast, radar_metadata)

fig = plt.figure(figsize=(5, 12))

leadtimes_min = [30, 60, 90, 120, 150, 180]
n_leadtimes = len(leadtimes_min)
for n, leadtime in enumerate(leadtimes_min):
    # Nowcast with blending into NWP

    plt.subplot(n_leadtimes, 3, n * 3 + 1)
    plot_precip_field(
        precip_forecast_mm[1, int(leadtime / timestep) - 1, :, :],
        geodata=radar_metadata,
        title=f"Blended forecast +{leadtime} min",
        axis="on",
        colorscale="STEPS-NL",
        colorbar=False,
    )

    # Raw NWP forecast

    plt.subplot(n_leadtimes, 3, n * 3 + 2)
    plot_precip_field(
        nwp_precip_mm[0, int(leadtime / timestep) - 1, :, :],
        geodata=nwp_metadata,
        title=f"NWP +{leadtime} min",
        axis="on",
        colorscale="STEPS-NL",
        colorbar=False,
    )

    # Raw precip forecast
    plt.subplot(n_leadtimes, 3, n * 3 + 3)

    plot_precip_field(
        radar_precip_mm[0, int(leadtime / timestep) - 1, :, :],
        geodata=radar_metadata,
        title=f"Nowcast +{leadtime} min",
        axis="on",
        colorscale="STEPS-NL",
        colorbar=False,
    )
plt.show()


import matplotlib.pyplot as plt

# Choose fewer lead times (4 instead of 6)
leadtimes_min = [30, 60, 90, 120]
n_leadtimes = len(leadtimes_min)

fig, axes = plt.subplots(n_leadtimes, 3, figsize=(12, 12))  # wider and taller

for n, leadtime in enumerate(leadtimes_min):
    plt.subplot(n_leadtimes, 3, n * 3 + 1)
    # Nowcast with blending into NWP
    plot_precip_field(
        precip_forecast_mm[1, int(leadtime / timestep) - 1, :, :],
        geodata=radar_metadata,
        title=f"Blended forecast +{leadtime} min",
        colorscale="STEPS-NL",
        colorbar=False,
        # axis="off",
        ax=axes[n, 0],  # pass the axis
    )

    plt.subplot(n_leadtimes, 3, n * 3 + 2)
    # Raw NWP forecast
    plot_precip_field(
        nwp_precip_mm[0, int(leadtime / timestep) - 1, :, :],
        geodata=nwp_metadata,
        title=f"NWP +{leadtime} min",
        colorscale="STEPS-NL",
        colorbar=False,
        axis="off",
        # ax=axes[n, 1],
    )

    plt.subplot(n_leadtimes, 3, n * 3 + 3)
    # Raw precip forecast
    plot_precip_field(
        radar_precip_mm[0, int(leadtime / timestep) - 1, :, :],
        geodata=radar_metadata,
        title=f"Nowcast +{leadtime} min",
        colorscale="STEPS-NL",
        colorbar=False,
        axis="off",
        # ax=axes[n, 2],
    )

plt.savefig(
    "C:/Users/Joep.Bosdijk/git/DestinE-code/blended_forecast_comparison.png", dpi=300
)  # save the figure
plt.tight_layout()
plt.show()
################################################################################
# References
# ~~~~~~~~~~
#
# Bowler, N. E., and C. E. Pierce, and A. W. Seed. 2004. "STEPS: A probabilistic
# precipitation forecasting scheme which merges an extrapolation nowcast with
# downscaled NWP." Forecasting Research Technical Report No. 433. Wallingford, UK.
#
# Bowler, N. E., and C. E. Pierce, and A. W. Seed. 2006. "STEPS: A probabilistic
# precipitation forecasting scheme which merges an extrapolation nowcast with
# downscaled NWP." Quarterly Journal of the Royal Meteorological Society 132(16):
# 2127-2155. https://doi.org/10.1256/qj.04.100
#
# Seed, A. W., and C. E. Pierce, and K. Norman. 2013. "Formulation and evaluation
# of a scale decomposition-based stochastic precipitation nowcast scheme." Water
# Resources Research 49(10): 6624-664. https://doi.org/10.1002/wrcr.20536
