    
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


from datetime import datetime, timedelta


from pathlib import Path
import xarray as xr

import re
from pathlib import Path

date_pattern = re.compile(r"_(\d{8})_")
LAT_SLICE = slice(56.4, 48.4)
LON_SLICE = slice(-1.0, 11.87)

def hour_block(hour):
    if hour < 9:
        return "00"
    elif hour < 15:
        return "06"
    elif hour < 21:
        return "12"
    elif hour < 25:
        return "18"
    # elif hour < 31:
    #     return "25-30"
    # elif hour < 37:
    #     return "31-36"
    # elif hour < 43:
    #     return "37-42"
    else:
        return 
def extract_date_infunc(path : Path):
    match = date_pattern.search(path.name)
    if not match:
        raise ValueError(f"Cannot extract date from {path.name}")
    ymd = match.group(1)
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    
def blocks_meeting_condition(path):

    with xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""}
    ) as ds:
        ds = ds.sel(latitude=LAT_SLICE, longitude=LON_SLICE)

        # If spatial selection removes everything, skip safely
        if ds.sizes.get("latitude", 0) == 0 or ds.sizes.get("longitude", 0) == 0:
            return {}

        R = ds["tp"] *1000
        R = R - R.shift({'step': 1})
        R = R.dropna('step', how="all")

        # Hourly wet-area counts
        wet_area = (R > 1).sum(dim=("latitude", "longitude"))
        wet_area_middle = (R > 3).sum(dim=("latitude", "longitude"))
        wet_area_high = (R > 10).sum(dim=("latitude", "longitude"))

        # Explicit hour → block mapping (safe)
        hour = [int(i / 3600000000000)  for i in wet_area.step.values]
        wet_area = wet_area.assign_coords(step=hour)
        wet_area_middle = wet_area_middle.assign_coords(step=hour)
        wet_area_high = wet_area_high.assign_coords(step=hour)
        block = xr.apply_ufunc(
            hour_block,
            wet_area.step,
            vectorize=True,
            dask="allowed",
            output_dtypes=[str],
        )

        wet_area = wet_area.assign_coords(block=block)
        wet_area_middle = wet_area_middle.assign_coords(block=block)
        wet_area_high = wet_area_high.assign_coords(block=block)

        # Aggregate per block
        wet_blk = wet_area.groupby("block").max()
        wet_middle_blk = wet_area_middle.groupby("block").max()
        wet_high_blk = wet_area_high.groupby("block").max()

        condition = (wet_blk > 1300) | (wet_high_blk > 32)
        condition_high = (wet_middle_blk > 1300) | (wet_high_blk > 64)

        condition_dict = condition.to_series().to_dict()
        condition_dict_high = condition_high.to_series().to_dict()

        day = datetime.strptime(extract_date_infunc(path), '%Y-%m-%d')
        print(day)
        rain_date = []
        rain_date_high = []
        for start_hour, flag in condition_dict.items():
            if not flag:
                continue
            if start_hour == 'None':
                continue
            start_dt = day + timedelta(hours=int(start_hour))
            rain_date.append(start_dt)
        
        for start_hour, flag in condition_dict_high.items():
            if not flag:
                continue
            if start_hour == 'None':
                continue
            start_dt_high = day + timedelta(hours=int(start_hour))
            rain_date_high.append(start_dt_high)

        return rain_date, rain_date_high


path_trial = "/srv/data/nas/project_data/p111_ecmwf_destine/ExtremesDT/2024/5/original/DestinE_ExtremesDT_20240524_218.228-219.228-228.128.grib"

results = []

path_grib = '/srv/data/nas//project_data/p111_ecmwf_destine/ExtremesDT'

paths = [path for path in Path(path_grib).rglob("*.grib")]

for path in paths[:920]:
    try:
        dates, dates_high = blocks_meeting_condition(path)
        results.append([dates, dates_high])

    except:
        print(f'skipping file: {path}')



rainy_days_first = np.load('/srv/data/nas//project_data/p111_ecmwf_destine/rainy_days_first.npy', allow_pickle = True)

np.save('/srv/data/nas//project_data/p111_ecmwf_destine/rainy_days_mild_and_heavy', results)




results_mild = [result[0] for result in results]
results_heavy = [result[1] for result in results]

results_mild_processed = [x for xs in results_mild_processed for x in xs if x != []]
results_heavy_processed = [x for xs in results_heavy for x in xs if x != []]

from itertools import chain

rainy_days_total = list(chain(rainy_days_first,results_mild_processed))

with open('/srv/data/nas//project_data/p111_ecmwf_destine/rainy_days.txt', "r") as f:
    text = f.read()

np.save('/srv/data/nas//project_data/p111_ecmwf_destine/rainy_days_total', rainy_days_total)


results_heavy_total = list(chain(results_heavy_processed, results_heavy_processed_part_2))

np.save('/srv/data/nas//project_data/p111_ecmwf_destine/rainy_days_heavy_total', results_heavy_total)

days = np.load('/srv/data/nas//project_data/p111_ecmwf_destine/rainy_days_heavy_total.npy', allow_pickle=True)

days_not_yet_download = days[49:]
np.save('/srv/data/nas//project_data/p111_ecmwf_destine/rainy_days_heavy_total_undownloaded', days_not_yet_download)

import re

pattern = re.compile(
    r"""
    (?P<date>\d{8})        # YYYYMMDD
    [^{}]{0,300}?          # anything before the dict
    :
    \s*
    (?P<dict>\{[^}]*\})   # the dictionary itself (single-level)
    """,
    re.VERBOSE | re.DOTALL,
)

# results = []

# import ast
# results = []

# for m in pattern.finditer(text):
#     ymd = m.group("date")
#     dict_text = m.group("dict")
#     if "None" not in dict_text:
#         day = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
#         try:
#             for key, value in ast.literal_eval(dict_text).items():
#                 print(key, value)
#                 if value:
#                     day_datetime = datetime.strptime(day, '%Y-%m-%d')
#                     day_comb = day_datetime + timedelta(hours= int(key))
#                     print(day_comb)
#                     results.append(day_comb)
#         except:
#             print(f'failed day: {day} ')
#             continue

# np.save('/srv/data/nas//project_data/p111_ecmwf_destine/rainy_days_first', results)

# len(results)

# import re
# from pathlib import Path

# date_pattern = re.compile(r"_(\d{8})_")

# def extract_date(path: Path) -> str:
#     match = date_pattern.search(path.name)
#     if not match:
#         raise ValueError(f"Cannot extract date from {path.name}")
#     ymd = match.group(1)
#     return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    

# events = []

# for path, block_dict in results.items():
#     day = extract_date(path)

#     for start_hour, flag in block_dict.items():
#         if not flag:
#             continue
#         if start_hour == 'None':
#             continue
#         start_dt = day + timedelta(hours=int(start_hour))

#         events.append([start_dt])
