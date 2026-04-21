# cd /usr/people/whan/ResearchDataLab/floodMIND/running_rt/nowcast-blend-code/
# source /nobackup_1/users/whan/floodmind/floodmind_rt_env/bin/activate  
# python download_datasets/main_download.py

# -*- coding: utf-8 -*-

import sys
import os
import io
import numpy as np
import xarray as xr
import pandas as pd

from datetime import datetime, timedelta
import time

sys.path.insert(0, "../nowcast_blend/")


#download
from download_ifs import download_ifs, pre_process_ifs_data



import warnings

# one function makes this warning: 
# # /usr/people/whan/ResearchDataLab/floodMIND/running_rt/DestinE_code/nowcast_blend/preprocess/preprocess_radar.py:175: UserWarning: ypixelsize does not match y1, y2 and array shape, using ypixelsize for pixel size
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="nowcast_blend.preprocess.preprocess_radar"
)

# how to import the config interactively:
#with hydra.initialize(version_base=None, config_path="configs"):
#    cfg: DictConfig = hydra.compose(config_name="nowcast-blend.yaml")


import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


    
def closest_ecmwf_available(dt: datetime) -> str:
    """
    Returns the latest ECMWF cycle available at the given datetime.
    
    ECMWF init times: 00, 06, 12, 18 UTC
    Availability: ~7 hours after init
    """
    # ECMWF init hours
    init_hours = [0, 6, 12, 18]
    
    # Availability delay in hours
    availability_delay = 7
    
    # Adjust datetime back by availability delay
    dt_adjusted = dt - timedelta(hours=availability_delay)
    
    # Find all init hours <= adjusted hour
    available_inits = [h for h in init_hours if h <= dt_adjusted.hour]
    
    if available_inits:
        latest_init = max(available_inits)
        dt_init = dt.replace(hour=latest_init, minute=0, second=0, microsecond=0)
    else:
        # No available run today → use previous day's 18Z
        latest_init = 18
        dt_init = (dt - timedelta(days=1)).replace(hour=latest_init, minute=0, second=0, microsecond=0)
    
    return dt_init.strftime("%Y%m%d%H")
  
verb = True  


@hydra.main(version_base=None, config_path="../configs", config_name="download_ifs.yaml")
def main(cfg: DictConfig):     

    # execute every 6 hours. work from the current date or the config file if rt is false:
    if cfg.settings.rt:
        if verb:
            log.info(f"Using real time date as config == {cfg.settings.rt}")
        date=datetime.now()
        yr = date.year
        mnth = date.month
        date_str = date.strftime('%Y%m%d%H')
    else:
        if verb:
            log.info(f"Using date from the config as config == {cfg.settings.rt}")
        yr = int(cfg.rundate.year)
        mnth = int(cfg.rundate.month)
        day = int(cfg.rundate.day)
        hour = int(cfg.rundate.hour)
        date = datetime(yr, mnth, day, hour)
        date_str = date.strftime('%Y%m%d%H')
    
    
    ifs_init_time = closest_ecmwf_available(date)
    log.info(f"Running the code for date {date_str} and closest ECMWF init time is {ifs_init_time}")
       
    # region define paths from the config file:
    # base path
    base_input_dir = cfg.paths.input_project + cfg.paths.input_general    
    # ifs paths:
    ifs_path = base_input_dir + cfg.paths.input_ifs + str(yr) + '/' + format(str(mnth).zfill(2)) + '/'
    # make sure destine folder exists
    for folder in [ifs_path]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    ifs_file_original = ifs_path + f"IFS_{ifs_init_time}_{cfg.settings.param}.grib"
    
    
    # region - ifs
    # ifs - download and preprocessed
    log.info(f"----------------------------------------------------------------------------------------------")
    log.info(f"2b. IFS data - download if needed and preprocess:")
    log.info(f"----------------------------------------------------------------------------------------------")
    if not os.path.exists(ifs_file_original):
        log.info(f"downloading ifs file: {ifs_file_original}")
        download_ifs(ifs_init_time = ifs_init_time, ifs_file_original = ifs_file_original, cfg = cfg)
    else:
        if verb:
            log.info(f"ifs file already downloaded: {ifs_file_original}")

    
    # endregion
    


if __name__ == "__main__":
    main()