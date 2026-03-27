# cd /usr/people/whan/ResearchDataLab/floodMIND/running_rt/nowcast-blend-code/
# source /nobackup_1/users/whan/floodmind/floodmind_rt_env/bin/activate  
# python scripts/main.py

# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "/usr/people/whan/ResearchDataLab/floodMIND/running_rt/nowcast-blend-code/")
import os
import io
import numpy as np
import xarray as xr
import pandas as pd

from datetime import datetime, timedelta
import time


#from configparser import ConfigParser
import hydra
from omegaconf import DictConfig

import pysteps
from pysteps import io
from pysteps.utils import conversion, transformation

from nowcast_blend.utils.utils import floor_to_30min, validate_destine_file, ensure_destine_time_dim, validate_destine_time_range
from nowcast_blend.download.download_radar import run_download_radar
from nowcast_blend.preprocess.preprocess_radar import convert_input_to_xarray_dataset
from nowcast_blend.download.download_destine import download_destine
from nowcast_blend.preprocess.preprocess_destine import pre_process_destine_data
#from nowcast_blend.blending.dgmr_for_blending import run_dgmr_ensemble
#from nowcast_blend.blending.blending import blending_function


#with hydra.initialize(version_base=None, config_path="../configs"):
#    cfg: DictConfig = hydra.compose(config_name="nowcast-blend.yaml")

#with hydra.initialize(version_base=None, config_path="./running_rt/nowcast-blend-code/configs"):
#    cfg: DictConfig = hydra.compose(config_name="nowcast-blend.yaml")

# use hydra configs
@hydra.main(version_base=None, config_path="../configs", config_name="nowcast-blend.yaml")
def main(cfg: DictConfig):     
    # import config info from config file:
    #config = ConfigParser(inline_comment_prefixes=("#", ";"))
    #config.read('./configs/nowcast-blend.ini')
    verb = cfg.settings.verbose
    
    # define the date: either from RT or from the config file
    if cfg.settings.rt:
        if verb:
            print(f"Using real time date as config == {cfg.settings.rt}")
        date_orig = datetime.now()
        #KW: round it down to the closest 30-minutes because I get errors subsetting the DestinEtimes when I'm using any other minutes
        date = floor_to_30min(date_orig)
        yr = date.year
        mnth = date.month
        date_str = date.strftime('%Y%m%d%H')
        date_str_day = date_str[:-2]
    else:
        if verb:
            print(f"Using date from the config as config == {cfg.settings.rt}")
        yr = cfg.rundate.year
        mnth = cfg.rundate.month
        day = cfg.rundate.day
        hour = cfg.rundate.hour
        date = datetime(yr, mnth, day, hour)
        date_str = date.strftime('%Y%m%d%H')
        date_str_day = date_str[:-2]
        
       
    # region define paths from the config file:
    # base path
    base_input_dir = cfg.paths.input_project + cfg.paths.input_general
    # destine paths:
    destine_path = base_input_dir + cfg.paths.input_destine + str(yr) + '/' + format(str(mnth).zfill(2)) + '/'
    destine_file_preprocessed = destine_path  + f'/DestinE_ExtremesDT_{date_str_day}_228_hres_interp_nlgrid_{cfg.settings.timestep_interval}_{cfg.settings.timesteps}.nc' 
    destine_file_original = destine_path + f'/DestinE_ExtremesDT_{date_str_day}_{cfg.settings.param}.grib'
    destine_file_original_nc = destine_path + f'/DestinE_ExtremesDT_{date_str}_{cfg.settings.param}_regrid_nl.nc'
    # make sure destine folder exists
    for folder in [destine_path]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    
    # dgmr path:
    dgmr_path = base_input_dir + cfg.paths.input_dgmr + str(yr) + '/' + format(str(mnth).zfill(2))
    dgmr_file = dgmr_path + f'/DGMR_{date_str}_step_min_{cfg.settings.timestep_interval}_len_{cfg.settings.timesteps}_ens_{cfg.settings.n_ens_members_dgmr}.npy'
    # make sure dgmr folder exists
    for folder in [dgmr_path]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # radar paths:
    if cfg.settings.gauge_adjusted:
        radar_path = base_input_dir + cfg.paths.input_radar_gauge_adj_knmi
    else:
        radar_path = base_input_dir + cfg.paths.input_radar_knmi
    
    # blended paths:
    blended_path =  base_input_dir + cfg.paths.input_blended + str(yr) + '/' + format(str(mnth).zfill(2))
    for folder in [blended_path]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    if cfg.settings.pysteps_nowcast:
        blended_file = blended_path + f'/Blended_forecast_{date_str}_step_min_{cfg.settings.timestep_interval}_len_{cfg.settings.timesteps}_pysteps_nowcast.npy'
        blended_file_weights = blended_path + f'/Blended_forecast_{date_str}_step_min_{cfg.settings.timestep_interval}_len_{cfg.settings.timesteps}_pysteps_nowcast_weights.npy'
    else:
        if cfg.settings.multi_model:
            multi_extention = '_IFS'
        else:
            multi_extention = ''

        if cfg.settings.custom_weights == None:
            custom_weights_extention = ''
        else:
            custom_weights_extention = '_optimised_weights'

        if cfg.settings.noise:
            noise_extention = '_noise'
        else:
            noise_extention = ''
        
        if cfg.settings.probmatching:
            probmatching_extention = '_probmatch'
        else:
            probmatching_extention = ''
        
        blended_file = blended_path + f'/Blended_forecast_{date_str}_step_min_{cfg.settings.timestep_interval}_len_{cfg.settings.timesteps}_ens_dgmr_{cfg.settings.n_ens_members_dgmr}_ens_{cfg.settings.n_ens_members}.npy'
        blended_file_weights = blended_path + f'/Blended_forecast_{date_str}_step_min_{cfg.settings.timestep_interval}_len_{cfg.settings.timesteps}_ens_dgmr_{cfg.settings.n_ens_members_dgmr}_ens_{cfg.settings.n_ens_members}{multi_extention}{noise_extention}{probmatching_extention}{custom_weights_extention}{cfg.settings.custom_extention}_weights.npy'

    # endregion
    
    
    print(f"----------------------------------------------------------------------------------------------")
    print(f"Running blending code for date == {date}")
    print(f"----------------------------------------------------------------------------------------------")
    start_time = time.time()
    
    # region - radar
    print(f"----------------------------------------------------------------------------------------------")
    print(f"1. Radar data - download if needed and import:")
    print(f"----------------------------------------------------------------------------------------------")
    # Download
    radar_files = run_download_radar(date=date, gauge_adjusted=cfg.settings.gauge_adjusted, input_dir=radar_path)
    # Import
    # start to unpack the radar files
    # load radar files
    importer_kwargs = {"accutime": 5, "qty": "ACRR", "pixelsize": 1000.0}

    # Read the data from the archive
    try:
        importer = io.get_method("knmi_hdf5", "importer")
        R, _, metadata_radar = io.read_timeseries(radar_files, importer, **importer_kwargs)
    except:
        if verb:
            print('Input data unreadable. Abort script.')

    # Convert to rain rate
    R, metadata_radar = conversion.to_rainrate(R, metadata_radar)
    del metadata_radar['transform']
    R[np.isnan(R)] = 0

    R_xr = convert_input_to_xarray_dataset(R,None,metadata_radar,radar_files[1][0], 5 * 60)
    if verb:
        print(f"Radar time range: {R_xr['time'].values[0]} to {R_xr['time'].values[-1]}")
        
    # endregion
        
    # region - destine
    print(f"----------------------------------------------------------------------------------------------")
    print(f"2a. DestinE data - download if needed and preprocess:")
    print(f"----------------------------------------------------------------------------------------------")
    # Check if files exist, if they are correct, and otherwise download and preprocess
    if os.path.exists(destine_file_preprocessed):
        try:
            if verb:
                print(f"pre-processed file found for DestinE: {destine_file_preprocessed}, checking if it is correct...")
            # Open the file:
            destinE_nlgrid = xr.open_dataset(destine_file_preprocessed, engine="netcdf4")#KW: adde , engine="netcdf4"
            
            # slice the timesteps to match the radar timesteps:
            #time_slice = slice(pd.to_datetime(R_xr['time'][-1].values) + timedelta(minutes=5), pd.to_datetime(R_xr['time'][-1].values) + timedelta(minutes = int(cfg.settings.timestep_interval) * int(cfg.settings.timesteps))+ timedelta(minutes=5) )
            #destine_nlgrid_blend = destinE_nlgrid.sel(time=time_slice)           
            #print(f'destine_nlgrid_blend time range: {time_slice}')
            #len_nwp=len(destine_nlgrid_blend['time'])
            destine_nlgrid_blend = validate_destine_file(destinE_nlgrid, R_xr, cfg)
            
            # check it has the correct number of time steps
            #assert len_nwp == (int(cfg.settings.timesteps) + 1), f'Not the correct length timesteps in destine file, length is currently: {len_nwp} while it should be {(int(cfg.settings.timesteps) + 1)} '    
        except AssertionError:
            # if there is the wrong number of time steps compared to radar, then do the preprocessing again
            destine_nlgrid_blend = pre_process_destine_data(
                files = destine_file_original,
                timestep_interval=cfg.settings.timestep_interval,
                timesteps=cfg.settings.timesteps,
                date_str=date_str_day,
                date=date,
                radar_path=radar_path,
                destineE_datafolder=destine_path,
                historical_destine=cfg.settings.historical_destine,
                radar_xr=R_xr)

    #if the preprocessed file does not exist:
    # check if the original file exists. if it does then do the preprocessing:
    elif os.path.exists(destine_file_original):
        try:
            print(f"Pre-processing the original DestinE file: {destine_file_original} since no pre-processed file was found")
            #param = '228' # KW: change the param to match the filename used in the function
            #extention = 'grib'
            # checking the original files have the correct times:
            destinE_nlgrid = xr.open_dataset(destine_file_original)
            destinE_nlgrid = ensure_destine_time_dim(destinE_nlgrid)
            # Validate timestamps BEFORE preprocessing
            _ = validate_destine_file(destinE_nlgrid, R_xr, cfg)

            if verb:
                print("Original file valid. Proceeding with preprocessing...")
        
        except AssertionError as e:
            print(f"Original file invalid: {e}. Redownloading...")
            # If original file is wrong → redownload
            download_destine(
                date=date,
                historical=cfg.settings.historical_destine,
                destine_path=destine_path,
                destine_file_original=destine_file_original,
                destine_file_original_nc=destine_file_original_nc,
                param=cfg.settings.param,
                extention=cfg.settings.destine_extension
            )
          
        # Always preprocess (after validation or redownload)  
        destine_nlgrid_blend = pre_process_destine_data(
            files=destine_file_original,
            timestep_interval=cfg.settings.timestep_interval,
            timesteps=cfg.settings.timesteps, 
            date_str=date_str_day,
            date=date,
            radar_path=radar_path,
            destineE_datafolder=destine_path,
            historical_destine=cfg.settings.historical_destine,
            radar_xr=R_xr
            )
    # if both the original and preprocessed files are missing - download and then preprocess
    else:
        #TODO change folder structure in this as well
        #files  = download_destinE_data.download_destine(date, destineE_datafolder) ## original script but download_destineE_data isn't needed since it is imported above
        print(f"Downloading destine data:")
        files  = download_destine(
            date = date,
            historical = cfg.settings.historical_destine,
            destine_path = destine_path,
            destine_file_original = destine_file_original,
            destine_file_original_nc = destine_file_original_nc,
            param = cfg.settings.param,
            extention = cfg.settings.destine_extension) #KW: removed "destineE_datafolder" and added historical_destine as argument
        #param = '228_regrid_nl'
        #extention = 'nc'
        destine_nlgrid_blend = pre_process_destine_data(destine_file_original, cfg.settings.timestep_interval, cfg.settings.timesteps, date_str_day, date, radar_path, destine_path, cfg.settings.historical_destine, R_xr)

    # endregion
    
   


if __name__ == "__main__":
    main()