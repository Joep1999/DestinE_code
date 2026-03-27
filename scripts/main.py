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


import pysteps
from pysteps import io
from pysteps.utils import conversion, transformation

# utils
from nowcast_blend.utils.utils import floor_to_30min, validate_destine_file, closest_ecmwf_available
from nowcast_blend.utils.formatting import convert_npy_to_nc_file
#download
from nowcast_blend.download.download_radar import run_download_radar
from nowcast_blend.download.download_destine import download_destine, check_destine_available
from nowcast_blend.download.download_ifs import download_ifs
# preprocess
from nowcast_blend.preprocess.preprocess_radar import convert_input_to_xarray_dataset
from nowcast_blend.preprocess.preprocess_destine import pre_process_destine_data
from nowcast_blend.preprocess.preprocess_ifs import pre_process_ifs_data

#nowcast
from nowcast_blend.nowcast.dgmr_for_blending import run_dgmr_ensemble
#blend
from nowcast_blend.blending.blending import blending_function



# how to import the config interactively:
#with hydra.initialize(version_base=None, config_path="configs"):
#    cfg: DictConfig = hydra.compose(config_name="nowcast-blend.yaml")


import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="nowcast-blend.yaml")
def main(cfg: DictConfig):     
    # import config info from config file:
    #config = ConfigParser(inline_comment_prefixes=("#", ";"))
    #config.read('./configs/nowcast-blend.ini')
    verb = cfg.settings.verbose
    
    # define the date: either from RT or from the config file
    if cfg.settings.rt:
        if verb:
            log.info(f"Using real time date as config == {cfg.settings.rt}")
        date_orig = datetime.now()
        #KW: round it down to the closest 30-minutes because I get errors subsetting the DestinEtimes when I'm using any other minutes
        date = floor_to_30min(date_orig)
        yr = date.year
        mnth = date.month
        date_str = date.strftime('%Y%m%d%H')
        date_str_day = date_str[:-2]
    else:
        if verb:
            log.info(f"Using date from the config as config == {cfg.settings.rt}")
        yr = cfg.rundate.year
        mnth = cfg.rundate.month
        day = cfg.rundate.day
        hour = cfg.rundate.hour
        date = datetime(yr, mnth, day, hour)
        date_str = date.strftime('%Y%m%d%H')
        date_str_day = date_str[:-2]
       
    ifs_init_time = closest_ecmwf_available(date)
       
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
    
    # ifs paths:
    ifs_path = base_input_dir + cfg.paths.input_ifs + str(yr) + '/' + format(str(mnth).zfill(2)) + '/'
    # make sure destine folder exists
    for folder in [ifs_path]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    ifs_file_original = ifs_path + f"IFS_{date_str}_init{ifs_init_time}_{cfg.settings.param}.grib"
    ifs_file_preprocessed = ifs_path  + f'/IFS_{date_str}_init{ifs_init_time}_{cfg.settings.param}_hres_interp_nlgrid_{cfg.settings.timestep_interval}_{cfg.settings.timesteps}.nc' 
    
    
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
    blended_path =  cfg.paths.input_project + cfg.paths.output + cfg.paths.input_blended + str(yr) + '/' + format(str(mnth).zfill(2))
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
    
    
    log.info(f"----------------------------------------------------------------------------------------------")
    log.info(f"Running blending code for date == {date}")
    log.info(f"----------------------------------------------------------------------------------------------")
    start_time = time.time()
    
    # region - radar
    log.info(f"----------------------------------------------------------------------------------------------")
    log.info(f"1. Radar data - download if needed and import:")
    log.info(f"----------------------------------------------------------------------------------------------")
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
            log.info('Input data unreadable. Abort script.')

    # Convert to rain rate
    R, metadata_radar = conversion.to_rainrate(R, metadata_radar)
    del metadata_radar['transform']
    R[np.isnan(R)] = 0

    R_xr = convert_input_to_xarray_dataset(R,None,metadata_radar,radar_files[1][0], 5 * 60)
    if verb:
        log.info(f"Radar time range: {R_xr['time'].values[0]} to {R_xr['time'].values[-1]}")
        
    # endregion
        
    # region - destine
    log.info(f"----------------------------------------------------------------------------------------------")
    log.info(f"2a. DestinE data - download if needed and preprocess:")
    log.info(f"----------------------------------------------------------------------------------------------")
    # Check if destine is available for todays date:
    log.info(f"Checking if destine is available for date == {date}")
    destine_avail = check_destine_available(date)
    log.info(f"destine_avail == {destine_avail}")
    if not destine_avail:
        destine_date = (date - timedelta(days=1)).strftime("%Y%m%d")
        log.info(f"Using date == {destine_date}")
        
    
    
    # Check if files exist, if they are correct, and otherwise download and preprocess
    if os.path.exists(destine_file_preprocessed):
        try:
            if verb:
                log.info(f"pre-processed file found for DestinE: {destine_file_preprocessed}, checking if it is correct...")
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
    elif os.path.exists(destine_file_original):
        try:
            log.info(f"Pre-processing the original DestinE file: {destine_file_original} since no pre-processed file was found")
            #param = '228' # KW: change the param to match the filename used in the function
            #extention = 'grib'
            # checking the original files have the correct times:
            destinE_nlgrid = xr.open_dataset(destine_file_original, engine="netcdf4")
            # Validate timestamps BEFORE preprocessing
            _ = validate_destine_file(destinE_nlgrid, R_xr, cfg)

            if verb:
                log.info("Original file valid. Proceeding with preprocessing...")
        
        except AssertionError as e:
            log.info(f"Original file invalid: {e}. Redownloading...")
            # If original file is wrong → redownload
            destine_file_original, destine_file_original_nc = download_destine(
                date=date,
                historical=cfg.settings.historical_destine,
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
        log.info(f"Downloading destine data:")
        destine_file_original, destine_file_original_nc  = download_destine(
            date = date,
            historical = cfg.settings.historical_destine,
            destine_file_original = destine_file_original,
            destine_file_original_nc = destine_file_original_nc,
            param = cfg.settings.param,
            extention = cfg.settings.destine_extension) #KW: removed "destineE_datafolder" and added historical_destine as argument
        #param = '228_regrid_nl'
        #extention = 'nc'
        destine_nlgrid_blend = pre_process_destine_data(destine_file_original, cfg.settings.timestep_interval, cfg.settings.timesteps, date_str_day, date, radar_path, destine_path, cfg.settings.historical_destine, R_xr)

    # endregion
    
    # region - ifs
    # ifs - download and preprocessed
    if cfg.settings.multi_model:
        log.info(f"----------------------------------------------------------------------------------------------")
        log.info(f"2b. IFS data - download if needed and preprocess:")
        log.info(f"----------------------------------------------------------------------------------------------")
        log.info("Put in the ifs stuff")
        if not os.path.exists(ifs_file_original):
            log.info(f"downloading ifs file: {ifs_file_original}")
            download_ifs(ifs_init_time = ifs_init_time, ifs_file_original = ifs_file_original, param = cfg.settings.param)
        # preprocesses:
        if not os.path.exists(ifs_file_preprocessed):
            pre_process_ifs_data(ifs_file_original, ifs_file_preprocessed, date, cfg.settings.timestep_interval, cfg.settings.timesteps, knmi_input_dir, radar_xr)
            
    
    
    
    # endregion
    
    # region - dgmr
    log.info(f"----------------------------------------------------------------------------------------------")
    log.info(f"3. DGMR...    ")
    log.info(f"----------------------------------------------------------------------------------------------")
    #TODO no option to change the timlength yet for DGMR (now only works if it accounts to 6 hours)
    
    if not os.path.exists(dgmr_file):
        log.info(f"DGMR file is missing so we have to make the nowcast: {dgmr_file}")
        # if DGMR hasn't been run, then run it and save it
        DGMR_det_long = run_dgmr_ensemble(R_xr.precip_intensity.values, ens_members = int(cfg.settings.n_ens_members_dgmr), forecast_length = int((int(cfg.settings.timestep_interval) * int(cfg.settings.timesteps)) / 90))
        np.save(dgmr_file, DGMR_det_long, allow_pickle=True)
    else:
        # if dgmr has been run, then just import it from file
        log.info(f"Importing DGMR file: {dgmr_file}")
        DGMR_det_long = np.load(dgmr_file, allow_pickle=True)

    # Select only the relevant time steps from the DGMR output
    if int(cfg.settings.timestep_interval) !=5:
        step = int(cfg.settings.timestep_interval) // 5
        DGMR_det = DGMR_det_long[:,::step]
    else:
        DGMR_det = DGMR_det_long

    # check we have the right number of times
    assert len(DGMR_det[0]) == (int(cfg.settings.timesteps) + 1), f'length of DGMR output is not the same as the timesteps value, len is {len(DGMR_det[0])}!'
    
    # not used currently, but here to check if times are correct
    new_times_DGMR = pd.date_range(R_xr['time'][-1].values, pd.to_datetime(R_xr['time'][-1].values) + timedelta(minutes = 5 * int(cfg.settings.timesteps) * (int(cfg.settings.timestep_interval) / 5) ), freq=f"{cfg.settings.timestep_interval}min")
    
    # endregion
    
    # region - metadata
    log.info(f"----------------------------------------------------------------------------------------------")
    log.info(f"4. Organise metadata and data...    ")
    log.info(f"----------------------------------------------------------------------------------------------")
    # organise the metadata
    destine_nlgrid_blend_metadata = metadata_radar
    destine_nlgrid_blend_metadata['timestamps'] = destine_nlgrid_blend.time.values
    destine_nlgrid_blend_metadata['institution'] = destine_nlgrid_blend.institution
    destine_nlgrid_blend_metadata['unit'] = 'mm/h'
    destine_nlgrid_blend_metadata['threshold'] = float(0.1)
    metadata_radar['transform'] = None
    metadata_DGMR = metadata_radar
    # Log-transform the data
    metadata_radar['timestamps'] = destine_nlgrid_blend_metadata['timestamps']
    DGMR_det_db, metadata_radar_db = transformation.dB_transform(
        DGMR_det, metadata_radar, threshold=0.1, zerovalue=-15.0
    )

    if DGMR_det_db.ndim == 3:
        DGMR_det_db = DGMR_det_db[None, :]

    converter = pysteps.utils.get_method("mm/h")
    radar_precip, metadata_radar = converter(R_xr.precip_intensity.values, metadata_radar)

    if cfg.settings.multi_model  == True: 
        log.info(f"Uncomment and fix the below line")
        #destinE_nlgrid_blend_val, destinE_nlgrid_blend_metadata = converter(IFS_ExtremesDT_blend.tp.values, destinE_nlgrid_blend_metadata)
    else:
        destine_nlgrid_blend_val, destine_nlgrid_blend_metadata = converter(destine_nlgrid_blend.tp.values, destine_nlgrid_blend_metadata)

    # Threshold the data
    radar_precip[radar_precip < 0.1] = 0.0
    destine_nlgrid_blend_val[destine_nlgrid_blend_val < 0.1] = 0.0

    # transform the data to dB
    transformer = pysteps.utils.get_method("dB")
    radar_precip, radar_metadata = transformer(radar_precip, metadata_radar, threshold=0.1)
    nwp_precip, nwp_metadata = transformer(destine_nlgrid_blend_val, destine_nlgrid_blend_metadata, threshold=0.1)

    # r_nwp has to be four dimentional (n_models, time, y, x).
    # If we only use one model:
    if nwp_precip.ndim == 3:
        nwp_precip = nwp_precip[None, :]
        
    # endregion
    
    # region - blending
    log.info(f"----------------------------------------------------------------------------------------------")
    log.info(f"5. Do the blending...")
    log.info(f"----------------------------------------------------------------------------------------------")
    blending_function(
        blended_file = blended_file, 
        blended_file_weights=blended_file_weights,
        config=cfg,
        radar_precip=radar_precip,
        nwp_precip=nwp_precip,
        DGMR_det_db=DGMR_det_db,
        radar_metadata=radar_metadata,
        nwp_metadata=nwp_metadata)
    
    # endregion
    
    log.info(f"----------------------------------------------------------------------------------------------")
    log.info(f"6. write to netcdf...")
    log.info(f"----------------------------------------------------------------------------------------------")
    convert_npy_to_nc_file(blended_file, dgmr_file, destine_nlgrid_blend_metadata, metadata_DGMR)

    
    log.info((time.time() - start_time)/60, "minutes")

            
    
    
    



if __name__ == "__main__":
    main()