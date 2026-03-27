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
from nowcast_blend.blending.dgmr_for_blending import run_dgmr_ensemble
from nowcast_blend.blending.blending import blending_function


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
            print("Using real time date")
        date_orig = datetime.now()
        #KW: round it down to the closest 30-minutes because I get errors subsetting the DestinEtimes when I'm using any other minutes
        date = floor_to_30min(date_orig)
        yr = date.year
        mnth = date.month
        date_str = date.strftime('%Y%m%d%H')
        date_str_day = date_str[:-2]
    else:
        if verb:
            print("Using date from the config")
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
    # ==================================================
    # 2ai. ENSURE ORIGINAL FILE IS VALID
    # ==================================================
    # if the original file doesn't exist then we need to download it:
    if not os.path.exists(destine_file_original):
        if verb:
            print(f"Original file missing so we will download it: {destine_file_original}")
        
        download_destine(
            date=date,
            historical=cfg.settings.historical_destine,
            destine_path=destine_path,
            destine_file_original=destine_file_original,
            destine_file_original_nc=destine_file_original_nc,
            param=cfg.settings.param,
            extention=cfg.settings.destine_extension
        )
    # if the original file does exist then we need to open it, convert steps to times if needed, and then check it has the right times.   
    if os.path.exists(destine_file_original):
        try:
            if verb:
                print(f"Original file exists. Validating it: {destine_file_original}")

            ds_orig = xr.open_dataset(destine_file_original)
            ds_orig = ensure_destine_time_dim(ds_orig)
            ds_orig = validate_destine_time_range(ds_orig, R_xr, cfg)
            #ds_orig = validate_destine_file(ds_orig, R_xr, cfg)

            #if verb:
            #    print("Original file is valid. Go check the preprocessed files")
            #break  # exit loop
    # if it doesn't have the right times then we need to remove the files and download them again:
        except AssertionError as e:
            print(f"Original file invalid: {e}")
            
            print("Deleting original file/s...")
            destine_file_original_idx = destine_file_original + ".idx"
            if os.path.exists(destine_file_original):
                os.remove(destine_file_original)

            if os.path.exists(destine_file_original_idx):
                os.remove(destine_file_original_idx)

            print("Redownloading original file...")
            download_destine(
                date=date,
                historical=cfg.settings.historical_destine,
                destine_path=destine_path,
                destine_file_original=destine_file_original,
                destine_file_original_nc=destine_file_original_nc,
                param=cfg.settings.param,
                extention=cfg.settings.destine_extension
            ) 

    # ==================================================
    # 2aii. ENSURE PREPROCESSED FILE IS VALID
    # ==================================================
    # if the post-processed file exists then we need to check the times are right
    if os.path.exists(destine_file_preprocessed):
        try:
            if verb:
                print("Validating preprocessed file...")

            ds_pre = xr.open_dataset(destine_file_preprocessed, engine="netcdf4")
            ds_pre = validate_destine_file(ds_pre, R_xr, cfg)

            if verb:
                print("Preprocessed file is valid so go to the next steps")
            destine_nlgrid_blend = ds_pre.sel(
                time=slice(
                    pd.to_datetime(R_xr['time'][-1].values) + timedelta(minutes=5),
                    pd.to_datetime(R_xr['time'][-1].values)
                    + timedelta(minutes=int(cfg.settings.timestep_interval) * int(cfg.settings.timesteps) + 5)
                )
            )
            return destine_nlgrid_blend

        except AssertionError as e:
            print(f"Preprocessed file invalid: {e}")
            print("Reprocessing...")

    else:
        if verb:
            print("Preprocessed file missing then we need to do the preprocessing...")

    # ==================================================
    # 2aiii. PREPROCESS 
    # ==================================================
    # something goes wrong here. It works but we should only post-process the destine file if needed.
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
    
    
    # endregion
    
    # region - ifs
    # ifs - download and preprocessed
    if cfg.settings.multi_model:
        print(f"----------------------------------------------------------------------------------------------")
        print(f"2b. IFS data - download if needed and preprocess:")
        print(f"----------------------------------------------------------------------------------------------")
        print("Put in the ifs stuff")
        
    # endregion
    
    # region - dgmr
    print(f"----------------------------------------------------------------------------------------------")
    print(f"3. DGMR...    ")
    print(f"----------------------------------------------------------------------------------------------")
    #TODO no option to change the timlength yet for DGMR (now only works if it accounts to 6 hours)
    
    if not os.path.exists(dgmr_file):
        # if DGMR hasn't been run, then run it and save it
        DGMR_det_long = run_dgmr_ensemble(R_xr.precip_intensity.values, ens_members = int(cfg.settings.n_ens_members_dgmr), forecast_length = int((int(cfg.settings.timestep_interval) * int(cfg.settings.timesteps)) / 90))
        np.save(dgmr_file, DGMR_det_long, allow_pickle=True)
    else:
        # if dgmr has been run, then just import it from file
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
    print(f"----------------------------------------------------------------------------------------------")
    print(f"4. Organise metadata and data...    ")
    print(f"----------------------------------------------------------------------------------------------")
    # organise the metadata
    destine_nlgrid_blend_metadata = metadata_radar
    destine_nlgrid_blend_metadata['timestamps'] = destine_nlgrid_blend.time.values
    destine_nlgrid_blend_metadata['institution'] = destine_nlgrid_blend.institution
    destine_nlgrid_blend_metadata['unit'] = 'mm/h'
    destine_nlgrid_blend_metadata['threshold'] = float(0.1)
    metadata_radar['transform'] = None
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
        print(f"Uncomment and fix the below line")
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
    print(f"----------------------------------------------------------------------------------------------")
    print(f"5. Do the blending...")
    print(f"----------------------------------------------------------------------------------------------")
    blending_function(
        blended_file = blended_file, 
        blended_file_weights=blended_file_weights,
        config=cfg,
        radar_precip=radar_precip,
        nwp_precip=nwp_precip,
        DGMR_det_db=DGMR_det_db,
        radar_metadata=radar_metadata,
        nwp_metadata=nwp_metadata)
    
    print((time.time() - start_time)/60, "minutes")

            
    
    
    



if __name__ == "__main__":
    main()