from pathlib import Path
import numpy as np
import os
import requests
from datetime import datetime, timedelta
from pysteps import io, rcparams
import shutil
from pysteps.utils import conversion
import pandas as pd
import xarray as xr
import optuna
import sys
sys.path.insert(0, "/home/joep/git/wi-research/p111_ecmwf_destine/verification_dashboard")
from config import BASE, DATASETS, DATASET_TO_VARIANT, BLENDING_CONFIG

import sys
sys.path.insert(0, "/home/joep/git/wi-research/p111_ecmwf_destine/")
import blending_operational

GAMMA_base = np.array([
                    [0.99805, 0.9933],
                    [0.9925,  0.9752],
                    [0.9776, 0.923],
                    [0.9297,  0.750],
                    [0.796,   0.367],
                    [0.482,   0.069],
                ])

regr_pars_base = np.array([
        [130.0, 165.0, 120.0, 55.0, 50.0, 15.0],
        [155.0, 220.0, 200.0, 75.0, 1e5,  1e5],
    ])

clim_cor_values_base = np.array([0.848, 0.537, 0.237, 0.065, 0.02, 0.0044])


def blend_path(year, month, date_str, variant, timestep_interval, timesteps):
    folder = BASE / "blended_forecast" / str(year) / str(month)
    

    if variant.startswith('blend_optimised'):
        study_name = f"pysteps_blending_{date_str}"
        storage_weights = f'sqlite:///optuna_blending_study.db'
        settings_best_blend = load_settings_weights_blending(study_name = study_name, storage = storage_weights)
        try:
            noise = settings_best_blend['use_noise']
        except:
            noise=True
        probmatching = settings_best_blend['use_probmatching']
        suffix = DATASET_TO_VARIANT[variant]

        if probmatching:
            suffix ='_IFS_probmatch' + str(suffix)[4:]

        if noise:
            suffix ='_IFS_noise' + str(suffix)[4:]
    else:
        suffix = DATASET_TO_VARIANT[variant]

    return folder / (
        f"Blended_forecast_{date_str}"
        f"_step_min_{timestep_interval}_len_{timesteps}"
        f"_ens_dgmr_5_ens_20{suffix}.npy"
    )


def blend_path_pysteps(year, month, date_str, variant, timestep_interval, timesteps):
    folder = BASE / "blended_forecast" / str(year)/ str(month)
    suffix = DATASET_TO_VARIANT[variant]

    return folder / (
        f"Blended_forecast_{date_str}"
        f"_step_min_{timestep_interval}_len_{timesteps}"
        f"{suffix}.npy"
    )

def nowcast_path(year, month, date_str, timestep_interval, timesteps):
    folder = BASE / "DGMR" / str(year) / str(month)
    return folder / (
        f"DGMR_{date_str}"
        f"_step_min_{timestep_interval}_len_{timesteps}_ens_5.npy"
    )

def ifs_path(year, month, date_str, timestep_interval, timesteps):
    folder = BASE / "IFS" / str(year).zfill(2) / str(month).zfill(2) / "pre-processed"
    return folder / (
        f"IFS_{date_str}"
        f"_hres_interp_nlgrid_{timestep_interval}_{timesteps}.nc"
    )

def extremesdt_path(year, month, date_str,timestep_interval, timesteps):
    folder = BASE / "ExtremesDT" / str(year).zfill(2) / str(month).zfill(2) / "pre-processed"
    return folder / (
        f"DestinE_ExtremesDT_{date_str}"
        f"_218.228-219.228-228.128_hres_interp_nlgrid_{timestep_interval}_{timesteps}.nc"
    )



def list_blend_runs(year, month):
    path = BASE / "blended_forecast" / str(year) / str(month)
    return sorted(path.glob("Blended_forecast_*.npy"))

def open_memmap(path):
    return np.load(path, mmap_mode="r")

def download_radar_knmi_check(gauge_adjusted,start_date, input_dir, time_interval, timesteps):
    before_start_date = start_date + timedelta(minutes=-10)
    if gauge_adjusted:
        url = 'https://api.dataplatform.knmi.nl/open-data/v1/datasets/nl_rdr_data_rtcor_5m/versions/1.0/files'
        startfile = before_start_date.strftime('RAD_NL25_RAC_RT_%Y%m%d%H%M.h5')
    else:
        url = 'https://api.dataplatform.knmi.nl/open-data/datasets/radar_reflectivity_composites/versions/2.0/files'
        startfile = before_start_date.strftime('RAD_NL25_PCP_NA_%Y%m%d%H00.h5')


    api_key = '5e554e19274a9600012a3eb10174be35b75442a7a5e2ba066642a279'

    interval_factor = time_interval // 5

    file_list = requests.get(url, headers={'Authorization': api_key},
                            params={"startAfterFilename": startfile,
                                    "maxKeys": int(interval_factor * (timesteps+1))}) #add a buffer of 1 to be sure to have all the files, correct files are selected later

    file_list = file_list.json().get('files')

    # Parse timestamps from filenames
    def extract_datetime(filename):
        # Extract the 14-digit timestamp from the filename
        date_str = filename.split('_')[-1].split('.')[0]
        return datetime.strptime(date_str, "%Y%m%d%H%M")

    # Find the index of the dictionary with that timestamp
    index_start = next(
        (i for i, d in enumerate(file_list) if extract_datetime(d['filename']) == start_date),
        None  # returns None if not found
    )
    
    end_date = start_date+ timedelta(minutes = timesteps * time_interval)
    print('last timestep is: ',end_date)

    # Find the index of the dictionary with that timestamp
    index_end = next(
        (i for i, d in enumerate(file_list) if extract_datetime(d['filename']) == end_date),
        None  # returns None if not found
    )

    file_list = file_list[index_start:index_end+1]


    # Download the last 3 available files
    for ii in range(len(file_list)):
        fn = file_list[ii]['filename']
        print('Downloading: ', fn)
        
        yr = fn[16:20]
        mnth = fn[20:22]
        day = fn[22:24]
        hour = fn[24:26]
        minute = fn[26:28]
        
        local_folder_today = input_dir + '/{}/{}/{}/'.format(yr,mnth,day)

        for folder in [local_folder_today]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        
        direc = local_folder_today
        
        if not os.path.exists(direc+fn):
        
        
            get_file_response = requests.get(url+'/'+fn+'/url', headers={'Authorization': api_key})
            
            download_url = get_file_response.json().get("temporaryDownloadUrl")
            
            dataset_file = requests.get(download_url, stream=True)
        
            if dataset_file.status_code == 200:
                with open(direc+fn, 'wb') as f:
                    dataset_file.raw.decode_content = True
                    shutil.copyfileobj(dataset_file.raw, f)
    
    fns = io.find_by_date(
        start_date, input_dir, "%Y/%m/%d", "RAD_NL25_RAC_RT_%Y%m%d%H%M", "h5", time_interval, num_next_files=timesteps
    )
    assert len(fns[0]) >= timesteps, f'fns does not contain enough radar images for DGMR (needs {timesteps}, contains {len(fns[0])})'
    assert not None in fns[0]
    print(fns[0])
    return fns


def load_radar_avg(date, timestep_interval=30, timesteps=24):

    # wrap your existing radar-loading logic here
    fns = None
    gauge_adjusted = True
    knmi_input_dir = '/srv/data/nas/input_general/knmi_radar_gauge_adj/'

    try:
        if gauge_adjusted:
            fns = io.find_by_date(
                date, knmi_input_dir, "%Y/%m/%d", "RAD_NL25_RAC_RT_%Y%m%d%H%M", "h5", timestep_interval, num_next_files=timesteps
            )
            assert len(fns[0]) >= timesteps, f'fns does not contain enough radar images for DGMR (needs {timesteps}, contains {len(fns[0])})'

            assert not None in fns[0]
        else:
            fns = io.find_by_date(
                date, knmi_input_dir, "%Y/%m/%d", "RAD_NL25_PCP_NA_%Y%m%d%H%M", "h5", timestep_interval, num_next_files=timesteps
            )
            assert len(fns[0]) >= timesteps, f'fns does not contain enough radar images for DGMR (needs {timesteps}, contains {len(fns[0])})'

            assert not None in fns[0]
        if None in fns:
            raise AssertionError("(Part of Radar files not found.")
    except:
        fns = download_radar_knmi_check(gauge_adjusted,date, knmi_input_dir, timestep_interval, timesteps)
        assert len(fns[0]) >= timesteps, f'fns does not contain enough radar images for DGMR (needs {timesteps}, contains {len(fns[0])})'
        assert not None in fns[0]


    # load Radar files
    importer_kwargs = {"accutime": 5, "qty": "ACRR", "pixelsize": 1000.0}

    # Read the data from the archive
    try:
        importer = io.get_method("knmi_hdf5", "importer")
        R, _, metadata_radar = io.read_timeseries(fns, importer, **importer_kwargs)
    except:
        print('Input data unreadable. Abort script.')

    #Convert to rain rate
    R, metadata_radar = conversion.to_rainrate(R, metadata_radar)

    del metadata_radar['transform']
    R[np.isnan(R)] = 0
    metadata_radar['timestamps'] = metadata_radar['timestamps']#[::step]
    # R_selected_images_mean = R_selected_images.mean(axis=(1, 2))[:-1]

    return R[:-1], metadata_radar

def slice_xr_time(nwp_data, metadata_radar, timestep_interval, timesteps):
    time_slice = slice(pd.to_datetime(metadata_radar[0]), pd.to_datetime(metadata_radar[0]) + timedelta(minutes = timestep_interval * timesteps) ) 
    nwp_data_slice = nwp_data.sel(
        time=time_slice).tp.values
    if nwp_data_slice.ndim == 4:
        return np.swapaxes(nwp_data_slice,0,1) #Becasue IFS comes out with strcuture time, ens, lat,lon while we want ens, time, lat , lon
    else:
        return nwp_data_slice
    
def slice_np_time(np_data, metadata_radar, timestep_interval, timesteps):
    
    step = timestep_interval // 5
    np_data_selected = np_data[:,::step]

    return np_data_selected

import re
from pathlib import Path



def available_blend_dates(base, variants):
    """
    variants: list like ["", "_ifs", "_ifs_noise"]
    returns: sorted list of datetime strings YYYYMMDDHH
    """
    date_sets = []
    DATE_RE = re.compile(r"Blended_forecast_(\d{10})")

    for variant in variants:
        suffix = variant

        files = base.glob(f"**/*{suffix}.npy")
        dates = set()

        for f in files:
            m = DATE_RE.search(f.name)
            if m:
                dates.add(m.group(1))

        date_sets.append(dates)

    return sorted(set.intersection(*date_sets))

def round_down(x):
    rounded =  ((x - 1) // 6) * 6
    return f"{rounded:02d}"

import numpy as np

def dict_to_array(d):
    """
    Convert a {(i, j): value} dictionary into a NumPy array.
    """
    # Determine array size
    max_i = max(k[0] for k in d.keys())
    max_j = max(k[1] for k in d.keys())

    arr = np.zeros((max_i + 1, max_j + 1))

    for (i, j), value in d.items():
        arr[i, j] = value

    return arr

    


def load_settings_weights_blending(study_name, storage):
    
    #Load datasets outside of functions
    clim_cor_values_base = np.array([0.848, 0.537, 0.237, 0.065, 0.02, 0.0044])


    # Create or load the study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    best_values = {}

    for key, val in study.best_trial.params.items():
        print(key)
        print(val)
        # key looks like 'GAMMA_0_1', 'regr_pars_1_2', etc.
        parts = key.split("_")
        name = parts[0]
        
        if name == 'use':
            best_values['use_'+ parts[1]] = {}
            best_values['use_' + parts[1]] = val
        else:

            if name not in best_values:
                best_values[name] = {}
        # handle GAMMA_i_j etc
            indices = tuple(map(int, parts[1:]))  # (i, j)
            best_values[name][indices] = val

    GAMMA = dict_to_array(best_values['GAMMA'])
    regr_pars = dict_to_array(best_values['regr'])
    if 'clim_cor' in best_values.keys():
        clim_cor_values = np.array([
        best_values['clim_cor'][i] for i in sorted(best_values['clim_cor'])
        ])
    else:
        clim_cor_values = clim_cor_values_base

    probmatching = best_values['use_probmatching']

    if 'use_noise' in best_values:
        noise = best_values['use_noise']

        custom_weights = {
        "GAMMA": GAMMA,
        "regr_pars": regr_pars,
        "clim_cor_values": clim_cor_values,
        "use_noise": noise,
        "use_probmatching": probmatching
        }

    else:
        custom_weights = {
        "GAMMA": GAMMA,
        "regr_pars": regr_pars,
        "clim_cor_values": clim_cor_values,
        "use_probmatching": probmatching
        }

    return custom_weights


def load_climatological_weights():    
    GAMMA_base = np.array([
                    [0.99805, 0.9933],
                    [0.9925,  0.9752],
                    [0.9776, 0.923],
                    [0.9297,  0.750],
                    [0.796,   0.367],
                    [0.482,   0.069],
                ])

    regr_pars_base = np.array([
        [130.0, 165.0, 120.0, 55.0, 50.0, 15.0],
        [155.0, 220.0, 200.0, 75.0, 1e5,  1e5],
    ])

    clim_cor_values_base = np.array([0.848, 0.537, 0.237, 0.065, 0.02, 0.0044])
    
    custom_weights = {
        "GAMMA": GAMMA_base,
        "regr_pars": regr_pars_base,
        "clim_cor_values": clim_cor_values_base,
    }
    return GAMMA_base, regr_pars_base, clim_cor_values_base

def load_weights(date,key, timestep_interval,timesteps, optimised=True):
    date_str = date.strftime('%Y%m%d%H')
    year = date.year
    month = date.month
    hour = date.hour
    date_str_day = date_str[:-2]
    hour_ifs = round_down(hour)
    date_str_ifs = date_str_day + str(hour_ifs)


    # I have a subfolder with the optimal weights saved as npy files. (consider using it instead of the study. best values?)
    if optimised:
        study_name = f"pysteps_blending_{date_str}"
        storage_weights = f'sqlite:///optuna_blending_study.db'
        settings_best_blend = load_settings_weights_blending(study_name = study_name, storage = storage_weights)
        
        #use_blend ifs here because we need to determine which optimisation was best, and therefore which file to open, so use the be blend_ifs patha nd add nois and probmatch to it if needed.
        path_blend = blend_path(year, month, date_str, 'blend_ifs', timestep_interval, timesteps) 
        try:
            noise = settings_best_blend['use_noise']
        except:
            noise=True
        probmatching = settings_best_blend['use_probmatching']

        if noise:
            path_blend = str(path_blend)[:-4] + "_noise" + ".npy"
        
        if probmatching:
            path_blend = str(path_blend)[:-4] + "_probmatch" + ".npy"

        if optimised:
            path_blend = str(path_blend)[:-4] + "_optimised_weights" + ".npy"
        
        if key == 'blend_optimised_upper_cascade':
            path_blend = str(path_blend)[:-4] + "_upper_cascade" + ".npy"

        #weird that there is an extra 'weights slash'
        # folder = BASE / "blended_forecast" / "weights" / (str(path_blend)[:-4] +  "_weights.npy")
        
        folder = str(path_blend)[:-4] +  "_weights.npy"
        
        if settings_best_blend['GAMMA'].shape == (2,2):
            weights_raw = np.load(folder, allow_pickle=True)
            weights_raw = np.array(weights_raw, dtype =object)
            print('weights_raw: ', weights_raw.item()['GAMMA'], weights_raw.item()['regr_pars'])
            print('settings_best_blend: ', settings_best_blend['GAMMA'], settings_best_blend['regr_pars'])


            if (settings_best_blend['GAMMA'] != weights_raw.item()['GAMMA'][:-4,:]).any() or (settings_best_blend['regr_pars']!= weights_raw.item()['regr_pars'][:,:-4]).any():
                print('settings best blend and saved weights (only upper cascasde) did not match, so re-running blending')
                GAMMA_base, regr_pars_base, clim_cor_values_base = load_climatological_weights()
                custom_weights = {
                                    "GAMMA":np.vstack([ settings_best_blend['GAMMA'], GAMMA_base[-4:]]) ,
                                    "regr_pars": np.hstack([settings_best_blend['regr_pars'], regr_pars_base[:,-4:]]),
                                    "clim_cor_values": clim_cor_values_base,
                                }
                precip_forecast_mm, radar_precip_mm, nwp_precip_mm, weights_raw = blending_operational.run_blending_operational(date, BLENDING_CONFIG['historical_destine'], BLENDING_CONFIG['knmi_input_dir'], BLENDING_CONFIG['destineE_datafolder'], BLENDING_CONFIG['timesteps'], BLENDING_CONFIG['timestep_interval'], BLENDING_CONFIG['n_ens_members'], BLENDING_CONFIG['n_ens_members_dgmr'], BLENDING_CONFIG['weights_method'], custom_weights, BLENDING_CONFIG['return_weights'], BLENDING_CONFIG['re_do_blending'], BLENDING_CONFIG['multi_model'], noise, probmatching)

                weights_raw = np.array(weights_raw, dtype =object)
            
            assert (settings_best_blend['GAMMA'] == weights_raw.item()['GAMMA'][:-4,:]).all() and (settings_best_blend['regr_pars'] == weights_raw.item()['regr_pars'][:,:-4]).all()

        
        elif key != 'blend_optimised_upper_cascade':
            weights_raw = np.load(folder, allow_pickle=True)
            weights_raw = np.array(weights_raw, dtype =object)
            print('weights_raw: ', weights_raw.item()['GAMMA'], weights_raw.item()['regr_pars'])
            print('settings_best_blend: ', settings_best_blend['GAMMA'], settings_best_blend['regr_pars'])


            if (settings_best_blend['GAMMA'] != weights_raw.item()['GAMMA'][:-2,:]).any() or (settings_best_blend['regr_pars']!= weights_raw.item()['regr_pars'][:,:-2]).any():
                print('settings best blend and saved weights did not match, so re-running blending')
                GAMMA_base, regr_pars_base, clim_cor_values_base = load_climatological_weights()
                custom_weights = {
                                    "GAMMA":np.vstack([ settings_best_blend['GAMMA'], GAMMA_base[-2:]]) ,
                                    "regr_pars": np.hstack([settings_best_blend['regr_pars'], regr_pars_base[:,-2:]]),
                                    "clim_cor_values": clim_cor_values_base,
                                }
                precip_forecast_mm, radar_precip_mm, nwp_precip_mm, weights_raw = blending_operational.run_blending_operational(date, BLENDING_CONFIG['historical_destine'], BLENDING_CONFIG['knmi_input_dir'], BLENDING_CONFIG['destineE_datafolder'], BLENDING_CONFIG['timesteps'], BLENDING_CONFIG['timestep_interval'], BLENDING_CONFIG['n_ens_members'], BLENDING_CONFIG['n_ens_members_dgmr'], BLENDING_CONFIG['weights_method'], custom_weights, BLENDING_CONFIG['return_weights'], BLENDING_CONFIG['re_do_blending'], BLENDING_CONFIG['multi_model'], noise, probmatching)

                weights_raw = np.array(weights_raw, dtype =object)
            
            assert (settings_best_blend['GAMMA'] == weights_raw.item()['GAMMA'][:-2,:]).all() and (settings_best_blend['regr_pars'] == weights_raw.item()['regr_pars'][:,:-2]).all()
        
        elif key =='blend_optimised_upper_cascade':
            try:
                weights_raw = np.load(folder, allow_pickle=True)
                weights_raw = np.array(weights_raw, dtype =object)
                print('weights_raw: ', weights_raw.item()['GAMMA'], weights_raw.item()['regr_pars'])
                print('settings_best_blend: ', settings_best_blend['GAMMA'], settings_best_blend['regr_pars'])


                assert (settings_best_blend['GAMMA'][:-2,:] == weights_raw.item()['GAMMA'][:-4,:]).all() 
                assert (settings_best_blend['regr_pars'][:,:-2]== weights_raw.item()['regr_pars'][:,:-4]).all()

            except:
                GAMMA_base, regr_pars_base, clim_cor_values_base = load_climatological_weights()
                custom_weights = {
                                    "GAMMA":np.vstack([ settings_best_blend['GAMMA'][:-2], GAMMA_base[-4:]]) ,
                                    "regr_pars": np.hstack([settings_best_blend['regr_pars'][:,:-2], regr_pars_base[:,-4:]]),
                                    "clim_cor_values": clim_cor_values_base,
                                }
                extention = "_upper_cascade"
                precip_forecast_mm, radar_precip_mm, nwp_precip_mm, weights_raw = blending_operational.run_blending_operational(date, BLENDING_CONFIG['historical_destine'], BLENDING_CONFIG['knmi_input_dir'], BLENDING_CONFIG['destineE_datafolder'], BLENDING_CONFIG['timesteps'], BLENDING_CONFIG['timestep_interval'], BLENDING_CONFIG['n_ens_members'], BLENDING_CONFIG['n_ens_members_dgmr'], BLENDING_CONFIG['weights_method'], custom_weights, BLENDING_CONFIG['return_weights'], BLENDING_CONFIG['re_do_blending'], BLENDING_CONFIG['multi_model'], noise, probmatching,  custom_extention = extention)

                weights_raw = np.array(weights_raw, dtype =object)
                
    else: 
        path_blend = blend_path(year, month, date_str, key, timestep_interval, timesteps) 
        folder = BASE / "blended_forecast" / "weights" / (str(path_blend)[:-4] +  "_weights.npy")
        weights_raw = np.load(folder, allow_pickle=True)


    
    return weights_raw


def load_saved_datasets(selected_keys, date,timestep_interval, timesteps):
    date_str = date.strftime('%Y%m%d%H')
    print(f"Loading datasets for {date_str} with keys {selected_keys}")
    year = date.year
    month = date.month
    hour = date.hour
    date_str_day = date_str[:-2]
    hour_ifs = round_down(hour)
    date_str_ifs = date_str_day + str(hour_ifs)

    series = {}
    radar, metadata_radar = load_radar_avg(date, timestep_interval, timesteps)
    metadata_radar['timestamps'] = metadata_radar['timestamps'][:-1]
    metadata_radar_datetimes = metadata_radar['timestamps']

    if 'destine'in selected_keys:
        destine = slice_xr_time(xr.open_dataset(extremesdt_path(year, month, date_str_day, timestep_interval, timesteps)), metadata_radar_datetimes,timestep_interval, timesteps)[1:]  #date_str short

    if 'ifs' in selected_keys:
        ifs = slice_xr_time(xr.open_dataset(ifs_path(year, month, date_str_ifs,timestep_interval, timesteps)), metadata_radar_datetimes, timestep_interval, timesteps)[:,1:]  #date_str 06

    if 'destine_ifs' in selected_keys:
        ifs = slice_xr_time(xr.open_dataset(ifs_path(year, month, date_str_ifs,timestep_interval, timesteps)), metadata_radar_datetimes, timestep_interval, timesteps)[:,1:]  #date_str 06
        destine = slice_xr_time(xr.open_dataset(extremesdt_path(year, month, date_str_day, timestep_interval, timesteps)), metadata_radar_datetimes,timestep_interval, timesteps)[1:]  #date_str short
        destine_ifs = np.concatenate([ifs, [destine]])
    
    if 'blend' in selected_keys:
        blend = open_memmap(blend_path(year, month, date_str, 'blend', timestep_interval, timesteps))
    
    if 'blend_ifs' in selected_keys:
        blend_ifs = open_memmap(blend_path(year, month, date_str, 'blend_ifs', timestep_interval, timesteps))
    
    if 'blend_ifs_noise' in selected_keys:
        blend_ifs_noise = open_memmap(blend_path(year, month, date_str, 'blend_ifs_noise', timestep_interval, timesteps))
    
    #use load_weights function here to make sure that the run that is loaded in is actually the most optimised one. 
    if 'blend_optimised' in selected_keys:
        try: 
            study_name = f"pysteps_blending_{date_str}"
            storage_weights = f'sqlite:///optuna_blending_study.db'
            settings_best_blend = load_settings_weights_blending(study_name = study_name, storage = storage_weights)
            
            #this asserts  that the forecast thta is saved is actually made using the best weights (weights which it returns are not actually needed)
            weights = load_weights(date,'blend_optimised', timestep_interval,timesteps, optimised=True)

            path_blend = blend_path(year, month, date_str, 'blend_ifs', timestep_interval, timesteps) 
            try:
                noise = settings_best_blend['use_noise']
            except:
                noise=True
            probmatching = settings_best_blend['use_probmatching']

            if noise:
                path_blend = str(path_blend)[:-4] + "_noise" + ".npy"
            
            if probmatching:
                path_blend = str(path_blend)[:-4] + "_probmatch" + ".npy"

            path_blend = str(path_blend)[:-4] + "_optimised_weights" + ".npy"

            blend_optimised = open_memmap(path_blend)
        
        except:
            study_name = f"pysteps_blending_{date_str}"
            storage_weights = f'sqlite:///optuna_blending_study_{date_str}.db'
            custom_weights =  load_settings_weights_blending(study_name, storage_weights)
            precip_forecast_mm, radar_precip_mm, nwp_precip_mm, weights = blending_operational.run_blending_operational(date, BLENDING_CONFIG['historical_destine'], BLENDING_CONFIG['knmi_input_dir'], BLENDING_CONFIG['destineE_datafolder'], BLENDING_CONFIG['timesteps'], BLENDING_CONFIG['timestep_interval'], BLENDING_CONFIG['n_ens_members'], BLENDING_CONFIG['n_ens_members_dgmr'], BLENDING_CONFIG['weights_method'], custom_weights, BLENDING_CONFIG['return_weights'], BLENDING_CONFIG['re_do_blending'], BLENDING_CONFIG['multi_model'], BLENDING_CONFIG['nostepsnoise'],BLENDING_CONFIG['probmatching'])
            blend_optimised = open_memmap(blend_path(year, month, date_str, 'blend_optimised', timestep_interval, timesteps))

    if 'blend_optimised_upper_cascade' in selected_keys:
        
        study_name = f"pysteps_blending_{date_str}"
        storage_weights = f'sqlite:///optuna_blending_study.db'
        settings_best_blend = load_settings_weights_blending(study_name = study_name, storage = storage_weights)
        
        #this asserts  that the forecast thta is saved is actually made using the best weights (weights which it returns are not actually needed)
        weights = load_weights(date,'blend_optimised_upper_cascade', timestep_interval,timesteps, optimised=True)

        path_blend = blend_path(year, month, date_str, 'blend_ifs', timestep_interval, timesteps) 
        try:
            noise = settings_best_blend['use_noise']
        except:
            noise=True
        probmatching = settings_best_blend['use_probmatching']

        if noise:
            path_blend = str(path_blend)[:-4] + "_noise" + ".npy"
        
        if probmatching:
            path_blend = str(path_blend)[:-4] + "_probmatch" + ".npy"

        path_blend = str(path_blend)[:-4] + "_optimised_weights" + ".npy"

        blend_optimised_upper_cascade = open_memmap(path_blend)
    
    if 'blend_optimised_kmeans_estimation' in selected_keys:
        #this asserts  that the forecast that is saved is actually made using the best weights (weights which it returns are not actually needed)
        try:
            load_cluster_dict = np.load(BASE / "machine_learning" / "category_dict_k9.npy", allow_pickle=True)
            cluster = load_cluster_dict.item()[date_str]
            cluster_weights_dict = np.load(BASE / "machine_learning" / f"mean_clusters_k9.npy", allow_pickle=True)
            cluster_weights = cluster_weights_dict.item()[cluster]
            probmatching = cluster_weights['use_probmatching']

            path_blend = blend_path(year, month, date_str, 'blend_ifs', timestep_interval, timesteps) 
            


            #add noise extention here
            path_blend = str(path_blend)[:-4] + "_noise" + ".npy"

            if probmatching:
                path_blend = str(path_blend)[:-4] + "_probmatch" + ".npy"

            path_blend = str(path_blend)[:-4] + "_optimised_weights_kmeans_estimation_k9" + ".npy"
            print('opening_kmeans estimation path:')
            print(path_blend)

            blend_optimised_kmeans_estimation = open_memmap(path_blend)
        
        except:
            print('kmeans estimation blend does not exist, so calculating it using the cluster weights')            
            load_cluster_dict = np.load(BASE / "machine_learning" / "category_dict_k9.npy", allow_pickle=True)
            cluster = load_cluster_dict.item()[date_str]
            cluster_weights_dict = np.load(BASE / "machine_learning" / f"mean_clusters_k9.npy", allow_pickle=True)
            cluster_weights = cluster_weights_dict.item()[cluster]

            custom_weights = {
                            "GAMMA":np.vstack([ cluster_weights['GAMMA'], GAMMA_base[-4:]]) ,
                            "regr_pars": np.hstack([cluster_weights['regr_pars'][:,:-4], regr_pars_base[:,-4:]]),
                            "clim_cor_values": clim_cor_values_base,
                        }
            print('cluster_weights: ', custom_weights['GAMMA'], custom_weights['regr_pars'])
            
            probmatching = cluster_weights['use_probmatching']
            noise = True
            precip_forecast_mm, radar_precip_mm, nwp_precip_mm, weights_raw = blending_operational.run_blending_operational(date, BLENDING_CONFIG['historical_destine'], BLENDING_CONFIG['knmi_input_dir'], BLENDING_CONFIG['destineE_datafolder'], BLENDING_CONFIG['timesteps'], BLENDING_CONFIG['timestep_interval'], BLENDING_CONFIG['n_ens_members'], BLENDING_CONFIG['n_ens_members_dgmr'], BLENDING_CONFIG['weights_method'], custom_weights, BLENDING_CONFIG['return_weights'], BLENDING_CONFIG['re_do_blending'], BLENDING_CONFIG['multi_model'], noise, probmatching, custom_extention= "_kmeans_estimation_k9")

            path_blend = blend_path(year, month, date_str, 'blend_ifs', timestep_interval, timesteps) 

            #add noise extention here
            path_blend = str(path_blend)[:-4] + "_noise" + ".npy"


            if probmatching:
                path_blend = str(path_blend)[:-4] + "_probmatch" + ".npy"


            path_blend = str(path_blend)[:-4] + "_optimised_weights_kmeans_estimation_k9" + ".npy"
            blend_optimised_kmeans_estimation = open_memmap(path_blend)
        
    if 'blend_optimised_machine_learning_prediciton' in selected_keys:
        #this asserts  that the forecast thta is saved is actually made using the best weights (weights which it returns are not actually needed)
        try:
            cluster_weights_dict = np.load(BASE / "machine_learning" / f"mean_clusters_k9.npy", allow_pickle=True)

            prediction_all = np.load(BASE/ "machine_learning" / "predictions_57_k9.npy", allow_pickle=True )
            prediction = prediction_all.item()[date_str]

            cluster_weights = cluster_weights_dict.item()[prediction]

            probmatching = cluster_weights['use_probmatching']

            path_blend = blend_path(year, month, date_str, 'blend_ifs', timestep_interval, timesteps) 

            #add noise extention here
            path_blend = str(path_blend)[:-4] + "_noise" + ".npy"

            if probmatching:
                path_blend = str(path_blend)[:-4] + "_probmatch" + ".npy"

            path_blend = str(path_blend)[:-4] + "_optimised_weights_machine_learning_57_k9" + ".npy"

            blend_optimised_machine_learning_prediciton = open_memmap(path_blend)
        
        except:
            print('machine learning estimated blend does not exist, so calculating it using the predicted cluster weights')
            
            cluster_weights_dict = np.load(BASE / "machine_learning" / f"mean_clusters_k9.npy", allow_pickle=True)
    
            prediction_all = np.load(BASE/ "machine_learning" / "predictions_57_k9.npy", allow_pickle=True )
            prediction = prediction_all.item()[date_str]

            cluster_weights = cluster_weights_dict.item()[prediction]

            custom_weights = {
                            "GAMMA":np.vstack([ cluster_weights['GAMMA'], GAMMA_base[-4:]]) ,
                            "regr_pars": np.hstack([cluster_weights['regr_pars'], regr_pars_base[:,-4:]]),
                            "clim_cor_values": clim_cor_values_base,
                        }
            print('cluster_weights: ', custom_weights['GAMMA'], custom_weights['regr_pars'])
            
            probmatching = cluster_weights['use_probmatching']
            noise = True
            precip_forecast_mm, radar_precip_mm, nwp_precip_mm, weights_raw = blending_operational.run_blending_operational(date, BLENDING_CONFIG['historical_destine'], BLENDING_CONFIG['knmi_input_dir'], BLENDING_CONFIG['destineE_datafolder'], BLENDING_CONFIG['timesteps'], BLENDING_CONFIG['timestep_interval'], BLENDING_CONFIG['n_ens_members'], BLENDING_CONFIG['n_ens_members_dgmr'], BLENDING_CONFIG['weights_method'], custom_weights, BLENDING_CONFIG['return_weights'], BLENDING_CONFIG['re_do_blending'], BLENDING_CONFIG['multi_model'], noise, probmatching, custom_extention= "_machine_learning_57_k9")

            path_blend = blend_path(year, month, date_str, 'blend_ifs', timestep_interval, timesteps) 

            #add noise extention here
            path_blend = str(path_blend)[:-4] + "_noise" + ".npy"

            if probmatching:
                path_blend = str(path_blend)[:-4] + "_probmatch" + ".npy"


            path_blend = str(path_blend)[:-4] + "_optimised_weights_machine_learning_57_k9" + ".npy"
            blend_optimised_machine_learning_prediciton = open_memmap(path_blend)

    if 'pysteps_nowcast' in selected_keys:
            pysteps_nowcast = open_memmap(blend_path_pysteps(year, month, date_str, 'pysteps_nowcast', timestep_interval, timesteps))

    if 'nowcast' in selected_keys:
        nowcast = slice_np_time(open_memmap(nowcast_path(year, month, date_str,timestep_interval ,timesteps)), metadata_radar_datetimes, timestep_interval, timesteps)[:,1:] #date_str_hr'
    
    for key in selected_keys:
        series[key] = eval(key)
    
    return series, metadata_radar


