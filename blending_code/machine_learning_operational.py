
from datetime import datetime
import pandas as pd
import scoringrules as sr
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
import scoringrules as sr
sys.path.insert(0, "/home/joep/git/wi-research/p111_ecmwf_destine/verification_dashboard")
from metrics import domain_average
from config import BASE

# Make the prediction dataset
selected_keys = ['destine_ifs', 'nowcast']
timestep_interval = 30
timesteps = 24


def calculate_crps_by_leadtime(radar_images, rainfall_images):
    
    crps_score = sr.crps_ensemble(radar_images, rainfall_images, m_axis = 0)
    crps_score_mean = np.nanmean(crps_score, axis = (0,1))
    
    return crps_score_mean

def round_down(x):
    rounded =  ((x - 1) // 6) * 6
    return f"{rounded:02d}"

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
    
    if 'nowcast' in selected_keys:
        nowcast = slice_np_time(open_memmap(nowcast_path(year, month, date_str,timestep_interval ,timesteps)), metadata_radar_datetimes, timestep_interval, timesteps)[:,1:] #date_str_hr'
    
    for key in selected_keys:
        series[key] = eval(key)
    
    return series, metadata_radar


def build_training_data(date_str, multi_model = True):
    features = []
    date = datetime.strptime(date_str, "%Y%m%d%H")
    # --- Load NWP (no -1 hour shift) ---
    if multi_model:
        key_nwp = 'destine_ifs'
    else:
        key_nwp = 'destine'

    selected_keys = [key_nwp]
    series_mwp, metadata_radar_nwp = load_saved_datasets(
        selected_keys,
        date,
        timestep_interval,
        timesteps
    )
    # Use full forecast without excluding any timestep
    series_mwp_forecast = series_mwp[key_nwp]
    series_average = {}
    # --- Load nowcast (unchanged datetime, no slicing) ---
    selected_keys = ['nowcast']
    series_nowcast, metadata_radar_nowcast = load_saved_datasets(
        selected_keys,
        date,
        timestep_interval,
        timesteps
    )
    # Use full nowcast without excluding last timestep
    series_nowcast_forecast = series_nowcast['nowcast']
    # --- Domain averages ---
    series_average[key_nwp] = domain_average(series_mwp_forecast)
    series_average["nowcast"] = domain_average(series_nowcast_forecast)
    # --- Extract first timestep (shape: n_ensembles,) ---
    nwp_t0 = series_average[key_nwp][:, 0]      # 5 members
    nowcast_t0 = series_average["nowcast"][:, 0]      # 4 members
    # # 0 --- Ensemble mean difference (robust comparison) ---
    # mean_diff_t0 = nwp_t0.mean() - nowcast_t0.mean()
    # features.append(mean_diff_t0)
    # print(len(features))
    # # # --- Feature construction ---
    selected_keys = [key_nwp, 'nowcast']
    # # # 1. first 3 hours accumulation difference
    # features.append(series_average["destine_ifs"][:, :3].sum() - series_average["nowcast"][:, :3].sum())
    # print(len(features))
    # # 2. first timestep  difference
    # features.append(series_average["destine_ifs"][:, 0].mean() - series_average["nowcast"][:, 0].mean())
    # print(len(features))
    #3,4,5,6
    for key_variable in selected_keys:
        # Mean and standard deviation over entire series / ensembles
        features.append(series_average[key_variable].mean())
        #features.append(series_average[key_variable].std())
        mean_for_slope = series_average[key_variable].mean(axis=0)
        features.append(
            np.polyfit(range(len(mean_for_slope)), mean_for_slope, 1)[0]
        )
        print(len(features))
    
    radar_images = series_nowcast_forecast[0, 0]
    rainfall_images = series_mwp_forecast[:,0]
    features.append(calculate_crps_by_leadtime(radar_images, rainfall_images))
    return features