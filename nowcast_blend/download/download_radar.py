import os
import requests
import shutil
from pysteps import io

from datetime import datetime, timedelta

from nowcast_blend.utils.utils import round_to_5min

import logging
log = logging.getLogger(__name__)


def round_to_5min(dt):
    minutes = dt.minute
    rounded = int(round(minutes / 5.0) * 5)
    diff = rounded - minutes
    return (dt + timedelta(minutes=diff)).replace(second=0, microsecond=0)


def download_radar_knmi(gauge_adjusted,last_hour,date, input_dir):
    if gauge_adjusted:
        url = 'https://api.dataplatform.knmi.nl/open-data/v1/datasets/nl_rdr_data_rtcor_5m/versions/1.0/files'
        lastfile = last_hour.strftime('RAD_NL25_RAC_RT_%Y%m%d%H%M.h5')
    else:
        url = 'https://api.dataplatform.knmi.nl/open-data/datasets/radar_reflectivity_composites/versions/2.0/files'
        lastfile = last_hour.strftime('RAD_NL25_PCP_NA_%Y%m%d%H%M.h5')


    api_key = '5e554e19274a9600012a3eb10174be35b75442a7a5e2ba066642a279'



    file_list = requests.get(url, headers={'Authorization': api_key},
                            params={"startAfterFilename": lastfile,
                                    "maxKeys": 12})

    file_list = file_list.json().get('files')

    # Download the last 3 available files
    for ii in range(len(file_list)-4,len(file_list)):
        fn = file_list[ii]['filename']
        log.info(fn)
        
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
    if gauge_adjusted:
        fns = io.find_by_date(
            date, input_dir, "%Y/%m/%d", "RAD_NL25_RAC_RT_%Y%m%d%H%M", "h5", 5, num_prev_files=3
        )
    else:
        fns = io.find_by_date(
            date, input_dir, "%Y/%m/%d", 'RAD_NL25_PCP_NA_%Y%m%d%H%M', "h5", 5, num_prev_files=3
        )

        assert len(fns[0]) == 4, f'fns does not contain enough radar images for DGMR (needs 4, contains {len(fns[0])})'
    return fns


def run_download_radar(date, gauge_adjusted, input_dir):
    # inset a date and time (in utc)
    last_hour = date + timedelta(hours=-1)
    date_5min = round_to_5min(date) - timedelta(minutes=5) #round to 5 minutes, then substract 5 minutes so that DGMR is initialised on the hour exactly
    #date_5min = round_to_5min(date) #Currently running DGMR on 5 past the hour, but including last radar image -> gives 6hours +5 minutes which is needed for blending
    last_hour_5min = round_to_5min(last_hour) - timedelta(minutes=5) #see reason above for not using this
    # check if data exists, otherwise download
    fns = None
    try:
        if gauge_adjusted:
            fns = io.find_by_date(date_5min, input_dir, "%Y/%m/%d", "RAD_NL25_RAC_RT_%Y%m%d%H%M", "h5", 5, num_prev_files=3)
            assert len(fns[0]) == 4, f'fns does not contain enough radar images for DGMR (needs 4, contains {len(fns[0])})'
        else:
            fns = io.find_by_date(date_5min, input_dir, "%Y/%m/%d", "RAD_NL25_PCP_NA_%Y%m%d%H%M", "h5", 5, num_prev_files=3)
            assert len(fns[0]) == 4, f'fns does not contain enough radar images for DGMR (needs 4, contains {len(fns[0])})' 
        if None in fns:
            raise AssertionError("(Part of Radar files not found.")
        if None in fns[0]:
            raise AssertionError("(Part of Radar files not found.")
        if None in fns[1]:
            raise AssertionError("(Part of Radar files not found.")
    except:
        fns = download_radar_knmi(gauge_adjusted,last_hour_5min,date_5min, input_dir)
        
    return(fns)