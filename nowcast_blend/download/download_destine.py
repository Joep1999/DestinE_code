from datetime import datetime, timedelta
import copy
import os
import re
from polytope.api import Client

import logging
log = logging.getLogger(__name__)

def check_destine_available(date, destine_folder):
    time_range_acc = "/".join(f"{i}-{i+1}" for i in range(2))
    date_str = date.strftime('%Y%m%d')  
    param = "228"
    request = {
                "class": "d1",
                "expver": "0001",
                "grid":"0.05/0.05",
                "stream": "oper",
                "dataset": "extremes-dt",
                "date": date_str,
                "time": "0000",
                "type": "fc",
                "levtype": "sfc",
                "step": time_range_acc,
                "param": param,
        }  
    log.info(f"Checking if data exists for date == {date}")
    client = Client(
        address="polytope.lumi.apps.dte.destination-earth.eu",
        )
    try:
        client.retrieve("destination-earth", request, output_file = "tmp_destine.grib") 
        return True
    except Exception as e:
        log.warning(f"Data not available for {date}: {e}")
        return False

def download_destine(date, historical, destine_file_original, destine_file_original_nc, param, extention):
    #import os
    #from polytope.api import Client
    #from configparser import ConfigParser
    #config = ConfigParser()
    #config.read('/usr/people/whan/ResearchDataLab/floodMIND/running_rt/config_paths.ini')

    #input_dir = config['PATHS']['input_project']

    #destineE_datafolder = input_dir + config['PATHS']['input_general'] + config['PATHS']['input_destineE']
    #input_dir_proj = config['PATHS']['input_general'] + config['PATHS']['input_radar_gauge_adj_knmi']
    # You can pass your email and apikey here, or put them in ~/.polytopeapirc (as JSON)
    # You can also set POLYTOPE_USER_EMAIL and POLYTOPE_USER_KEY in your environment

    # Make sure the whole timerange is downloaded
    #time_range = "/".join(str(i) for i in range(97))
    #time_range_acc = "/".join(f"{i}-{i+1}" for i in range(24)) #KW: 24 hours is not enough when we have eg nowcast at 1pm for 12 hours
    time_range_acc = "/".join(f"{i}-{i+1}" for i in range(73))

    # KW: this isn't needed because I've already passed todays date as an argument to the function.
    # Use for operational download
    # date = datetime.now() - timedelta(days=1)

    #Uncomment for historical download
    # dates = np.arange(20250722, 20250732, 1)
    # for date_numb in dates:

        # date = datetime.strptime(str(date_numb),'%Y%m%d')    

    date_str = date.strftime('%Y%m%d')    
    yr = date.year
    mnth = date.month

    #Make sure local folder exists
    #destine_path_yearmonth = destine_path + '{}/{}/'.format(yr,str(mnth).zfill(2)) # KW: added zfill to make sure the month has 2 digits to match the format in other functions
    #for folder in [destine_path_yearmonth]:
    #    if not os.path.exists(folder):
    #        os.makedirs(folder)

    if historical == False:
        log.info(f"Downloading new forecast for {date_str} from DestinE Extremes DT when historical == False -- from polytope API")
        client = Client(
        address="polytope.lumi.apps.dte.destination-earth.eu",
        )

        # Optionally revoke previous requests
        client.revoke("all")
        ##Use to download data over Netherlands for convective rainrate and large scale rainrate (228219/228218)
        # 288 for accumulated
        param = "228"
        if param == '228':
            time_range = time_range_acc
        request = {
                "class": "d1",
                "expver": "0001",
                "grid":"0.05/0.05",
                "stream": "oper",
                "dataset": "extremes-dt",
                "date": date_str,
                "time": "0000",
                "type": "fc",
                "levtype": "sfc",
                "step": time_range_acc,
                "param": param,
        }
    else:
        #from ecmwfapi import ECMWFDataServer ## original script but didn't work
        #client = ECMWFDataServer()
        
        from ecmwfapi import ECMWFService
        server = ECMWFService("mars")

        # KW: why two requests here for to get extremesDT from ECMWF? Doesn't the second overwrite the first?
        request={
            "area": "80/-20/20/30",
            "class": "rd",
            "dataset": "research",
            "date": "2023-10-11/to/2023-10-29",
            "expver": "i4ql",
            "grid": "0.05/0.05",
            "levtype": "sfc",
            "param": "tprate",
            "step": "0/1/2/3/4",
            "stream": "oper",
            "target": "output.grib",
            "time": "00:00:00",
            "type": "fc"
        }
        request={
            "area": "80/-20/20/30",
            'class': 'od',
            'date': '2023-10-01',
            'expver': 1,
            'levtype': 'sfc',
            'number': '1/2/3',
            'param': '228.128',
            'step': '0/1/2/3/4/5/6/7/8/9/10/11/12',
            'stream': 'enfo',
            'time': '00:00:00',
            'type': 'pf',
            'target': 'output.grib',
        }
    # if 'feature' in request:
    #     extention = '.covjson'
    # else:
    #     extention = '.grib'
    
    #    The data will be saved in the current working directory
    #destine_file_date  = destine_path_yearmonth + f'DestinE_ExtremesDT_{date_str}_{param}{extention}'
    #destine_file_date_regrid_nc = destine_path_yearmonth + f'DestinE_ExtremesDT_{date_str}_{param}_regrid_nl.nc'

    #local_file_today = local_folder_today + 'DestinE_ExtremesDT_20231101_218.228-219.228-228.128.grib'
    
    #KW: original line was "files = client...". This fails when the extremesDT data isn't available yet
    #files = client.retrieve("destination-earth", request, output_file= local_file_today)
    req = copy.deepcopy(request)
    try:
        log.info(f"Trying to download data for {destine_file_original}...")
        files = client.retrieve("destination-earth", request, output_file=destine_file_original)
        log.info(f"Success for {destine_file_original}")
        # return files, destine_file_original

    except Exception as e:
        #KW: this is very silly. Instead of just taking the file for yesterdays date, it downloads yesterdays data and stores it as todays file.
        log.info(f"Failed for today ({request['date']}): {e}")

        # fallback to previous day
        prev_date = (datetime.strptime(request["date"], "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        req["date"] = prev_date
        
        #fix lead times
        #req["step"] = make_time_range(48) #KW: 24 makes problems
        log.info(f"Trying previous day: {prev_date}, step == {req['step']}")
        #req["step"] = "/".join(f"{i}-{i+1}" for i in range(48))
        #log.info(f"Trying previous day: {prev_date}, step == {req['step']}")
        #log.info(f"steps = {req['step']}")

        log.info(f"Trying previous day: {prev_date}")
        destine_file_original = re.sub(r"\d{8}", prev_date, destine_file_original)
        destine_file_original_nc = re.sub(r"\d{8}", prev_date, destine_file_original_nc)
        if not os.path.exists(destine_file_original):
            log.info(f"prev_date file does not exist either so we download it: {destine_file_original}")
            files = client.retrieve("destination-earth", req, output_file=destine_file_original) # filename is wrong
        else:
            log.info(f"prev_date file already exists so we just use that")
    
    
    
    #server.execute(request, local_file_today)

    if extention == '.grib':

        from cdo import Cdo
        cdo = Cdo()
        log.info(cdo.version())
        cdo.run(
            "-P 4 -f nc sellonlatbox,-1,11.87,48.4,56.4 "
            "-remapnn,r5120x2560 -setgridtype,regular "
            f"{destine_file_original} {destine_file_original_nc}"
        )

    return destine_file_original, destine_file_original_nc

#to test the download
# destine_file_original, destine_file_original_nc = download_destine(
#                 date=datetime(2026,4,1,10),
#                 historical=False,
#                 destine_file_original='/srv/data/nas/project_data/p111_ecmwf_destine/ExtremesDT/2026/04/DestinE_ExtremesDT_2026040110_228.grib',
#                 destine_file_original_nc='/srv/data/nas/project_data/p111_ecmwf_destine/ExtremesDT/2026/04/DestinE_ExtremesDT_2026040110_228_regrid_nl.nc',
#                 param=228,
#                 extention='.grib'
#             )

import hydra
from omegaconf import DictConfig
import logging

@hydra.main(version_base=None, config_path="../../configs", config_name="nowcast-blend.yaml")
def main(cfg: DictConfig):  
    
    destine_file_original, destine_file_original_nc = download_destine(
                date=datetime(2026,7,1,10),
                historical=False,
                destine_file_original='/srv/data/nas/project_data/p111_ecmwf_destine/ExtremesDT/2026/04/DestinE_ExtremesDT_2026040110_228.grib',
                destine_file_original_nc='/srv/data/nas/project_data/p111_ecmwf_destine/ExtremesDT/2026/04/DestinE_ExtremesDT_2026040110_228_regrid_nl.nc',
                param=228,
                extention='.grib'
            )
    
    check_destine_available(datetime(2026,7,1,10), destine_folder=cfg.paths.input_project + '.grib')

if __name__ == "__main__":
    main()