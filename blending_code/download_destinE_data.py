

def download_destine(date, historical=False):
    import os
    from polytope.api import Client
    from configparser import ConfigParser
    config = ConfigParser()
    config.read('/srv/config/config_scripts.ini')

    input_dir = config['paths']['input_project']

    destineE_datafolder = input_dir + 'p111_ecmwf_destine/'

    input_dir_proj = config['paths']['input_general'] + "knmi_radar_gauge_adj/"
    # You can pass your email and apikey here, or put them in ~/.polytopeapirc (as JSON)
    # You can also set POLYTOPE_USER_EMAIL and POLYTOPE_USER_KEY in your environment

    # Make sure the whole timerange is downloaded
    time_range = "/".join(str(i) for i in range(97))
    time_range_acc = "/".join(f"{i}-{i+1}" for i in range(24))

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
    local_folder_today = destineE_datafolder + '{}/{}/'.format(yr,mnth)
    for folder in [local_folder_today]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    if historical == False:
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
        from ecmwfapi import ECMWFDataServer

        client = ECMWFDataServer()

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
    if 'feature' in request:
        extention = '.covjson'
    else:
        extention = '.grib'
        #    The data will be saved in the current working directory
    local_file_today  = local_folder_today + f'DestinE_ExtremesDT_{date_str}_{param}{extention}'
    local_file_today_regrid_nc = local_folder_today + f'DestinE_ExtremesDT_{date_str}_{param}_regrid_nl.nc'

    local_file_today = local_folder_today + 'DestinE_ExtremesDT_20231101_218.228-219.228-228.128.grib'
    files = client.retrieve("destination-earth", request, output_file= local_file_today)

    if extention == '.grib':

        from cdo import Cdo
        cdo = Cdo()
        print(cdo.version())
        cdo.run(
            "-P 4 -f nc sellonlatbox,-1,11.87,48.4,56.4 "
            "-remapnn,r5120x2560 -setgridtype,regular "
            f"{local_file_today} {local_file_today_regrid_nc}"
        )

    return local_file_today_regrid_nc
