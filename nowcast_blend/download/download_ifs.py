import logging
log = logging.getLogger(__name__)

from ecmwfapi import ECMWFService


def download_ifs(ifs_init_time, ifs_file_original, param):
    
    
    server = ECMWFService("mars")
    request = {
        "area": "80/-20/-50/180",
        "class": "od",
        "date": ifs_init_time[:8],
        "number": "1/to/50",  # ensemble members 1-50
        "expver": "1", #
        "grid": "0.05/0.05",
        "levtype": "sfc",
        "param": param,
        "step": "0/to/36/by/1", #time_range_acc,
        "stream": "enfo",
        #"target": local_file_today,
        "time": f"{ifs_init_time[8:10]}:00:00",
        "type": "pf",
    }

    # ---------------------------------------------------------------------
    # DOWNLOAD grib
    # ---------------------------------------------------------------------
    try:
        server.execute(request, ifs_file_original)
        print(f"Download completed for {ifs_init_time}: {ifs_file_original}")
    except Exception as e:
        print(f"script terminated because of exception: {e}")
        raise