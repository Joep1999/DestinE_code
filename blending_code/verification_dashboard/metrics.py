# metrics.py
import numpy as np

def get_hotspot_indices(radar_precip):
    return np.unravel_index(
        np.argmax(radar_precip.sum(axis=0)),
        radar_precip.shape[1:]
    )

def extract_hotspot_timeseries(radar, extremesdt,ifs, nowcast, blend, blend_ifs, blend_ifs_nonoise):
    y, x = get_hotspot_indices(radar)

    return {
        "radar": radar[:, y, x],
        "destine": extremesdt[ :, y, x],
        "ifs": ifs[:, :, y, x],
        "nowcast": nowcast[:, :, y, x],
        "blend": blend[:, :, y, x],
        "blend_ifs": blend_ifs[:, :, y, x],
        "blend_ifs_nonoise": blend_ifs_nonoise[:, :, y, x],
    }

def domain_average(arr):
    return arr.mean(axis=(-2, -1))

def extract_domain_average(radar, extremesdt, ifs, nowcast, blend, blend_ifs, blend_ifs_nonoise):
    return {
        "radar": radar.mean(axis=(1, 2)),
        "destine": extremesdt.mean(axis=(1, 2)),
        "ifs": ifs.mean(axis=(2, 3)),
        "nowcast": nowcast.mean(axis=(2, 3)),
        "blend": domain_average(blend),
        "blend_ifs": domain_average(blend_ifs),
        "blend_ifs_nonoise": domain_average(blend_ifs_nonoise),
    }

#calculate CRPS over leadtimes ()
import scoringrules as sr

def calculate_crps_by_leadtime(radar_images, rainfall_images):
    
    crps_score = sr.crps_ensemble(radar_images, rainfall_images, m_axis = 0)
    crps_score_mean = np.nanmean(crps_score, axis = (1,2))
    
    return crps_score_mean



    