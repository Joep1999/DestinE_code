
import joblib
import numpy as np


import logging
log = logging.getLogger(__name__)


from datetime import datetime
import pandas as pd
import scoringrules as sr
from pathlib import Path
import numpy as np
from datetime import datetime
import pandas as pd
import scoringrules as sr

def domain_average(arr):
    return arr.mean(axis=(-2, -1))


def calculate_crps_by_leadtime(radar_images, rainfall_images):
    
    crps_score = sr.crps_ensemble(radar_images, rainfall_images, m_axis = 0)
    crps_score_mean = np.nanmean(crps_score, axis = (0,1))
    
    return crps_score_mean


def build_training_data(date_str, series_nwp_forecast, series_nowcast_forecast ):
    features = []
    date = datetime.strptime(date_str, "%Y%m%d%H")
    # --- Load NWP (no -1 hour shift) ---
    # if multi_model:
    #     key_nwp = 'destine_ifs'
    # else:
    #     key_nwp = 'destine'

    # selected_keys = [key_nwp]
    # series_nwp, metadata_radar_nwp = load_saved_datasets(
    #     selected_keys,
    #     date,
    #     timestep_interval,
    #     timesteps
    # )
    # # Use full forecast without excluding any timestep
    # series_nwp_forecast = series_nwp[key_nwp]
    series_average = {}
    # # --- Load nowcast (unchanged datetime, no slicing) ---
    # selected_keys = ['nowcast']
    # series_nowcast, metadata_radar_nowcast = load_saved_datasets(
    #     selected_keys,
    #     date,
    #     timestep_interval,
    #     timesteps
    # )
    # Use full nowcast without excluding last timestep
    # series_nowcast_forecast = series_nowcast['nowcast']
    # --- Domain averages ---
    series_average['nwp'] = domain_average(series_nwp_forecast)
    series_average["nowcast"] = domain_average(series_nowcast_forecast)
    # --- Extract first timestep (shape: n_ensembles,) ---
    # nwp_t0 = series_average[key_nwp][:, 0]      # 5 members
    # nowcast_t0 = series_average["nowcast"][:, 0]      # 4 members
    # # 0 --- Ensemble mean difference (robust comparison) ---
    # mean_diff_t0 = nwp_t0.mean() - nowcast_t0.mean()
    # features.append(mean_diff_t0)
    # print(len(features))
    # # # --- Feature construction ---
    selected_keys = ['nwp', 'nowcast']
    # # # 1. first 3 hours accumulation difference
    # features.append(series_average["destine_ifs"][:, :3].sum() - series_average["nowcast"][:, :3].sum())
    # print(len(features))
    # # 2. first timestep  difference
    # features.append(series_average["destine_ifs"][:, 0].mean() - series_average["nowcast"][:, 0].mean())
    # print(len(features))
    #3,4,5,6
    for key_variable in selected_keys:
        log.info(f'calculating features for {key_variable}')
        # Mean and standard deviation over entire series / ensembles
        features.append(series_average[key_variable].mean())
        #features.append(series_average[key_variable].std())
        if series_average[key_variable].ndim == 2:
            mean_for_slope = series_average[key_variable].mean(axis=0)
        else:
            mean_for_slope = series_average[key_variable]
        
        log.info(f'shape of {key_variable} before slope is: {mean_for_slope}')
        features.append(
            np.polyfit(range(len(mean_for_slope)), mean_for_slope, 1)[0]
        )
        log.info(f'shape of {key_variable} after slope is: {np.polyfit(range(len(mean_for_slope)), mean_for_slope, 1)[0]}')
        print(len(features))
    
    radar_images = series_nowcast_forecast[0, 0]
    rainfall_images = series_nwp_forecast[:,0]
    features.append(calculate_crps_by_leadtime(radar_images, rainfall_images))
    return features


def run_machine_learning(destineE_datafolder, date_str, series_nwp_forecast, series_nowcast_forecast ):
    kmeans_number = 9
    BASE = Path(destineE_datafolder)
    prediction_features = build_training_data(date_str, series_nwp_forecast, series_nowcast_forecast)
    filename = BASE / f"RF_model_57_k{kmeans_number}.sav"
    loaded_model = joblib.load(filename)
    prediction_features_array = np.array([prediction_features])
    probs = loaded_model.predict_proba(prediction_features_array)
    predicted_class = np.argmax(probs, axis=1)

    loaded_kmeans_weights = np.load(BASE / f'mean_clusters_k{kmeans_number}.npy', allow_pickle=True)
    predicted_weights = loaded_kmeans_weights.item()[predicted_class[0]]

    return predicted_weights