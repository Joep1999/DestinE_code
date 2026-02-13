
from config import BASE, DATASETS, DATASET_TO_VARIANT, BLENDING_CONFIG
import sys
sys.path.insert(0, "/home/joep/git/wi-research/p111_ecmwf_destine/")
import blending_operational

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

def load_optimised_weights_blending(study_name, storage):

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
    custom_weights = {
    "GAMMA": GAMMA,
    "regr_pars": regr_pars,
    "clim_cor_values": clim_cor_values,
    }
    return custom_weights

def run_blending_for_visualisation(date):
    date_str = date.strftime('%Y%m%d%H')
    year = date.year
    month = date.month
    study_name = f"pysteps_blending_{date_str}"
    storage_weights = f'/home/joep/optuna_blending_study_{date_str}.db'
    custom_weights =  load_optimised_weights_blending(study_name, storage_weights)
    precip_forecast_mm, radar_precip_mm, nwp_precip_mm, weights = blending_operational.run_blending_operational(date, BLENDING_CONFIG['historical_destine'], BLENDING_CONFIG['knmi_input_dir'], BLENDING_CONFIG['destineE_datafolder'], BLENDING_CONFIG['timesteps'], BLENDING_CONFIG['timestep_interval'], BLENDING_CONFIG['n_ens_members'], BLENDING_CONFIG['n_ens_members_dgmr'], BLENDING_CONFIG['weights_method'], custom_weights, BLENDING_CONFIG['return_weights'], BLENDING_CONFIG['re_do_blending'], BLENDING_CONFIG['multi_model'], BLENDING_CONFIG['nostepsnoise'],BLENDING_CONFIG['probmatching'])
    blend_ifs_nonoise = open_memmap(blend_path(year, month, date_str, DATASET_TO_VARIANT['blend_optimsed'], timestep_interval, timesteps))
