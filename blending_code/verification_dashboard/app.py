"""
PLotting of datasets in DestE project

adding a dataset:
1) add dataset name + doiscription + color to 'config.py' file dictionaries 
2) add loading logic to 'data_access.py' load_saved_datasets function

"""

import streamlit as st
import pandas as pd
from datetime import datetime
import xarray as xr
import sys
import numpy as np


sys.path.insert(0, "/home/joep/git/wi-research/p111_ecmwf_destine/verification_dashboard")

from data_access import load_saved_datasets, load_weights
from metrics import domain_average, calculate_crps_by_leadtime
from plotting import plot_hotspot, plot_domain_average, plot_crps, plot_cases, plot_cascade_levels, plot_multiple_cascade_blocks
from data_access import available_blend_dates
from config import BASE, DATASETS, DATASET_TO_VARIANT

#Example date only for testing of functions:
date_str = '2024052411'
date = datetime.strptime(date_str, "%Y%m%d%H")
year = date.year
month = date.month
hour = date.hour
selected_keys = ['blend_ifs', 'blend_optimised', "blend_optimised_upper_cascade"]
timestep_interval = 30
timesteps = 24


if "available_dates" not in st.session_state:
    st.session_state.available_dates = None
    

st.set_page_config(layout="wide")
st.title("Blended forecast verification")
st.subheader("Datasets to display")


selected_keys = []
for key, label in DATASETS.items():
    if st.checkbox(label[0]):
        selected_keys.append(key)

save_fig = False
if st.checkbox("Save figure"):
    save_fig = True


plot_type = st.selectbox("plot type", ['domain average precipitation', 'hotspot precipitation', 'Precipitation maps', 'CRPS score', 'weights comparison'])
dates_all = []
if st.button("Generate available dates"):
    if plot_type !='weights comparison':
        required_variants = []
        for key in selected_keys:
            if key == 'blend_optimised' or key == 'blend_optimised_upper_cascade': #skipping this becasue we can generate it in the data access function. 
                continue #make something here that checks for optuna study objects in /home/joep/
            if DATASET_TO_VARIANT.get(key,[]) != []:
                required_variants.append(DATASET_TO_VARIANT.get(key,[]))
            
        st.session_state.available_dates = available_blend_dates(
            BASE / "blended_forecast",
            required_variants
        )
    else:
        required_variants = []
        for key in selected_keys:
            if DATASET_TO_VARIANT.get(key,[]) != []:
                required_variants.append(DATASET_TO_VARIANT.get(key,[]) + '_weights')
            
                dates = available_blend_dates(
                        BASE / "blended_forecast",
                        required_variants
                    )
                dates_all.append(dates)
                print(dates_all)
        common_dates = set(dates_all[0]).intersection(*dates_all[1:])
        st.session_state.available_dates = common_dates



if plot_type !='CRPS score':

    if st.session_state.available_dates:
        date_str = st.selectbox(
            "Available dates",
            st.session_state.available_dates,
            key="date_selector"
        )

        date = datetime.strptime(date_str, "%Y%m%d%H")
        timestep_interval = st.selectbox("timstep interval", [30])
        timesteps = st.selectbox("timsteps", [24])
        
        # year = date.year
        # month = date.month
        # hour = date.hour
        # date_str_day = date_str[:-2]
        # hour_ifs = round_down(hour)
        # date_str_ifs = date_str_day + str(hour_ifs)


    
if selected_keys != st.session_state.get("last_selected_keys"):
    st.session_state.available_dates = None
    st.session_state.last_selected_keys = selected_keys


if st.button("Generate plot"):
    if plot_type == 'domain average precipitation':
        with st.spinner("Loading data..."):
            
            series, metadata_radar = load_saved_datasets(selected_keys, date,timestep_interval, timesteps)
            series_average = {}

            for key in selected_keys:
                series_average[key] = domain_average(series[key])
           
            if len(series_average[key].shape) == 2:
                time_axis = pd.date_range(
                    start=date, periods=series_average[key].shape[1], freq=f"{timestep_interval}min"
                )
            elif len(series_average[key].shape) == 1:
                time_axis = pd.date_range(
                    start=date, periods=series_average[key].shape[0], freq=f"{timestep_interval}min"
                )

        fig = plot_domain_average(time_axis, series_average)
        st.pyplot(fig)

        if save_fig == True:
            fig.savefig(
                f"/srv/data/nas/project_data/p111_ecmwf_destine/verification/"
                f"{plot_type}_{date_str}.png",
                dpi=300,
                bbox_inches="tight",
            )
            print('figure saved')
    

    elif plot_type == 'weights comparison':

        weights_total = []
        for key in selected_keys:
            if key[:5] != 'blend':
                continue

            if key == 'blend_optimised' or key == 'blend_optimised_upper_cascade':
                try:
                    print('trying to load optimised weights for ', key)
                    weights = load_weights(date,key,timestep_interval,timesteps, optimised=True)
                    weights_total.append(weights)
                except:
                    print('no optimised weights found for ', key)
                    weights = load_weights(date,key,timestep_interval,timesteps, optimised=False)
                    weights_total.append(weights)
            else:
                print('trying to load weights for ', key)
                weights = load_weights(date,key,timestep_interval,timesteps, optimised=False)
                weights_total.append(weights)

        if len(weights_total) == 1:

            fig, axes = plot_cascade_levels(data = weights, date_str = date_str, lead_times=None, cascade_distances=None, gamma=weights.item()['GAMMA'], regr_pars=weights.item()['regr_pars'], savefig = save_fig)

        else:
            fig = plot_multiple_cascade_blocks(datasets = weights_total, date_str = date_str, titles=selected_keys, gamma=True, regr_pars=True)
        
        st.pyplot(fig)
        
        if save_fig == True:
            fig.savefig(
                f"/srv/data/nas/project_data/p111_ecmwf_destine/verification/"
                f"{plot_type}_{date_str}.png",
                dpi=300,
                bbox_inches="tight",
            )
            print('figure saved')

    elif plot_type == 'Precipitation maps':
        with st.spinner("Loading data..."):
            
            series, metadata_radar = load_saved_datasets(selected_keys, date,timestep_interval, timesteps)
            print(series.keys())
            ens_number = 0
            #turn this on if you want to identify the highets ensemble member
            # hotspot_idx = series['blend_ifs_noise'].sum(axis=(1, 2, 3)).argmax()
            # ens_number = hotspot_idx
            # print(ens_number)
            fig = plot_cases(series['radar'], metadata_radar, series, ens_number, timesteps, timestep_interval)
            st.pyplot(fig)
            
        if save_fig == True:
            fig = plot_cases(series['radar'], metadata_radar, series, ens_number, timesteps, timestep_interval, save_figure=True)
            fig.savefig(
                f"/srv/data/nas/project_data/p111_ecmwf_destine/verification/"
                f"{plot_type}_{date_str}.png",
                dpi=300,
                bbox_inches="tight",
            )
            print('figure saved')
        

    elif plot_type == 'CRPS score':
        with st.spinner("Loading data..."):

            series_total = {}
            for date_str in st.session_state.available_dates:
                date = datetime.strptime(date_str, "%Y%m%d%H")
                
                series, metadata_radar = load_saved_datasets(selected_keys, date,timestep_interval, timesteps)
                for key in selected_keys:
                    if key == 'radar':
                        continue
                    #first calculate CRPS based
                    new = calculate_crps_by_leadtime(series['radar'], series[key])
                    print(key)
                    print(new)
                    if key not in series_total:
                        series_total[key] = new
                    elif len(series_total[key].shape) == 1:
                        series_total[key] = np.concatenate([[series_total[key]],[new]])
                    else:
                        series_total[key] = np.concatenate([series_total[key],[new]])


            
            series_mean = {}
            for key in selected_keys:
                if key == 'radar':
                    continue
                series_mean[key] = np.mean(series_total[key], axis = 0)
            
            lead_times = np.arange(0.5, ((len(series_mean[key])+1) * timestep_interval) / 60, timestep_interval / 60 ) 
        
        fig = plot_crps(lead_times, series_mean)
        st.pyplot(fig)
        

        if save_fig == True:
            fig.savefig(
                f"/srv/data/nas/project_data/p111_ecmwf_destine/verification/"
                f"{plot_type}_{date_str}.png",
                dpi=300,
                bbox_inches="tight",
            )
            print('figure saved')
    
    elif plot_type == 'hotspot precipitation':
        with st.spinner("Loading data..."):
            
            series, metadata_radar = load_saved_datasets(selected_keys, date,timestep_interval, timesteps)
            series_hotspot = {}
            

            # concatenate all cases in time
            radar_all = series['radar']   # (T, y, x)

            # boolean mask of "wet enough" timesteps
            wet = radar_all >= 2.0   # threshold in mm (adjust if units differ)

            # fraction of wet timesteps per grid cell
            wet_fraction = wet.mean(axis=0)   # shape (y, x)

            # find cells that satisfy your persistence criterion
            mask = wet_fraction >= 0.3

            if not np.any(mask):
                raise ValueError("No grid cell has >=5 mm in at least 50% of timesteps.")

            # among those, pick the one with the highest fraction
            fraction_masked = np.where(mask, wet_fraction, -np.inf)

            y, x = np.unravel_index(
                np.argmax(fraction_masked),
                wet_fraction.shape
            )

            y,x = 300,300
            
            for key in selected_keys:
                if len(series[key].shape) == 4:
                    series_hotspot[key] = series[key][:, :, y, x]
                elif len(series[key].shape) ==3:
                    series_hotspot[key] = series[key][:, y, x]
           
            if len(series_hotspot[key].shape) == 2:
                time_axis = pd.date_range(
                    start=date, periods=series_hotspot[key].shape[1], freq=f"{timestep_interval}min"
                )
            elif len(series_hotspot[key].shape) == 1:
                time_axis = pd.date_range(
                    start=date, periods=series_hotspot[key].shape[0], freq=f"{timestep_interval}min"
                )
        hotspot_idx = (y, x)
        fig = plot_hotspot(time_axis, series_hotspot, hotspot_idx)
        st.pyplot(fig)

        if save_fig == True:
            fig.savefig(
                f"/srv/data/nas/project_data/p111_ecmwf_destine/verification/"
                f"{plot_type}_{date_str}.png",
                dpi=300,
                bbox_inches="tight",
            )
            print('figure saved')
            
