# plotting.py
import matplotlib.pyplot as plt
import pandas as pd
from config import DATASETS
import pysteps.visualization.precipfields as pf

def plot_hotspot(time_axis, series_dict, hotspot_idx):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time_axis, series_dict["radar"], "k-", lw=2, label="Radar")

    for key, value in series_dict.items():
        if key == "radar":
            continue
        if len(value.shape) == 1:
            ax.plot(time_axis, value, DATASETS[key][1], alpha=0.7, lw=2, label=DATASETS[key][0])
        elif len(value.shape) == 2:
            for i in range(value.shape[0]):
                ax.plot(time_axis, value[i], DATASETS[key][1], alpha=0.3, label=DATASETS[key][0] if i == 0 else None)

    ax.set_ylabel("Precipitation (mm/h)")
    ax.legend()
    ax.set_title('Precipitation time series at hotspot index {}'.format(hotspot_idx))
    ax.grid(alpha=0.3)

    return fig

import matplotlib.pyplot as plt

def plot_domain_average(time_axis, series):
    print(series)
    fig, ax = plt.subplots(figsize=(12, 6))

    for key, value in series.items():
        
        if len(value.shape) == 2:
            for i in range(value.shape[0]):
                    if i ==0:
                        ax.plot(time_axis, value[i], DATASETS[key][1], alpha=0.3, label=DATASETS[key][0])
                    else:
                        ax.plot(time_axis, value[i], DATASETS[key][1], alpha=0.3)

        elif len(value.shape) == 1:
            ax.plot(time_axis, value, DATASETS[key][1], lw=2, label=DATASETS[key][0])


    ax.set_xlabel("Time")
    ax.set_ylabel("Domain avg precipitation rate (mm/h)")
    ax.set_title("Domain-average precipitation")
    ax.legend()
    ax.grid(alpha=0.3)

    return fig

def plot_settings_weights(settings_weights):

    import numpy as np
import matplotlib.pyplot as plt

# #Load datasets outside of functions
# destine_datafolder = '/srv/data/nas/project_data/p111_ecmwf_destine'
# weights = np.load(destine_datafolder + '/blended_forecast/Blended_forecast_2024052411_step_min_30_len_24_ens_dgmr_5_ens_20_IFS_probmatch_optimised_weights_weights.npy')
# weights = np.load('/srv/data/nas/project_data/p111_ecmwf_destine/blended_forecast/2024/5/Blended_forecast_2024052411_step_min_30_len_24_ens_dgmr_5_ens_20_IFS_probmatch_optimised_weights_weights.npy', allow_pickle=True)


def plot_cascade_levels(
    data,
    date_str,
    lead_times=None,
    cascade_distances=None,
    gamma=None,          # shape (6, 2)
    regr_pars=None,      # shape (2, 6)
    axes=None,
    add_legend=True,
    savefig=True
):

    """
    Parameters
    ----------
    data : np.ndarray-like (dict wrapped)
        data.item()['weights'] must have shape (3, 6, 25)
    axes : list of matplotlib.axes.Axes or None
        If None, a new figure is created (default behaviour)
    add_legend : bool
        Only add legend when creating a standalone figure
    savefig : bool
        Disable saving when embedding in larger figures
    """
    
    destine_datafolder = '/srv/data/nas/project_data/p111_ecmwf_destine'
    data_weights = data.item()['weights']
    assert data_weights.shape == (3, 6, 25), "Expected data shape (3, 6, 25)"

    if lead_times is None:
        lead_times = np.arange(data_weights.shape[2])

    if cascade_distances is None:
        cascade_distances = range(6)

    colors = ['tab:blue', 'goldenrod', 'tab:orange']
    linestyles = [':', '--', '-']
    labels = ['Nowcast', 'NWP', 'Noise']

    created_figure = False

    n_extra = int(gamma is not None) + int(regr_pars is not None)
    n_plots = 6 + n_extra

    if axes is None:
        ncols = 2
        nrows = int(np.ceil(n_plots / ncols))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(10, 4 * nrows), sharex=False
        )
        axes = axes.flatten()
        created_figure = True
    else:
        fig = axes[0].figure

    handles = []

    for c in range(6):
        ax = axes[c]

        for line_type in range(3):
            line, = ax.plot(
                lead_times,
                data_weights[line_type, c, :],
                color=colors[line_type],
                linestyle=linestyles[line_type],
                linewidth=2.0,
                label=labels[line_type]
            )

            if c == 0:
                handles.append(line)
        if c ==0:
            ax.set_title(f'Cascade level 0')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Weight')
        else:
            ax.set_title(f'Cascade level {c}')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)

    plot_idx = 6

    if gamma is not None:
        assert gamma.shape == (6, 2)

        ax = axes[plot_idx]
        x = np.arange(6)

        ax.plot(x, gamma[:, 0], marker='o', label='Gamma 1')
        ax.plot(x, gamma[:, 1], marker='o', label='Gamma 2')

        ax.set_title('GAMMA per cascade level')
        ax.set_xlabel('Cascade level')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        plot_idx += 1

    if regr_pars is not None:
        assert regr_pars.shape == (2, 6)

        ax = axes[plot_idx]
        x = np.arange(6)

        ax.plot(x, regr_pars[0, :], marker='o', label='Regr pars set 1')
        ax.plot(x, regr_pars[1, :], marker='o', label='Regr pars set 2')

        ax.set_title('Regression parameters per cascade level')
        ax.set_xlabel('Cascade level')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)


    axes[0].set_ylim(-0.4, 1)


    if add_legend and created_figure:
        fig.legend(
            handles,
            labels,
            loc='upper center',
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1)
        )
        # fig.subplots_adjust(top=0.82)

        fig.supxlabel('Lead time')
        fig.supylabel('Value')

    if created_figure and savefig:

        plt.savefig(
            destine_datafolder +
            f'/verification/weights_{date_str}_cascade_levels.png',
            dpi=300
        )

    return fig, axes



def plot_multiple_cascade_blocks(datasets, titles, date_str, ncols=2, gamma=True, regr_pars=True):
    nblocks = len(datasets)
    
    nrows = int(np.ceil(nblocks / ncols))

    fig = plt.figure(figsize=(20, 14 * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.35)

    if gamma and regr_pars:
        columns = 4
    else:
        columns = 3
    for i, (data, title) in enumerate(zip(datasets, titles)):
        r, c = divmod(i, ncols)


        inner = outer[r, c].subgridspec(columns, 2,hspace=0.45)
        if gamma:
            gamma_values = data.item()['GAMMA']
        if regr_pars:
            regr_pars_values = data.item()['regr_pars']
        
        axes = [fig.add_subplot(inner[j, k]) for j in range(columns) for k in range(2)]

        plot_cascade_levels(
            data,
            date_str,
            axes=axes,
            add_legend=True,
            gamma=gamma_values,
            regr_pars=regr_pars_values,
            savefig=False
        )

        # axes[0].set_title(title, fontsize=13, pad=28)

    return fig


def plot_crps(time_axis, series):
    
    fig, ax = plt.subplots(figsize=(12, 6))

    for key, value in series.items(): 
        print(key, value)
        ax.plot(time_axis, value, DATASETS[key][1], alpha=0.3, label=DATASETS[key][0])
        


    ax.set_xlabel('Lead Time (hours)')
    ax.set_ylabel('Average CRPS')
    ax.set_title("Average CRPS by Lead Time (minutes)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig


import matplotlib.pyplot as plt
from pysteps.visualization import plot_precip_field
import pysteps
import numpy as np

from matplotlib.gridspec import GridSpec

def plot_cases(radar_data, metadata_radar, forecasts_dict, ens_number, timesteps, timestep_interval, save_figure=False):
    #plot cases for checking 

    
    
    proj4str = metadata_radar["projection"]
    crs = pysteps.visualization.utils.proj4_to_cartopy(proj4str)
    n_figures = len(forecasts_dict)
    lead_times = np.arange(0, timesteps * timestep_interval, timestep_interval)
    if save_figure:
        lead_times_short = lead_times[::4] #+ [lead_times[-1:]]
        lead_times = lead_times_short
    print(lead_times)
    n_leadtimes = len(lead_times)
    
    n_cols = n_figures + 1
    n_rows = n_leadtimes

    fig = plt.figure( figsize=(3.5 * n_figures + 1.5, 3.5 * n_rows), frameon=False)
    
    gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        wspace=0.02,
        hspace=0.05
        )
    
    for n, leadtime in enumerate(lead_times):

        ax_label = fig.add_subplot(gs[n, 0])
        ax_label.axis("off")

        ax_label.text(
            0.95,
            0.5,
            f"T+{leadtime} min",
            ha="right",
            va="center",
            fontsize=30,
            transform=ax_label.transAxes
        )

        # Nowcast with blending into NWP
        if leadtime == 0:
            title_text = f"Radar"
        else: 
            title_text = ""

        ax = fig.add_subplot(gs[n, 1], projection=crs)
        ax = plot_precip_field(
            radar_data[int(leadtime / timestep_interval), :, :],
            geodata=metadata_radar,
            ax=ax,
            title=title_text,
            axis="off",
            colorscale="STEPS-NL",
            colorbar=False,
        )        
        ax.set_aspect("auto")
       
        plot_window = 1 
        for key,value in forecasts_dict.items():
            if key == 'radar':
                continue
            
            if leadtime == 0:
                title_text = f"{key}"
            else: 
                title_text = ""
            
            plot_window += 1

            # currently I have this setup only for blend_ifs_noise to identify the hotspot ensemble member. make setting for all?
            if key == 'blend_ifs_noise':
                ax = fig.add_subplot(gs[n, plot_window], projection=crs)
                ax = plot_precip_field(
                    value[ens_number, int(leadtime / timestep_interval), :, :],
                    geodata=metadata_radar,
                    ax=ax,
                    title=title_text,
                    axis="off",
                    colorscale="STEPS-NL",
                    colorbar=False,
                )
                ax.set_aspect("auto") 
            
            else:

                if len(value.shape)==4:
                    ax = fig.add_subplot(gs[n, plot_window], projection=crs)
                    ax = plot_precip_field(
                        value[0, int(leadtime / timestep_interval), :, :],
                        geodata=metadata_radar,
                        ax=ax,
                        title=title_text,
                        axis="off",
                        colorscale="STEPS-NL",
                        colorbar=False,
                    )
                    ax.set_aspect("auto")

                elif len(value.shape)==3:
                    ax = fig.add_subplot(gs[n, plot_window], projection=crs)
                    ax = plot_precip_field(
                        value[int(leadtime / timestep_interval), :, :],
                        geodata=metadata_radar,
                        ax=ax,
                        title=title_text,
                        axis="off",
                        colorscale="STEPS-NL",
                        colorbar=False,
                    )
                    ax.set_aspect("auto")
    fig.subplots_adjust(
    left=0.01,
    right=0.99,
    bottom=0.01,
    top=0.95,
    wspace=0.02,   # horizontal spacing
    hspace=0.05    # vertical spacing
    )
    
    return fig

