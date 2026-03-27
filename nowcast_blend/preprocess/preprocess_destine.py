import os
from datetime import timedelta # pre_process_destinE_data

import xarray as xr
import numpy as np
import pandas as pd

from pysteps.downscaling import rainfarm # pre_process_destinE_data
from scipy.ndimage import map_coordinates # advection_correction_backward
from pysteps import motion # advection_correction_backward

import logging
log = logging.getLogger(__name__)

from cdo import Cdo # pre_process_destinE_data
cdo = Cdo() # pre_process_destinE_data

def advection_correction_backward(R, T=5, t=1):
    """
    R = np.array([qpe_previous, qpe_current])
    T = time between two observations (5 min)
    t = interpolation timestep (1 min)
    """

    # Evaluate advection
    oflow_method = motion.get_method("LK")
    fd_kwargs = {"buffer_mask": 10}  # avoid edge effects
    V = oflow_method(np.log(R), fd_kwargs = fd_kwargs)

    # Perform temporal interpolation
    x, y = np.meshgrid(
        np.arange(R[0].shape[1], dtype=float), np.arange(R[0].shape[0], dtype=float)
    )
    ny, nx = R[0].shape
    n_steps = T // t
    sequence = np.zeros((n_steps, ny, nx)) 

    for idx, i in enumerate(range(t, T + t, t)):
        # pos1 = (y - i / T * V[1], x - i / T * V[0])
        # R1 = map_coordinates(R[0], pos1, order=1)

        pos2 = (y + (T - i) / T * V[1], x + (T - i) / T * V[0])
        R2 = map_coordinates(R[1], pos2, order=1)

        # Blend fields ?? check this?
        sequence[idx, :, :] = R2
        # Rd += (T - i) * R1 + i * R2
    return sequence


def cdo_to_netcdf(date, destinE_data, destinE_data_cut, numerical_data, destineE_datafolder, filename, freq, historical_destine):  
    log.info(f"before any processing: {destinE_data_cut.step.values[0]}, {destinE_data_cut.step.values[-1]}")
    
    if historical_destine == True:
        lat_min, lat_max = destinE_data_cut.latitude.values.min(), destinE_data_cut.latitude.values.max()
        lon_min, lon_max = destinE_data_cut.longitude.values.min(), destinE_data_cut.longitude.values.max()
        new_times = pd.date_range(destinE_data_cut.step.values[0], destinE_data_cut.step.values[-1], freq=freq)
    else: 
        #KW: problem with the coord names. It is latitude not lat
        lat_name = "lat" if "lat" in destinE_data_cut.coords else "latitude"
        lon_name = "lon" if "lon" in destinE_data_cut.coords else "longitude"
        #lat_min, lat_max = destinE_data_cut.lat.values.min(), destinE_data_cut.lat.values.max()
        #lon_min, lon_max = destinE_data_cut.lon.values.min(), destinE_data_cut.lon.values.max()
        lat_min, lat_max = destinE_data_cut[lat_name].values.min(), destinE_data_cut[lat_name].values.max()
        lon_min, lon_max = destinE_data_cut[lon_name].values.min(), destinE_data_cut[lon_name].values.max()
        #KW: problem also with the times
        try:
            # KW: this was the original line, but it fails when the time coordinate is called 'step' instead of 'time'
            new_times = pd.date_range(destinE_data_cut.time.values[0], destinE_data_cut.time.values[-1], freq=freq)
        # KW: this is to make it work with the polytope destine data, which has a 'step' coordinate instead of 'time'
        except Exception:
            try:
                # Fallback: add step (timedelta) to reference time
                #ref_time = destinE_data_cut.time.values
                #start = ref_time + destinE_data_cut.step.values[0]
                #end = ref_time + destinE_data_cut.step.values[-1]
                new_times = pd.date_range(destinE_data_cut.step.values[0], destinE_data_cut.step.values[-1], freq=freq)
            except Exception as e:
                log.info("Error creating new time range:", e)
                new_times = None


    # Interpolate to new time grid
    lat_new = np.linspace(lat_min, lat_max, numerical_data.shape[-2])
    lon_new = np.linspace(lon_min, lon_max, numerical_data.shape[-1])

    if numerical_data.ndim ==4:
        ensemble = [30, 50, 90]
        numerical_data = numerical_data.swapaxes(0,1) #put time first
        destinE_data_radar_scale_xr = xr.DataArray(
            numerical_data,
            dims=("time", "ensemble","lat", "lon"),
            coords={
                "time": new_times,
                "ensemble": ensemble,
                "lat": lat_new,
                "lon": lon_new,
            },
            name=list(destinE_data.data_vars)[-1]  # reuse variable name
        )
    else: 
        destinE_data_radar_scale_xr = xr.DataArray(
            numerical_data,
            dims=("time", "lat", "lon"),
            coords={
                "time": new_times,
                "lat": lat_new,
                "lon": lon_new,
            },
            name=list(destinE_data.data_vars)[-1]  # reuse variable name
        )

    # Copy variable attributes
    destinE_data_radar_scale_xr.attrs = destinE_data[list(destinE_data.data_vars)[0]].attrs

    yr = date.year
    mnth = date.month
    date_str = date.strftime('%Y%m%d')  

    #Make sure local folder exists
    #local_folder_today = destineE_datafolder + '{}/{}/'.format(yr,str(mnth).zfill(2)) # KW: added zfill to make sure the month has 2 digits to match the format in other functions
    #for folder in [local_folder_today]:
    #    if not os.path.exists(folder):
    #        os.makedirs(folder)


    # Wrap in dataset and copy global attrs
    ds_destinE_data_radar_scale_xr = xr.Dataset({destinE_data_radar_scale_xr.name: destinE_data_radar_scale_xr})
    ds_destinE_data_radar_scale_xr.attrs = destinE_data.attrs

    ds_destinE_data_radar_scale_xr['lat'].attrs['units'] = 'degrees_north'
    ds_destinE_data_radar_scale_xr['lon'].attrs['units'] = 'degrees_east'

    ds_destinE_data_radar_scale_xr.to_netcdf(filename )


def pre_process_destine_data(files, timestep_interval, timesteps, date_str, date, radar_path, destineE_datafolder, historical_destine, radar_xr):
    log.info(f'Pre-processing Destine file:',files)
    if date.hour>12:
        
        # KW: do we need this? I don't think so
        #destinE_data = xr.open_mfdataset(
        #    [files, files[:-5] + '_25-40.grib' ],
        #    combine="nested",
        #    concat_dim="step",
        #    engine="cfgrib"
        #)
        destinE_data =  xr.open_dataset(files)
        log.info('DestinE data combined and loaded successfully when date.hour > 12')
    else:
        destinE_data =  xr.open_dataset(files)
        log.info(f'DestinE data loaded successfully from {files}')
    
    destinE_data_np = destinE_data.tp.values
    #TODo make this neat
    param = '228' #'218.228-219.228-228.128' #KW: make the param match the rest
    destinE_data['tp'].attrs = {'long_name': 'Accumulated precipitation', 'units': 'mm/h', 'param': '193.1.0'} #KW: what is this new param about?

    accum_prcp = destinE_data['tp'] * 1000
    
    try: 
        accum_prcp_subset = accum_prcp.sel(latitude=slice(56.4, 48.4),longitude=slice(-1, 11.87)) 
        accum_prcp_subset = accum_prcp_subset.assign_coords(step=[pd.to_datetime( destinE_data['time'].values) + timedelta(hours =i) for i in range(len(destinE_data['step']))])
        precipitation = accum_prcp_subset - accum_prcp_subset.shift({'step': 1})
        precipitation = precipitation.dropna('step', how="all")

    except:
        precipitation = accum_prcp - accum_prcp.shift({'time': 1})
        precipitation = precipitation.dropna('time', how="all")

        # DOWNSCALE DESTINE DATA WITH RAINFARM TO RADAR RESOLUTION
    destinE_data_radar_scale = []
    H, W = precipitation.shape[1:]
    ds_factor = 4
    target_shape = (H * ds_factor, W * ds_factor)
    
    for i in range(precipitation.shape[0]):
        #log.info(f'Processing timestep {i}')
        precipitation_i=precipitation[i].values
        if np.all(precipitation_i == 0) or np.nanstd(precipitation_i) == 0:
            #log.info(f"Skipping timestep {i} (no variability)")
            #destinE_data_radar_scale.append(np.zeros_like(precipitation_i))
            destinE_data_radar_scale.append(np.zeros(target_shape))
            continue
        
        destinE_data_radar_scale.append(rainfarm.downscale(precipitation_i, ds_factor=ds_factor, kernel_type = 'gaussian')) #KW: this fails when it is all zeros
        #log.info(destinE_data_radar_scale[i].shape)

    destinE_data_radar_scale = np.array(destinE_data_radar_scale)

    #TODO: APPLY ADVECTION CORRECTION TO DESTINE DATA -> discuss with kyrie -> use something else?
    all_steps = [destinE_data_radar_scale[0:1]]  # keep first slice as (1, ny, nx)

    for i in range(destinE_data_radar_scale.shape[0] - 1):
        steps = advection_correction_backward(destinE_data_radar_scale[i : i + 2], T=60, t=timestep_interval)
        all_steps.append(steps)
    # Concatenate along time
    destinE_nlgrid_hres_advected = np.concatenate(all_steps, axis=0)

    # Write to netcdf so cdo can use the data
    log.info("starting: cdo_to_netcdf:")
    #log.info(precipitation)
    cdo_to_netcdf(date=date,
                  destinE_data=destinE_data,
                  destinE_data_cut=precipitation,
                  numerical_data=destinE_nlgrid_hres_advected,
                  destineE_datafolder=destineE_datafolder,
                  filename=destineE_datafolder + f'/DestinE_ExtremesDT_{date_str}_{param}_hres_advect_xr_{timestep_interval}_{timesteps}.nc',
                  freq=f"{timestep_interval}min",
                  historical_destine=historical_destine)

    # REGRID DESTINE DATA TO KNMI RADAR GRID
    log.info("starting: cdo.remapnn:")
    cdo.remapnn(
        f"{radar_path}knmi_grid.txt",                  # target grid
        input=destineE_datafolder + f'/DestinE_ExtremesDT_{date_str}_{param}_hres_advect_xr_{timestep_interval}_{timesteps}.nc',    # source file #KW: removing pre-processed/ from the path
        output= destineE_datafolder + f'DestinE_ExtremesDT_{date_str}_{param}_hres_interp_nlgrid_{timestep_interval}_{timesteps}.nc'         # output file #KW: removing pre-processed/ from the path
    )

    #Open created grid
    hres_interp_nlgrid_file=destineE_datafolder + f'DestinE_ExtremesDT_{date_str}_{param}_hres_interp_nlgrid_{timestep_interval}_{timesteps}.nc'
    destinE_nlgrid = xr.open_dataset(hres_interp_nlgrid_file) #KW: removing pre-processed/ from the path

    log.info(f"in pre_process_destine_data, radar_xr['time'][-1].values == {radar_xr['time'][-1].values}, timestep_interval == {timestep_interval}, timesteps == {timesteps} & timestep_interval * timesteps == {timestep_interval * timesteps}")
    time_slice = slice(pd.to_datetime(radar_xr['time'][-1].values) + timedelta(minutes=5), pd.to_datetime(radar_xr['time'][-1].values) + timedelta(minutes = timestep_interval * timesteps)+ timedelta(minutes=5) )
    log.info(f"time_slice = {time_slice}")
    #slice the timesteps to match the radar timesteps
    destinE_nlgrid_blend = destinE_nlgrid.sel(
        time=time_slice)
    
    exdt_time_slice = [destinE_nlgrid['time'][1].values, destinE_nlgrid['time'][-1].values]
    log.info(f'required time range: {time_slice}')
    log.info(f'ex dt time range: {exdt_time_slice}')
    
    # 
    if np.size(destinE_data.time.values) == 1:
        len_nwp=len(destinE_nlgrid_blend['time'])
        log.info(f'len_nwp based on time: {len_nwp}')
        log.info(f'timesteps == {timesteps}')
    else:
        len_nwp=len(destinE_nlgrid_blend['step'])
        log.info(f'len_nwp based on step: {len_nwp}')
        log.info(f'timesteps == {timesteps}')
        #KW: at some times of day this seems to fail. Not sure why.
    assert len_nwp == (timesteps + 1), f"Not the correct length timesteps in destine file ({hres_interp_nlgrid_file}), length is currently: {len_nwp} while it should be {(timesteps + 1)} "
    
    return destinE_nlgrid_blend