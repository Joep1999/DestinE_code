import logging
log = logging.getLogger(__name__)

import os



#def pre_process_IFS_data(date, timestep_interval, timesteps, knmi_input_dir, destineE_datafolder, historical_destine, radar_xr):

def pre_process_ifs_data(ifs_file_original, ifs_file_preprocessed, date, timestep_interval, timesteps, knmi_input_dir, radar_xr):
    # check if pre-processed version exists
    date_str = date.strftime('%Y%m%d%H')
    date_str_day = date_str[:-2]
    yr = date.year
    mnth = date.month
    hour = date.hour
    forecast_hour = round_down(hour)

    if not os.path.exists(f'/srv/data/nas/project_data/p111_ecmwf_destine/IFS/2024/5/pre-processed/IFS_{date_str_day}_hres_advect_xr_{timestep_interval}_{timesteps}.nc'):
        #open original data TODO fix that is uses the latest one
        local_folder_today = destineE_datafolder + '/IFS/{}/{}/'.format(yr,str(mnth).zfill(2))
        local_file_today = local_folder_today + f'original/IFS_{date_str_day}{forecast_hour}_228.128.grib'
        local_file_today_nc_nl = local_folder_today + f'original/IFS_{date_str_day}{forecast_hour}_228.128_subset.nc'
       
            # IFS_data_raw = xr.open_dataset(local_file_today)
            # Quicker way of loading grib data
        IFS_data_raw = xr.open_dataset(
            local_file_today,
            engine="cfgrib",
            backend_kwargs={
                "indexpath": "",        # disable slow index caching
            },
        )           

        # convert to mm/h
        IFS_data_raw['tp'].attrs = {'long_name': 'Accumulated precipitation', 'units': 'mm/h', 'param': '193.1.0'}
            # accum_prcp = IFS_data_raw['tp'] * 1000
        
        if not os.path.exists(local_file_today_nc_nl):
            accum_prcp_subset = IFS_data_raw['tp'].sel(latitude=slice(56.4, 48.4),longitude=slice(-1, 11.87)) 
            accum_prcp_subset = accum_prcp_subset.assign_coords(step=[pd.to_datetime( IFS_data_raw['time'].values) + timedelta(hours =i) for i in range(len(IFS_data_raw['step']))])
            accum_prcp_subset.to_netcdf(local_file_today_nc_nl)

        accum_prcp_subset = xr.open_dataset(local_file_today_nc_nl)


        precipitation_m = accum_prcp_subset - accum_prcp_subset.shift({'step': 1})
        precipitation_m = precipitation_m.dropna('step', how="all")

        precipitation = precipitation_m['tp'] * 1000

        fcst_avg = precipitation[:,:].mean(axis=(1, 2, 3))
        percentile_90 = np.where(fcst_avg.values == np.percentile(fcst_avg, 90, interpolation = 'nearest'))[0][0]
        percentile_50 = np.where(fcst_avg.values == np.percentile(fcst_avg, 50, interpolation = 'nearest'))[0][0]
        percentile_30 = np.where(fcst_avg.values == np.percentile(fcst_avg, 30, interpolation = 'nearest'))[0][0]
        precipitation_percentiles = xr.concat([precipitation[percentile_30], precipitation[percentile_50], precipitation[percentile_90]], dim="number")
        precipitation_percentiles  = np.array(precipitation_percentiles)

        # select ensembles (random or percentile based)
        
        # downscale with rainfarm
        IFS_data_radar_scale = np.zeros((precipitation_percentiles.shape[0], precipitation_percentiles.shape[1], 640, 1032))
        for i in range(precipitation_percentiles.shape[0]):
            for j in range(precipitation_percentiles.shape[1]):
                down = rainfarm.downscale(precipitation_percentiles[i][j], ds_factor=4, kernel_type='gaussian')
                IFS_data_radar_scale[i, j] = down
        
        # backward advection correction
        IFS_nlgrid_hres_advected = IFS_data_radar_scale[:,0:1]  # keep first slice as (1, ny, nx)
        in_between = np.zeros(IFS_data_radar_scale[:,0:2].shape)
        for j in range(IFS_data_radar_scale.shape[1]-1):
            for i in range(IFS_data_radar_scale.shape[0]):
                steps = advection_correction_backward(IFS_data_radar_scale[i][j : j + 2], T=60, t=timestep_interval)
                in_between[i] = steps
            
            IFS_nlgrid_hres_advected = np.concatenate((IFS_nlgrid_hres_advected,in_between), axis = 1)
        
        # regrid with cdo to knmi grid
         # Write to netcdf so cdo can use the data
        cdo_to_netcdf(date, IFS_data_raw, precipitation, IFS_nlgrid_hres_advected, destineE_datafolder,  local_folder_today + f'pre-processed/IFS_{date_str_day}{forecast_hour}_hres_advect_xr_{timestep_interval}_{timesteps}.nc', f"{timestep_interval}min", historical_destine)
        temp = xr.open_dataset( local_folder_today + f'pre-processed/IFS_{date_str_day}{forecast_hour}_hres_advect_xr_{timestep_interval}_{timesteps}.nc')
        #if not easy fix then do all ensembles separate
        # REGRID DESTINE DATA TO KNMI RADAR GRID
        cdo.remapnn(
            f"{knmi_input_dir}knmi_grid.txt",                  # target grid
            input=local_folder_today + f'pre-processed/IFS_{date_str_day}{forecast_hour}_hres_advect_xr_{timestep_interval}_{timesteps}.nc',    # source file
            output= local_folder_today + f'pre-processed/IFS_{date_str_day}{forecast_hour}_hres_interp_nlgrid_{timestep_interval}_{timesteps}.nc'         # output file
        )

        #Open created grid
        IFS_nlgrid = xr.open_dataset(local_folder_today + f'pre-processed/IFS_{date_str_day}{forecast_hour}_hres_interp_nlgrid_{timestep_interval}_{timesteps}.nc')

        time_slice = slice(pd.to_datetime(radar_xr['time'][-1].values) + timedelta(minutes=5), pd.to_datetime(radar_xr['time'][-1].values) + timedelta(minutes = timestep_interval * timesteps)+ timedelta(minutes=5) )

        IFS_nlgrid_blend = IFS_nlgrid.sel(
            time=time_slice)

        print(f'IFS_nlgrid_blend time range: {time_slice}')

        len_nwp=len(IFS_nlgrid_blend['time'])
        assert len_nwp == (timesteps + 1), f'Not the correct length timesteps in destine file, length is currently: {len_nwp} while it should be {(timesteps + 1)} '
        


        return IFS_nlgrid_blend

    else:
        IFS_nlgrid = xr.open_dataset(local_folder_today + f'pre-processed/IFS_{date_str_day}{forecast_hour}_hres_interp_nlgrid_{timestep_interval}_{timesteps}.nc')
        time_slice = slice(pd.to_datetime(radar_xr['time'][-1].values) + timedelta(minutes=5), pd.to_datetime(radar_xr['time'][-1].values) + timedelta(minutes = timestep_interval * timesteps)+ timedelta(minutes=5) )

        #slice the timesteps to match the radar timesteps
        IFS_nlgrid_blend = IFS_nlgrid.sel(
            time=time_slice)
            
        print(f'IFS_nlgrid_blend time range: {time_slice}')

        len_nwp=len(IFS_nlgrid_blend['time'])
        assert len_nwp == (timesteps + 1), f'Not the correct length timesteps in destine file, length is currently: {len_nwp} while it should be {(timesteps + 1)} '
        
        return IFS_nlgrid_blend
    