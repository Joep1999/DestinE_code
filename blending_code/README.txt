README:

IMPORTANT!!!
The blending algorithm works on a custom version of pysteps. Therefore please install pysteps from the GitHub directory as specified in the requirements light txt file

To run the optimization, first change the paths in the beginning of the optimize optuna script to match the paths of the knmi_data and the folder where you have the destine data.
after starting the optimization, you can check the progress by running 

Then, set the dates over which the algorithm will run. (maybe start with 1 date to check how it is going)
run optimize optuna. This will create a 'study object' which will in the end contain the weights that we would use to create the 'optimal blend'.

options that can be changed affecting blending function are (to make it run faster or with more timesteps / ensembles):
- timesteps_interval (currently 20-> 6 hours becomes 18 timesteps)
- n_ens_members (the model creates a ensemble from the two deterministic inputs)
- n_ens_members_dgmr (the amount of times dgmr is run) 

options that can be changed to affect the optimization are:
- n_trials=30 ( adjust to available time, each blending run should take about 5-10 minutes, with the exception of the first one, which will take longer since DGMR has to run )
- n_jobs=1	(you can add sequentiality here, but I did not experiment with this yet)
- timeout=6 * 3600,   (currently set to 6 hours max runtime)


General flow of the blending script:
- KNMI radar data is downloaded
- DestinE extremes DT data is loaded, downscaled, interpolated and the reprojected to match the radar projection and grid size
- DGMR is run
- Data is pre-processed for blending
- blending algorithm is run.

General flow of optimize optuna script:
- run blending once with the normal pysteps weights to determine what they are.
- startup thee study for the specified date
- run the blending in the optuna algorithm once with the baseline clim parameters and once with the dynamic pysteps paramters
- determine the difference between the parameters, and use this to set the boundaries for optuna
- let optuna optimize 





