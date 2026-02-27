
What is DestinE_code?
================

DestinE_code is a fork of pysteps: the open-source and community-driven Python library for probabilistic precipitation nowcasting, i.e. short-term ensemble prediction systems.

DestinE_code enhances the steps blending algorithm within pySTEPS by allowing custom weights to be used. 
These weights can be found through optimisation using optuna for historical cases, or by a pre-trained Random Forest machine learning algorithm.

The repository also contains an operations ready script built to use ECMWF ExtremesDT and IFS as forecast input, and radar images from the Royal Dutch Meteorological Institute (KNMI) as input to the nowcasting algorithm DGMR 

The repository also contains a streamlit powered verification dashboard for verifying the performance of the blends.

Installation
============

Install the custom fork of pysteps from git+https://github.com/Joep1999/DestinE_code.git@master#subdirectory=pysteps_destine

Important Note
============

Running DGMR is only possible in Linux environments.


Usage
=====

To run the blending or optimization first change the paths in the beginning of the blending_operaional or optimize_optuna script to match the paths of the knmi_data and the folder to which you have cloned the repository.
After starting the optimization, you can check the progress by running optuna-dashboard sqlite:///optuna_blending_study.db

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

Reference publications
======================

Imhoff, R.O., L. De Cruz, W. Dewettinck, C.C. Brauer, R. Uijlenhoet, K-J. van Heeringen, 
C. Velasco-Forero, D. Nerini, M. Van Ginderachter, and A.H. Weerts, 2023:
Scale-dependent blending of ensemble rainfall nowcasts and NWP in the open-source
pysteps library. *Q J R Meteorol Soc.*, 1-30,
doi: `10.1002/qj.4461 <https://doi.org/10.1002/qj.4461>`_.


Ravuri, S., Lenc, K., Willson, M. et al. Skilful precipitation nowcasting using deep generative models of radar. Nature 597, 672–677 (2021).
doi: `<https://doi.org/10.1038/s41586-021-03854-z>`_

The DGMR model is under the terms of the Creative Commons Attribution 4.0 International License.
