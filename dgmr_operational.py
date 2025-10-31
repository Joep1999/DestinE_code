#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 09:36:37 2023

The operational script that runs the DGMR nowcasting model for the Netherlands.

The script is downloading data from KNMI and processing it to a 2 hour nowcast.

DGMR is a machine learning model that is developed by DeepMind. It runs a 
nowcast up to 90 minutes ahead. To obtain a 2 hour nowcast, the model will run
twice. At the end, a probabilistic forecast is generated from the deterministic 
outcome. Finally, the data is saved. 

@author: joep
"""
######################
#Load packages and load model from google cloud ----------------------------------------------------------------------------------------------------
#######################
import tensorflow as tf
import tensorflow_hub
import os
import sys
import time as timing

from datetime import datetime, timedelta

import numpy as np
from pysteps import io
from pysteps.utils import conversion
import shutil
import requests
import wradlib
import pysftp
import paramiko
from base64 import decodebytes

from numba import jit, prange

from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('/srv/config/config_scripts.ini')
lib_path = config['paths']['lib']
sys.path.append(lib_path)
import wi_library as wi

input_dir = config['paths']['input_general'] + "knmi_radar/"

config = ConfigParser()
config.read('/srv/config/config_w2o.ini')
output_dir = config['paths']['steps_input'] + "/knmi_radar/dgmr/"

# Set HN ftp settings
host = "ftp.hydronet.nl"
username = "weatherimpact"
passwd = "vArbz9LE23$uaN"

gauge_adjusted = False

# Set the base bath from the Google Bucket
TFHUB_BASE_PATH = "gs://dm-nowcasting-example-data/tfhub_snapshots"

# Define a funciton to load the model
def load_module(input_height, input_width):
  """Load a TF-Hub snapshot of the 'Generative Method' model."""
  hub_module = tensorflow_hub.load(
      os.path.join(TFHUB_BASE_PATH, f"{input_height}x{input_width}"))
  # Note this has loaded a legacy TF1 model for running under TF2 eager mode.
  # This means we need to access the module via the "signatures" attribute. See
  # https://github.com/tensorflow/hub/blob/master/docs/migration_tf2.md#using-lower-level-apis
  # for more information.
  return hub_module.signatures['default']

# Load the model for size 1536 by 1280
module = load_module(1536, 1280)

# Settings for DGMR
NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 18

######################
#Define Functions ---------------------------------------------------------------------------------------------------------------
#######################
t1 = timing.time()

#Function for the generation of probabilistic values
@jit(nopython=True, parallel=True)
def count_neighboring_values_numba(arr,padded_arr, threshold, space_radius, time_radius, max_cells):

    counts = np.zeros(arr.shape, dtype=np.float64)
    radius_squared = space_radius**2

    for i in prange(time_radius, arr.shape[0] + time_radius):
        for j in prange(space_radius, arr.shape[1] + space_radius):
            for k in prange(space_radius, arr.shape[2] + space_radius):
                count = 0

                for t in range(i - time_radius, i + time_radius + 1):
                    for x in range(j - space_radius, j + space_radius + 1):
                        for y in range(k - space_radius, k + space_radius + 1):
                            if (x - j)**2 + (y - k)**2 <= radius_squared:
                                if padded_arr[t, x, y] > threshold:
                                    count += 1

                counts[i - time_radius, j - space_radius, k - space_radius] = count / max_cells

    return counts

#function needed to run the DGMR model
def predict(module, input_frames, num_samples=1,
            include_input_frames_in_result=False):
    """Make predictions from a TF-Hub snapshot of the 'Generative Method' model.
    Args:
      module: One of the raw TF-Hub modules returned by load_module above.
      input_frames: Shape (T_in,H,W,C), where T_in = 4. Input frames to condition
        the predictions on.
      num_samples: The number of different samples to draw.
      include_input_frames_in_result: If True, will return a total of 22 frames
        along the time axis, the 4 input frames followed by 18 predicted frames.
        Otherwise will only return the 18 predicted frames.
    
    Returns:
      A tensor of shape (num_samples,T_out,H,W,C), where T_out is either 18 or 22
      as described above.
    """
    input_frames = tf.math.maximum(input_frames, 0.)
    # Add a batch dimension and tile along it to create a copy of the input for
    # each sample:
    input_frames = tf.expand_dims(input_frames, 0)
    input_frames = tf.tile(input_frames, multiples=[num_samples, 1, 1, 1, 1])
    print(tf.shape(input_frames))
    # Sample the latent vector z for each sample:
    _, input_signature = module.structured_input_signature
    
    z_size = input_signature['z'].shape[1]
    z_samples = tf.random.normal(shape=(num_samples, z_size))
    
    inputs = {
        "z": z_samples,
        "labels$onehot" : tf.ones(shape=(num_samples, 1)),
        "labels$cond_frames" : input_frames
    }
    samples = module(**inputs)['default']
    print('model_ran')
    if not include_input_frames_in_result:
      # The module returns the input frames alongside its sampled predictions, we
      # slice out just the predictions:
      samples = samples[:, NUM_INPUT_FRAMES:, ...]
    
    # Take positive values of rainfall only.
    samples = tf.math.maximum(samples, 0.)
    return samples

#function needed to run the DGMR model
def extract_input_and_target_frames(radar_frames):
    """Extract input and target frames from a dataset row's radar_frames."""
    # We align our targets to the end of the window, and inputs precede targets.
    input_frames = radar_frames[-NUM_TARGET_FRAMES-NUM_INPUT_FRAMES : -NUM_TARGET_FRAMES]
    target_frames = radar_frames[-NUM_TARGET_FRAMES : ]
    return input_frames, target_frames

#function needed to run the DGMR model
def horizontally_concatenate_batch(samples):
    n, t, h, w, c = samples.shape
    # N,T,H,W,C => T,H,N,W,C => T,H,N*W,C
    return tf.reshape(tf.transpose(samples, [1, 2, 0, 3, 4]), [t, h, n*w, c])

#function not used in this script, but is useful when needing to compare forecasta and target frames
def predictRainfall(eventSelected, name):
    num_samples = 1
    input_frames, target_frames  = extract_input_and_target_frames(eventSelected)
    print(tf.shape(input_frames))
    print(tf.shape(target_frames))
    samples = predict(module, input_frames, num_samples=num_samples, include_input_frames_in_result=False)
    
    return(samples, target_frames)

def boolstr_to_floatstr(v):
    if v == True:
        return 0
    else:
        return 1
    
######################
#Start script --------------------------------------------------------------------------------------------------------------------
#######################

#Get the current datetime
now = datetime.now()

#%%
# Download data from KNMI dataplatform
last_hour = now + timedelta(hours=-1)
removal_days = 2

if gauge_adjusted:
    url = 'https://api.dataplatform.knmi.nl/open-data/v1/datasets/nl_rdr_data_rtcor_5m/versions/1.0/files'
    lastfile = last_hour.strftime('RAD_NL25_RAC_RT_%Y%m%d%H%M.h5')
else:
    url = 'https://api.dataplatform.knmi.nl/open-data/datasets/radar_reflectivity_composites/versions/2.0/files'
    lastfile = last_hour.strftime('RAD_NL25_PCP_NA_%Y%m%d%H00.h5')


api_key = '5e554e19274a9600012a3eb10174be35b75442a7a5e2ba066642a279'



file_list = requests.get(url, headers={'Authorization': api_key},
                         params={"startAfterFilename": lastfile,
                                 "maxKeys": 12})

file_list = file_list.json().get('files')

# Download the last 3 available files
for ii in range(len(file_list)-4,len(file_list)):
    fn = file_list[ii]['filename']
    
    yr = fn[16:20]
    mnth = fn[20:22]
    day = fn[22:24]
    hour = fn[24:26]
    minute = fn[26:28]
    
    local_folder_today = input_dir + '/{}/{}/{}/'.format(yr,mnth,day)

    for folder in [local_folder_today]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    
    direc = local_folder_today
    
    if not os.path.exists(direc+fn):
    
    
        get_file_response = requests.get(url+'/'+fn+'/url', headers={'Authorization': api_key})
        
        download_url = get_file_response.json().get("temporaryDownloadUrl")
        
        dataset_file = requests.get(download_url, stream=True)
    
        if dataset_file.status_code == 200:
            with open(direc+fn, 'wb') as f:
                dataset_file.raw.decode_content = True
                shutil.copyfileobj(dataset_file.raw, f)

#%%
# Create the destination folder
date = datetime(int(yr), int(mnth), int(day), int(hour),  int(minute), 0)
local_destination_dir = output_dir +  date.strftime('/%Y/%m/%d/')
for folder in [local_destination_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)

print('Open files')
# Find the radar files in the archive
if gauge_adjusted:
    fns = io.find_by_date(
        date, input_dir, "%Y/%m/%d", "RAD_NL25_RAC_RT_%Y%m%d%H%M", "h5", 5, num_prev_files=3
    )
else:
    fns = io.find_by_date(
        date, input_dir, "%Y/%m/%d", "RAD_NL25_PCP_NA_%Y%m%d%H%M", "h5", 5, num_prev_files=3
    )

importer_kwargs = {"accutime": 5, "qty": "DBZH", "pixelsize": 1000.0}

# Read the data from the archive
try:
    importer = io.get_method("knmi_hdf5", "importer")
    R, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)
except:
    print('Input data unreadable. Abort script.')
    sys.exit()

# Set modeldate
modeldate = metadata['timestamps'][-1]

# Check if the output file is already existent
# fn_out = modeldate.strftime('%Y/%m/%d/dgmr_nowcast_%Y%m%d%H%M.nc')
# if os.path.exists(output_dir + fn_out):
#     print("Output for modeldate already existent, abort script")
#     sys.exit()


# Define the lat and lon arrays from the metadata
lon = np.array([metadata['x1'] + metadata['xpixelsize'] * tt for tt in range(int((metadata['x2']-metadata['x1'])/metadata['xpixelsize']))])
lat = np.array([metadata['y1'] + metadata['ypixelsize'] * tt for tt in range(int((metadata['y2']-metadata['y1'])/metadata['ypixelsize']))])

#%%
#Apply clutter filter to the radar data

#Apply Gabella filter
clutter = []
for i,j in enumerate(R):
    clutter.append(wradlib.classify.filter_gabella(R[i,:,:], wsize=5))


# create binary matrix for true/false matrix
clutter = np.vectorize(boolstr_to_floatstr)(clutter).astype(float)

#multiply binary matrix with rain rate matrix to set the clutter pixels to 0.
R = np.multiply(R,clutter)

#Convert to rain rate
R, metadata = conversion.to_rainrate(R, metadata)

R[np.isnan(R)] = 0

#%%
#First pad the radar images, then run the DGMR model twice, to extend the model to 2 hours ahead.

input_DGMR = R[-4:]
print(np.shape(input_DGMR))
paddings = tf.constant([[0, 0], [385, 386], [290, 290]])
input_DGMR = tf.pad(input_DGMR, paddings, "CONSTANT")
input_DGMR = np.float32(input_DGMR)
input_DGMR = np.reshape(input_DGMR,(4,1536, 1280, 1))
input_DGMR[np.isinf(input_DGMR)] = 0.


#In the first prediciton, include_input_frames_in_result=True becasue we want to keep the radarframes, to use fo the probability prediciton
prediction_1 = predict(module, input_DGMR, num_samples=1, include_input_frames_in_result=True)
prediction_1 = np.reshape(prediction_1,(22,1536, 1280, 1))

#If you want a larger delta T for the neighbourhood, the number of radar frames that are kept has to be increased here.
prediction_1 = prediction_1[3:]

input_frames_2 = prediction_1[-4:]

print(np.shape(input_frames_2))
prediction_2 = predict(module, input_frames_2, num_samples=1, include_input_frames_in_result=False)
prediction_2 = np.reshape(prediction_2,(18,1536, 1280, 1))

#If you want a larger delta T for the neighbourhood, the number of frames that are kept from the second prediction has to be increased here
prediction_2_part = prediction_2[:7]

extended_predictions = np.concatenate((prediction_1,prediction_2_part))

DGMR_depad = extended_predictions[:, 385:1150, 290:990, :]

DGMR_det = np.reshape(DGMR_depad,(len(DGMR_depad), 765,700))

# Calculate the cumulative precipitation
DGMR_det_cum = []
#calclate the cumulative precipitation
for i in range(len(DGMR_det) - 23):
    DGMR_det_cum.append(DGMR_det[i:i+24, :,:].sum(axis = 0))

DGMR_det_cum_np = np.asarray(DGMR_det_cum)


#Here, you can adjust the thresholds and dimetnions of the post-processing method. 
threshold_10 = 10
threshold_25 = 25

space_radius = 6
time_radius = 1

#The strat of the calculation of the probabilities (make the grid that will be used to move over the dataset)
x, y,z = np.mgrid[-time_radius:time_radius +1, -space_radius:space_radius+1, -space_radius:space_radius+1]
distance_mask = np.sqrt(y**2 + z**2) <= space_radius

max_cells = np.sum(distance_mask == True)


#USED TO MAKE PROBABILITY PREDICTIONS FOR EVERY TIMESTEP

# DGMR_det_pad = np.pad(DGMR_det, ((time_radius, time_radius),
#                              (space_radius, space_radius),
#                              (space_radius, space_radius)), mode='constant')


# DGMR_prob_numba = count_neighboring_values_numba(DGMR_det,DGMR_det_pad, threshold,space_radius,time_radius, max_cells)

#USED TO MAKE CUMULATIVE PROBABILITY PREDICTIONS

DGMR_det_cum_pad = np.pad(DGMR_det_cum_np, ((time_radius, time_radius),(space_radius, space_radius), (space_radius, space_radius)), mode='constant')

#Calculation fo threshold 10
DGMR_prob_cum_numba_10 = count_neighboring_values_numba(DGMR_det_cum_np,DGMR_det_cum_pad, threshold_10,space_radius,time_radius, max_cells)
DGMR_prob_cum_numba_10 = DGMR_prob_cum_numba_10[-2]
DGMR_prob_cum_numba_10 = np.expand_dims(DGMR_prob_cum_numba_10, axis=0)

#Calculation for threshold 25
DGMR_prob_cum_numba_25 = count_neighboring_values_numba(DGMR_det_cum_np,DGMR_det_cum_pad, threshold_25,space_radius,time_radius, max_cells)
DGMR_prob_cum_numba_25 = DGMR_prob_cum_numba_25[-2]
DGMR_prob_cum_numba_25 = np.expand_dims(DGMR_prob_cum_numba_25, axis=0)

#Save the data
fn = modeldate.strftime('%Y/%m/%d/dgmr_nowcast_%Y%m%d%H%M.nc')
varname = 'Precipiation rate'
varunit = 'mm/h'
time_out = [5*tt for tt in range(len(DGMR_det))]
tunits = modeldate.strftime('Minutes since %Y-%m-%d %H:%M')
wi.write_netcdf(output_dir, fn, varname, varunit, DGMR_det, lat, lon, time_out, tunits)

upload_fn = output_dir + fn

hostkey=b'AAAAB3NzaC1yc2EAAAADAQABAAABAQD8eyYy63UlliXY+tW/5A6b65NsLkaAHG/g9hehmXCP6uwDdKjcfSGrG7UPiRS1Z6EikUecVHKvvj5kMJa3BtgDZhcpkE9KI9yaFyN1x8E7NUtVRYgt/cm7lk4IiI6vbRDhTo6bLH7L2CowdIdy6VhZ2XhB+SBAxaiok1KokJqZD8lPBNLwNG8ImixeDbEw2mGTKE/uKPPKogShm2NRCpze0Q5LSlp4bvDtIX5wEnhlQkuSP7DO0UbE93QZXvgSbAx9ItqSX4MvnoMmNhOxhjPhaptCLSUPV6zB2+6/nHXRc/++U2Ts1gb1G+HiNF2B5IJ69+voFgglEiV+bk0pZ6dx'
wi.sftp_upload(host, username, passwd, upload_fn, 'Import/DGMR/Ensemble', hostkey=hostkey)


keydata = hostkey
key = paramiko.RSAKey(data=decodebytes(keydata))
cnopts = pysftp.CnOpts()
cnopts.hostkeys.add(host, 'ssh-rsa', key)

with pysftp.Connection(host, username, password=passwd, cnopts=cnopts) as ftp:
    ftp.cwd('Import/DGMR/Ensemble')
    ftp_files = ftp.listdir()

    for ftp_file in ftp_files:
        remote_file_path = os.path.join(ftp.pwd, ftp_file)
        if ftp.stat(ftp_file).st_mtime < (timing.time() - (removal_days * 86400)):
            if ftp.isfile(ftp_file):
                ftp.remove(ftp_file)
    ftp.close()

# Data and NetCDF file handling
data_prob = np.array([DGMR_prob_cum_numba_10, DGMR_prob_cum_numba_25])
fn_prob = modeldate.strftime('%Y/%m/%d/dgmr_probabilities_%Y%m%d%H%M.nc')
varnames_prob = ['P.Likelihood.GreaterThan.10MM', 'P.Likelihood.GreaterThan.25MM']
varunits_prob = ['%', '%']
time_out_prob = [120]
tunits_prob = modeldate.strftime('Minutes since %Y-%m-%d %H:%M')
wi.write_netcdf(output_dir, fn_prob, varnames_prob, varunits_prob, data_prob, lat, lon, time_out_prob, tunits_prob)

# Upload the file to the FTP server
upload_fn_prob = os.path.join(output_dir, fn_prob)
with pysftp.Connection(host,username, password=passwd, cnopts=cnopts) as ftp:
    ftp.cwd('Import/DGMR/Probabilities')
    ftp.put(upload_fn_prob)
    ftp.close()

# Remove files older than 2 days from FTP (Import/DGMR/Probabilities)
with pysftp.Connection(host,username, password=passwd, cnopts=cnopts) as ftp:
    ftp.cwd('Import/DGMR/Probabilities')
    ftp_files = ftp.listdir()

    for ftp_file in ftp_files:
        remote_file_path = os.path.join(ftp.pwd, ftp_file)
        if ftp.stat(ftp_file).st_mtime < (timing.time() - (removal_days * 86400)):
            if ftp.isfile(ftp_file):
                ftp.remove(ftp_file)
    ftp.close()
t2 = timing.time()

print("Time elapsed: ",int((t2-t1)/60)," minutes and ",int((t2-t1)%60)," seconds")
