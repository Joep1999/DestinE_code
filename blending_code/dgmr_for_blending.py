#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGMR Nowcasting Module

This module runs the DGMR nowcasting model for the Netherlands using KNMI radar data.
It can be imported into other Python scripts or run as a standalone script.

Main function:
    run_dgmr() -> np.ndarray
        Returns DGMR_det (the deterministic nowcast)

Author: Joep (refactored as module by ChatGPT)
"""

import os, certifi
import sys
import time as timing
import shutil
import requests
import numpy as np
from datetime import datetime, timedelta
from base64 import decodebytes

from pysteps import io
from pysteps.utils import conversion


os.environ["CURL_CA_BUNDLE"] = certifi.where()

import tensorflow as tf
import tensorflow_hub

# Model location
TFHUB_BASE_PATH = "gs://dm-nowcasting-example-data/tfhub_snapshots"



# DGMR settings
NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 18


def load_module(input_height, input_width):
    """Load a TF-Hub snapshot of the DGMR model."""
    hub_module = tensorflow_hub.load(
        os.path.join(TFHUB_BASE_PATH, f"{input_height}x{input_width}")
    )
    return hub_module.signatures['default']


# Load DGMR model once
module = load_module(1536, 1280)

def predict(module, input_frames, num_samples=1, include_input_frames_in_result=False):
    """Run DGMR model prediction."""
    input_frames = tf.math.maximum(input_frames, 0.)
    input_frames = tf.expand_dims(input_frames, 0)
    input_frames = tf.tile(input_frames, multiples=[num_samples, 1, 1, 1, 1])

    _, input_signature = module.structured_input_signature
    z_size = input_signature['z'].shape[1]
    z_samples = tf.random.normal(shape=(num_samples, z_size))

    inputs = {
        "z": z_samples,
        "labels$onehot": tf.ones(shape=(num_samples, 1)),
        "labels$cond_frames": input_frames
    }
    samples = module(**inputs)['default']

    if not include_input_frames_in_result:
        samples = samples[:, NUM_INPUT_FRAMES:, ...]

    return tf.math.maximum(samples, 0.)


def run_dgmr(R, module = module, runtimes=4):
    # Load the model for size 1536 by 1280

    """Run the DGMR pipeline and return deterministic forecast (DGMR_det)."""
    t1 = timing.time()

    # --- Prepare DGMR input ---
    input_DGMR = R[-4:]
    paddings = tf.constant([[0, 0], [385, 386], [290, 290]])
    input_DGMR = tf.pad(input_DGMR, paddings, "CONSTANT")
    input_DGMR = np.reshape(np.float32(input_DGMR), (4, 1536, 1280, 1))
    input_DGMR[np.isinf(input_DGMR)] = 0.

    # --- Run DGMR twice ---
 
    prediction_1 = predict(module, input_DGMR, include_input_frames_in_result=True)
    prediction_1 = np.reshape(prediction_1, (22, 1536, 1280, 1))[3:]
    extended_predictions = prediction_1.copy()

    for i  in range(runtimes-1):
        prediction = predict(module, extended_predictions[-4:], include_input_frames_in_result=False)
        prediction = np.reshape(prediction, (18, 1536, 1280, 1))
        print(np.shape(extended_predictions))
        extended_predictions = np.concatenate((extended_predictions, prediction))
        # copy the last DGMR image, so that there is enough images for the blending
        if i == runtimes-1:
            extended_predictions = np.concatenate((extended_predictions, extended_predictions[-1]))



    # prediction_2 = predict(module, prediction_1[-4:], include_input_frames_in_result=False)
    # prediction_2 = np.reshape(prediction_2, (18, 1536, 1280, 1))

    # prediction_3 = predict(module, prediction_2[-4:], include_input_frames_in_result=False)
    # prediction_3 = np.reshape(prediction_3, (18, 1536, 1280, 1))

    # prediction_4 = predict(module, prediction_3[-4:], include_input_frames_in_result=False)
    # prediction_4 = np.reshape(prediction_4, (18, 1536, 1280, 1))

    # extended_predictions = np.concatenate((prediction_1, prediction_2, prediction_3,prediction_4))
    DGMR_det = np.reshape(extended_predictions[:, 385:1150, 290:990, :],
                          (len(extended_predictions), 765, 700))

    t2 = timing.time()
    print(f"DGMR run completed in {int((t2 - t1) / 60)} min {int((t2 - t1) % 60)} sec")

    return DGMR_det

import time as timing
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_dgmr_ensemble(R, ens_members, module = module, forecast_length=4):
    """Run DGMR ensemble in parallel."""
    print(f"Launching DGMR ensemble with {ens_members} members...")

    results = []
    t_start = timing.time()

    # Use separate processes for each ensemble member
    with ThreadPoolExecutor(max_workers=ens_members) as executor:
        futures = [executor.submit(run_dgmr, R, module, forecast_length) for _ in range(ens_members)]
        for i, f in enumerate(as_completed(futures), 1):
            result = f.result()
            results.append(result)
            print(f"Ensemble member {i}/{ens_members} finished.")

    # Stack ensemble results: shape = (ens_members, time, y, x)
    DGMR_ens = np.stack(results, axis=0)

    t_end = timing.time()
    print(f"DGMR ensemble completed in {int((t_end - t_start) / 60)} min {int((t_end - t_start) % 60)} sec")

    return DGMR_ens

if __name__ == "__main__":
    output = run_dgmr(R)
    print("DGMR_det shape:", output.shape)
