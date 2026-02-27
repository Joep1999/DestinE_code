
from pathlib import Path

BASE = Path("/srv/data/nas/project_data/p111_ecmwf_destine")

# These keys must match the keys returned by extract_domain_series().
DATASETS = {
    "radar": ["Radar (observed)", "k-"],
    "destine": ["DestinE ExtremesDT","r-"] ,
    "ifs": ["IFS ensemble", "y-"],
    "destine_ifs": ["IFS and DestinE multi-model ensemble", "y-"],
    "nowcast": ["Nowcast ensemble", "g-"],
    "blend_ifs": ["Blend (IFS)","c-"],
    "blend_ifs_noise": ["Blend with weights determined by pySTEPS","m-"],
    'blend_optimised': ["Blend with optimised weights", "darkorange"],
    'blend_optimised_upper_cascade': ["Blend with optimised weights for upper cascades", "brown"],
    'blend_optimised_kmeans_estimation': ["Blend with optimised weights estimated with k-means", "brown"],
    'blend_optimised_machine_learning_prediciton':  ["Blend with weights estimated with machine learning", "r-"],
    'pysteps_nowcast': ["PySTEPS Nowcast", "blue"]
}


DATASET_TO_VARIANT = {
    "blend": "_noise",
    "blend_ifs": "_IFS",
    "blend_ifs_noise": "_IFS_noise_probmatch",
    #"blend_optimised": "_IFS_noise_optimised_weights"
    "blend_optimised": f"_IFS_optimised_weights",
    "blend_optimised_upper_cascade": f"_IFS_noise_probmatch_optimised_weights_upper_cascade",
    'pysteps_nowcast': "_pysteps_nowcast"
}


BLENDING_CONFIG = {
"timesteps" : 24,
'timestep_interval' : 30,
'n_ens_members' : 20,
'n_ens_members_dgmr' : 5,
'weights_method' : 'custom',
'gauge_adjusted' : True,
'knmi_input_dir' : '/srv/data/nas/input_general/knmi_radar_gauge_adj/',
'destineE_datafolder' : '/srv/data/nas/project_data/p111_ecmwf_destine/',
'historical_destine' : True,
'multi_model' : True,
'return_weights' : True,
're_do_blending' : True, 
'noise' : True, 
'probmatching' : True
}