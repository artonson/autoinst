import numpy as np 

DATASET_PATH = '/Users/cedric/Datasets/semantic_kitti/'

config_tarl_spatial_dino = {
    "name": "spatial_1.0_tarl_0.5_dino_0.1_t_0.005",
    "out_folder": "ncuts_data_tarl_dino_spatial/",
    "gamma": 0.1,
    "alpha": 1.0,
    "theta": 0.5,
    "T": 0.005,
    "gt": True,
}

config_tarl_spatial = {
    "name": "spatial_1.0_tarl_0.5_t_0.03",
    "out_folder": "ncuts_data_tarl_spatial/",
    "gamma": 0.0,
    "alpha": 1.0,
    "theta": 0.5,
    "T": 0.03,
    "gt": True,
}

config_spatial = {
    "name": "spatial_1.0_t_0.075",
    "out_folder": "ncuts_data_spatial/",
    "gamma": 0.0,
    "alpha": 1.0,
    "theta": 0.0,
    "T": 0.075,
    "gt": True,
}

config_maskpls_tarl_spatial = {
    "name": "maskpls_comp_",
    "out_folder": "maskpls_7/",
    "gamma": 0.0,
    "alpha": 0.0,
    "theta": 0.0,
    "T": 0.0,
    "gt": True,
}


config_maskpls_tarl_spatial_dino = {
    "name": "maskpls_no_filter_5_",
    "out_folder": "maskpls_no_filter_5/",
    "gamma": 0.0,
    "alpha": 0.0,
    "theta": 0.0,
    "T": 0.0,
    "gt": True,
}

start_chunk = 0
start_seq = 0
seqs = list(range(0, 11))
exclude = [1, 4] #these only have static scenes

minor_voxel_size = 0.05
major_voxel_size = 0.35
chunk_size = np.array([25, 25, 25])  # meters
overlap = 3  # meters
ground_segmentation_method = "patchwork"
NCUT_ground = False

beta = 0.0
tarl_norm = False
proximity_threshold = 1.0

out_folder = "pcd_preprocessed/semantics/"