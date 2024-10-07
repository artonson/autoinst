import numpy as np 
import yaml 

DATASET_PATH = '/media/cedric/Datasets2/semantic_kitti/'

config_tarl_spatial_dino = {
    "name": "spatial_1.0_tarl_0.5_dino_0.1_t_0.005",
    "out_folder": "ncuts_data_tarl_dino_spatial/",
    "gamma": 0.1,
    "alpha": 1.0,
    "theta": 0.5,
    'beta':0.0,
    "T": 0.005,
    "gt": True,
}

config_tarl_spatial = {
    "name": "spatial_1.0_tarl_0.5_t_0.03",
    "out_folder": "ncuts_data_tarl_spatial/",
    "gamma": 0.0,
    "alpha": 1.0,
    "theta": 0.5,
    'beta':0.0,
    "T": 0.03,
    "gt": True,
}

config_spatial = {
    "name": "spatial_1.0_t_0.075",
    "out_folder": "ncuts_data_spatial/",
    "gamma": 0.0,
    "alpha": 1.0,
    "theta": 0.0,
    'beta':0.0,
    "T": 0.075,
    "gt": True,
}

config_maskpls_tarl_spatial = {
    "name": "maskpls_tarl_spatial_",
    "out_folder": "maskpls_tarl_spatial/",
    "gamma": 0.0,
    "alpha": 0.0,
    "theta": 0.0,
    'beta':0.0,
    "T": 0.0,
    "gt": True,
}

start_chunk = 0
start_seq = 0
seqs = list(range(0, 11))
exclude = [1, 4] #these only have static scenes

MINOR_VOXEL_SIZE = 0.05
MAJOR_VOXEL_SIZE = 0.35
CHUNK_SIZE = np.array([25, 25, 25])  # meters
OVERLAP = 3  # meters
ground_segmentation_method = "patchwork"
NCUT_ground = False
SPLIT_LIM = 0.01

BETA = 0.0
TARL_NORM = False
PROXIMITY_THRESHOLD = 1.0
HPR_RADIUS = 1000
NUM_DINO_FEATURES = 384
MEAN_HEIGHT = 0.6

ADJACENT_FRAMES_CAM=(16, 13)
ADJACENT_FRAMES_TARL=(10, 10)
CAM_IDS = [0]

GEN_SELF_TRAIN_DATA = False
if GEN_SELF_TRAIN_DATA:
    OVERLAP = 24  # high overlap for self-training data generation

with open("utils/semantic-kitti.yaml", "r") as stream:
    semyaml = yaml.safe_load(stream)
learning_map = semyaml["learning_map"]

MASKPLS_weights = '/media/cedric/Datasets21/Weights/KITTI/TARL_Spatial/mask_pls_oversegmented_epoch=07.ckpt'
TEST_MAP = True #breaks the code after the first sample map
CONFIG = config_spatial
OUT_FOLDER = "pcd_preprocessed/instances/"
OUT_FOLDER_NCUTS = OUT_FOLDER + CONFIG["out_folder"]
OUT_FOLDER_INSTANCES = OUT_FOLDER + "instances/"
OUT_FOLDER_TRAIN = OUT_FOLDER + "train/"