import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
src_path = os.path.abspath("../..")
if src_path not in sys.path:
    sys.path.append(src_path)
from dataset_utils import create_nuscenes_odometry_dataset
from dataset.filters.filter_list import FilterList
from dataset.filters.range_filter import RangeFilter
from dataset.filters.apply_pose import ApplyPose
import scipy
from scipy.spatial.distance import cdist
from normalized_cut import normalized_cut
from ncuts_utils import ncuts_chunk,kDTree_1NN_feature_reprojection_colors, get_merge_pcds
from dataset_utils import * 
from point_cloud_utils import get_pcd, transform_pcd, kDTree_1NN_feature_reprojection, remove_isolated_points, get_subpcd, get_statistical_inlier_indices, merge_chunks_unite_instances
from aggregate_pointcloud import aggregate_pointcloud
from visualization_utils import generate_random_colors, color_pcd_by_labels,generate_random_colors_map
from sam_label_distace import sam_label_distance
from chunk_generation import subsample_positions, chunks_from_pointcloud, indices_per_patch, tarl_features_per_patch, image_based_features_per_patch, dinov2_mean, get_indices_feature_reprojection
from metrics.metrics_class import Metrics
import shutil

DATASET_PATH = '/media/cedric/Datasets1/nuScenes_mini_v2/nuScenes'
SEQUENCE_NUM = 5

dataset = create_nuscenes_odometry_dataset(DATASET_PATH,SEQUENCE_NUM,ncuts_mode=True, sam_folder_name="SAM_Underseg", dinov2_folder_name="Dinov2")

ind_start = 0
ind_end = len(dataset)
minor_voxel_size = 0.05
major_voxel_size = 0.35
chunk_size = np.array([25, 25, 25]) #meters
overlap = 3 #meters
ground_segmentation_method = 'patchwork' 
NCUT_ground = False 
out_folder_ncuts = 'test_data/'
if os.path.exists(out_folder_ncuts):
        shutil.rmtree(out_folder_ncuts)
os.makedirs(out_folder_ncuts)

out_folder = 'pcd_preprocessed_nuscenes/'
if os.path.exists(out_folder) == False : 
        os.makedirs(out_folder)
        
        
if os.path.exists('{out_folder}all_poses_' + str(SEQUENCE_NUM) + '_' + str(0) + '.npz') == False:
        process_and_save_point_clouds(dataset,ind_start,ind_end,minor_voxel_size=minor_voxel_size,
                                major_voxel_size=major_voxel_size,icp=False,
                                out_folder=out_folder,sequence_num=SEQUENCE_NUM,
                                ground_segmentation_method=ground_segmentation_method,nuscenes_flag=True)