import numpy as np
import os.path as osp
import os
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import sys

src_path = os.path.abspath("../..")
if src_path not in sys.path:
    sys.path.append(src_path)

from dataset.kitti_odometry_dataset import KittiOdometryDataset, KittiOdometryDatasetConfig
from dataset.filters.filter_list import FilterList
from dataset.filters.kitti_gt_mo_filter import KittiGTMovingObjectFilter
from dataset.filters.range_filter import RangeFilter
from dataset.filters.apply_pose import ApplyPose

from functions import hidden_point_removal, project_point_cloud_to_image

def get_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd

if __name__ == '__main__':
    DATASET_PATH = os.path.join('/Users/laurenzheidrich/Downloads/','fused_dataset')
    SEQUENCE_NUM = 10


    config = KittiOdometryDatasetConfig(
        cache=True,
        dataset_path=DATASET_PATH,
        filters=ApplyPose(),
        correct_scan_calibration=True,
    )

    dataset = KittiOdometryDataset(config, SEQUENCE_NUM)

    config_filtered = KittiOdometryDatasetConfig(
        cache=True,
        dataset_path=DATASET_PATH,
        correct_scan_calibration=True,
        filters=FilterList(
            [
                KittiGTMovingObjectFilter(
                    os.path.join(
                        DATASET_PATH,
                        "sequences",
                        "%.2d" % SEQUENCE_NUM,
                        "labels",
                    )
                ),
                RangeFilter(2.5, 120),
                ApplyPose(),
            ]
        ),
    )
    path_calib = os.path.join(DATASET_PATH,"sequences","%.2d" % SEQUENCE_NUM,"calib.txt")

    dataset_filtered = KittiOdometryDataset(config_filtered, SEQUENCE_NUM)
 
    points_index = 20
    points_cleaned_o3d = get_pcd(dataset_filtered.get_point_cloud(points_index))

    image_left_PIL = dataset.get_image("cam2", points_index)
    image_right_PIL = dataset.get_image("cam3", points_index)

    image_left = cv2.cvtColor(np.array(image_left_PIL), cv2.COLOR_RGB2BGR)
    image_right = cv2.cvtColor(np.array(image_right_PIL), cv2.COLOR_RGB2BGR)

    ### Open Velodyne Point Cloud, RGB Image and Calibration File
    points_cleaned_np = np.asarray(points_cleaned_o3d.points)

    img = image_left

    points_cleaned_np_hpr = hidden_point_removal(points_cleaned_np, camera=[0,0,1.73])

    overlay_with_removal, inds_with_removal = project_point_cloud_to_image(path_calib, points_cleaned_np_hpr, img)

    overlay_without_removal, inds_without_removal = project_point_cloud_to_image(path_calib, points_cleaned_np, img) 

    cv2.imwrite('/Users/laurenzheidrich/Downloads/rem.jpg', overlay_with_removal)    
    cv2.imwrite('/Users/laurenzheidrich/Downloads/non.jpg', overlay_without_removal) 

    points_cleaned_o3d_hpr_fov = o3d.geometry.PointCloud()
    points_cleaned_o3d_hpr_fov.points = o3d.utility.Vector3dVector(points_cleaned_np_hpr[inds_with_removal, :])

    points_cleaned_o3d_fov = o3d.geometry.PointCloud()
    points_cleaned_o3d_fov.points = o3d.utility.Vector3dVector(points_cleaned_np[inds_without_removal, :])

    o3d.visualization.draw_geometries([points_cleaned_o3d_hpr_fov])
    o3d.visualization.draw_geometries([points_cleaned_o3d_fov])