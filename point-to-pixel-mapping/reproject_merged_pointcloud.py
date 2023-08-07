import numpy as np
import cv2
from scipy.sparse import lil_matrix
from point_cloud_utils import transform_pcd, get_pcd, point_to_label
from point_to_pixels import point_to_pixel

def reproject_merged_pointcloud(pcd_merge, dataset, start_index, sequence_length):
    '''
    Args:
        pcd_merge:          merged point cloud
        dataset:            dataset object
        start_index:        index of first point cloud
        sequence_length:    number of point clouds to merge
    Returns:
        feature_matrix:     feature matrix of merged point cloud
    '''

    left_cam = "cam2"

    pose_merge = dataset.get_pose(start_index)
    pcd_merge_t0 = transform_pcd(pcd_merge, pose_merge)
    num_points = pcd_merge_t0.shape[0]

    feature_matrix = lil_matrix((num_points, 0), dtype=np.int32)

    for i in range(sequence_length):
        points_index = start_index + i

        T_lidar2leftcam, K_leftcam = dataset.get_calibration_matrices(left_cam)
        
        left_label_PIL = dataset.get_sam_label(left_cam, points_index)
        left_label = cv2.cvtColor(np.array(left_label_PIL), cv2.COLOR_RGB2BGR)

        pose_to = np.linalg.inv(dataset.get_pose(points_index))

        pcd_merge_velo_ti = transform_pcd(pcd_merge_t0, pose_to)
        pcd_merge_leftcamframe = transform_pcd(pcd_merge_velo_ti, T_lidar2leftcam)

        point_to_pixel_dict_leftcam = point_to_pixel(pcd_merge_leftcamframe, K_leftcam, left_label.shape[0], left_label.shape[1])

        point_to_label_dict_leftcam = point_to_label(point_to_pixel_dict_leftcam, left_label)

        instance_ids = list(set(point_to_label_dict_leftcam.values()))
       
        features = lil_matrix((num_points, len(instance_ids)), dtype=np.int32)

        for index, instance_id in point_to_label_dict_leftcam.items():
            features[index, instance_ids.index(instance_id)] = 1

        old_shape = feature_matrix.get_shape()
        new_shape = (old_shape[0], old_shape[1] + len(instance_ids))
        new_feature_matrix = lil_matrix(new_shape, dtype=np.int32)
        new_feature_matrix[:, :old_shape[1]] = feature_matrix
        new_feature_matrix[:, old_shape[1]:] = features
        feature_matrix = new_feature_matrix

    return feature_matrix