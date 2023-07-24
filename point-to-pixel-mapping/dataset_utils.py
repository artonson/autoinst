import cv2
import numpy as np 
from point_cloud_utils import get_pcd

def get_data_from_dataset(dataset, points_index, left_cam, right_cam):
    """
    Returns the point cloud, left and right images, left and right labels, and the calibration matrices for a given index in the dataset.
    Args:
        dataset:        The kitty dataset object
        points_index:   The index of the point cloud in the dataset
        left_cam:       The left camera name
        right_cam:      The right camera name
    """

    pcd_o3d = get_pcd(dataset.get_point_cloud(points_index))
    pcd = np.asarray(pcd_o3d.points)

    left_image_PIL = dataset.get_image(left_cam, points_index)
    left_image = cv2.cvtColor(np.array(left_image_PIL), cv2.COLOR_RGB2BGR)

    right_image_PIL = dataset.get_image(right_cam, points_index)
    right_image = cv2.cvtColor(np.array(right_image_PIL), cv2.COLOR_RGB2BGR)

    left_label_PIL = dataset.get_sam_label(left_cam, points_index)
    left_label = cv2.cvtColor(np.array(left_label_PIL), cv2.COLOR_RGB2BGR)

    right_label_PIL = dataset.get_sam_label(right_cam, points_index)
    right_label = cv2.cvtColor(np.array(right_label_PIL), cv2.COLOR_RGB2BGR)

    T_lidar2leftcam, K_leftcam = dataset.get_calibration_matrices(left_cam)
    T_lidar2rightcam, K_rightcam = dataset.get_calibration_matrices(right_cam)

    return pcd, (left_image, right_image), (left_label, right_label), (T_lidar2leftcam, T_lidar2rightcam), (K_leftcam, K_rightcam)