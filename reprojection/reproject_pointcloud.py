import numpy as np
import cv2
from scipy.sparse import lil_matrix
from point_cloud_utils import transform_pcd, get_pcd, point_to_label, change_point_indices
from point_to_pixels import point_to_pixel
from hidden_points_removal import hidden_point_removal_o3d

def reproject_points_to_label(pcd, T_pcd2world, label, T_world2cam, K, hidden_point_removal=True):
    '''
    Args:
        pcd:            point cloud in camera coordinate [npoints, 3]
        T_pcd2world:    transformation matrix from point cloud to world coordinate
        label:          label image
        T_world2cam:    transformation matrix from world to camera coordinate
        K:              camera intrinsic matrix
    Returns:
        point_to_label_dict: dict that maps point indices to label
    '''

    T_pcd2cam = T_world2cam @ T_pcd2world

    pcd_camframe = transform_pcd(pcd, T_pcd2cam)

    if hidden_point_removal:
        hpr_mask = hidden_point_removal_o3d(pcd_camframe, camera=[0,0,0], radius_factor=1000)
        pcd_camframe = pcd_camframe[hpr_mask] 

    point_to_pixel_dict = point_to_pixel(pcd_camframe, K, label.shape[0], label.shape[1])

    point_to_label_dict = point_to_label(point_to_pixel_dict, label, label_is_color=False)

    if hidden_point_removal:
        point_to_label_dict = change_point_indices(point_to_label_dict, hpr_mask)

    return point_to_label_dict


def merge_associations(associations, num_points):
    '''
    Args:
        associations:       list of dicts that map point indices to instance ids
        num_points:         number of points in point cloud
    Returns:
        association_matrix: sparse matrix that maps point indices to instance ids
    '''

    instance_ids = []
    num_features = 0
    for association in associations:
        instance_ids.append(list(set(association.values())))
        num_features += len(instance_ids[-1])

    association_matrix = lil_matrix((num_points, num_features), dtype=np.int32)
    
    offset = 0
    num_iteration = 0
    for association in associations:
        for index, instance_id in association.items():
            association_matrix[index, offset + instance_ids[num_iteration].index(instance_id)] = 1
        offset += len(instance_ids[num_iteration])
        num_iteration += 1

    return association_matrix