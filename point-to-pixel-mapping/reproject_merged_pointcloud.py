import numpy as np
import cv2
from scipy.sparse import lil_matrix
from point_cloud_utils import transform_pcd, get_pcd, point_to_label, change_point_indices
from point_to_pixels import point_to_pixel
from hidden_points_removal import hidden_point_removal_o3d

def reproject_points_to_label(pcd, T_pcd2world, label, T_world2cam, K, hidden_point_removal=True, hpr_radius = 1000, label_is_color=True, return_hpr_mask=False):
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
        hpr_mask = hidden_point_removal_o3d(pcd_camframe, camera=[0,0,0], radius_factor=hpr_radius)
        pcd_camframe = pcd_camframe[hpr_mask] 

    point_to_pixel_dict = point_to_pixel(pcd_camframe, K, label.shape[0], label.shape[1])

    point_to_label_dict = point_to_label(point_to_pixel_dict, label, label_is_color=label_is_color)

    if hidden_point_removal:
        point_to_label_dict = change_point_indices(point_to_label_dict, hpr_mask)

    if return_hpr_mask:
        return point_to_label_dict, hpr_mask
    else:
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

def merge_features(feature_reprojections, num_points):
    '''
    Args:
        feature_reprojections: list of dicts that map point indices to feature vectors
        num_points:            number of points in point cloud
    Returns:
        features:              matrix that stores mean feature vector for each point
    '''

    num_features = feature_reprojections[0][next(iter(feature_reprojections[0]))].shape[0]

    feature_list = [[] for i in range(num_points)]

    for reprojection in feature_reprojections:
        for index, feature_vec in reprojection.items():
            feature_list[index].append(feature_vec)

    features = np.zeros((num_points, num_features))

    i = 0
    for feature in feature_list:
        if len(feature) > 0:
            features[i, :] = np.mean(feature, axis=0)
        i += 1

    return features