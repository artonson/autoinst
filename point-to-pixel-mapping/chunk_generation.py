import numpy as np
from scipy.spatial.distance import cdist
import open3d as o3d
import os
from point_cloud_utils import transform_pcd, get_pcd
from reproject_merged_pointcloud import reproject_points_to_label
import cv2
import scipy

def subsample_positions(positions, voxel_size=1):

    min_x = min(p[0] for p in positions)
    max_x = max(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    max_y = max(p[1] for p in positions)
    min_z = min(p[2] for p in positions)
    max_z = max(p[2] for p in positions)

    x_centers = np.arange(min_x, max_x + voxel_size, voxel_size)
    y_centers = np.arange(min_y, max_y + voxel_size, voxel_size)
    z_centers = np.arange(min_z, max_z + voxel_size, voxel_size)

    subsampled_indices = []

    for x_center in x_centers:
        for y_center in y_centers:
            for z_center in z_centers:
                chunk_center = np.array([x_center, y_center, z_center])
                closest_pose_index = np.argmin(cdist([chunk_center], positions))
                closest_pose = positions[closest_pose_index]
                
                distance = np.abs(np.array(closest_pose) - chunk_center)
                if distance[0] < 0.5 * voxel_size and distance[1] < 0.5 * voxel_size and distance[2] < 0.5 * voxel_size:
                    subsampled_indices.append(closest_pose_index)

    subsampled_indices = np.sort(subsampled_indices)

    return subsampled_indices

def chunks_from_pointcloud(pcd, T_pcd, positions, poses, first_position, indices, chunk_size, overlap):

    points = np.asarray(pcd.points)

    pcd_chunks = []
    center_pos = []
    center_ids = []

    distance = 0
    last_position = None
    for (position, pose, index) in zip(positions, poses, indices):
        if last_position is not None:
            distance += np.linalg.norm(position - last_position)
            if distance > (chunk_size[0]-overlap): # New chunk

                pos_pcd = position - first_position
                rot = np.linalg.inv(T_pcd[:3,:3])
                pos_pcd = rot @ pos_pcd

                max_position = pos_pcd + (0.5 * chunk_size)
                min_position = pos_pcd - (0.5 * np.array([chunk_size[0], chunk_size[1], 4.4])) # 4.4 to cut away points below street -> artefacts

                mask = np.where(np.all(points > min_position, axis=1) & np.all(points < max_position, axis=1))[0]
                pcd_cut = pcd.select_by_index(mask)
        
                pcd_chunks.append(pcd_cut)                
                center_pos.append(pos_pcd)
                center_ids.append(index)
                
                distance = 0
        last_position = position

    return pcd_chunks, center_pos, center_ids

def indices_per_patch(T_pcd, center_positions, positions, first_position, global_indices, chunk_size):

    patchwise_indices = []

    for center in center_positions:
        indices = []

        for (position, index) in zip(positions, global_indices):

            pos_pcd = position - first_position
            rot = np.linalg.inv(T_pcd[:3,:3])
            pos_pcd = rot @ pos_pcd

            if np.linalg.norm(center - pos_pcd) < 0.5 * chunk_size[1]:
                indices.append(index)
        patchwise_indices.append(indices)

    return patchwise_indices

def tarl_features_per_patch(dataset, pcd, center_id, T_pcd, center_position, global_indices, chunk_size, search_radius):

    concatenated_tarl_points = np.zeros((0, 3))
    concatenated_tarl_features = np.zeros((0, 96))

    center_index = global_indices.index(center_id)

    num_points = np.asarray(pcd.points).shape[0]

    for points_index in global_indices[max(0,center_index-11):min(len(global_indices)-1,center_index+11)]:
        
        # Load the TARL features & points
        tarl_features = dataset.get_tarl_features(points_index)
        coords = dataset.get_point_cloud(points_index)

        # Only take those within a certain range
        norm = np.linalg.norm(coords, axis=1)
        norm = np.logical_and(norm <= 25, norm >= 3)
        tarl_features = tarl_features[norm]
        coords = coords[norm]

        # Transform the coordinates 
        T_lidar2world = dataset.get_pose(points_index)
        T_local2global_pcd = np.linalg.inv(T_pcd) @ T_lidar2world
        coords = transform_pcd(coords, T_local2global_pcd)
        
        max_position = center_position + (0.5 * chunk_size)
        min_position = center_position - (0.5 * np.array([chunk_size[0], chunk_size[1], 4.4])) # 4.4 to cut away points below street -> artefacts

        mask = np.where(np.all(coords > min_position, axis=1) & np.all(coords < max_position, axis=1))[0]

        coords, tarl_features = coords[mask], tarl_features[mask]
        concatenated_tarl_points = np.concatenate((concatenated_tarl_points, coords))
        concatenated_tarl_features = np.concatenate((concatenated_tarl_features, tarl_features))

    tarl_pcd = get_pcd(concatenated_tarl_points)
    tarl_tree = o3d.geometry.KDTreeFlann(tarl_pcd)
    tarl_features = np.zeros((num_points, 96))

    i=0
    for point in np.asarray(pcd.points):
        [_, idx, _] = tarl_tree.search_radius_vector_3d(point, search_radius)
        features_in_voxel = concatenated_tarl_features[idx]
        
        increment = 1.0
        while features_in_voxel.shape[0]==0:
            increment += 0.2
            [_, idx, _] = tarl_tree.search_radius_vector_3d(point, increment * search_radius)
            features_in_voxel = concatenated_tarl_features[idx]

        tarl_features[i,:] = np.mean(features_in_voxel, axis=0)
        i+=1

    return tarl_features

def image_based_features_per_patch(dataset, pcd, T_pcd, global_indices, first_id, cams, cam_id):

    point_to_sam_label_reprojections = []
    point_to_dinov2_feature_reprojections = []

    first_index = global_indices.index(first_id)

    for points_index in global_indices[max(0,first_index-8):first_index+5]:

        # Load the SAM label
        label_PIL = dataset.get_sam_label(cams[cam_id], points_index)
        label = cv2.cvtColor(np.array(label_PIL), cv2.COLOR_RGB2BGR)

        # Load the DINOV2 feature map
        dinov2_feature_map = dataset.get_dinov2_features(cams[cam_id], points_index)
        dinov2_feature_map_zoomed = scipy.ndimage.zoom(dinov2_feature_map, (label.shape[0] / dinov2_feature_map.shape[0], label.shape[1] / dinov2_feature_map.shape[1], 1), order=0)
        
        # Load the calibration matrices
        T_lidar2world = dataset.get_pose(points_index)
        T_world2lidar = np.linalg.inv(dataset.get_pose(points_index))
        T_lidar2cam, K = dataset.get_calibration_matrices(cams[cam_id])
        T_world2cam = T_lidar2cam @ T_world2lidar
        
        # Compute the SAM label reporjections
        point_to_sam_label_reprojections.append(reproject_points_to_label(np.array(pcd.points), T_pcd, label, T_world2cam, K, hidden_point_removal=True))

        # Compute the DINOV2 feature map reprojections
        point_to_dinov2_feature_reprojections.append(reproject_points_to_label(np.array(pcd.points), T_pcd, dinov2_feature_map_zoomed, T_world2cam, K, hidden_point_removal=True, label_is_color=False))

    return point_to_sam_label_reprojections, point_to_dinov2_feature_reprojections