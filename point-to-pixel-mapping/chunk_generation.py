import numpy as np
from scipy.spatial.distance import cdist
import open3d as o3d
import os
from point_cloud_utils import transform_pcd, get_pcd, change_point_indices, get_statistical_inlier_indices, get_subpcd
from image_utils import masks_to_image
from hidden_points_removal import hidden_point_removal_o3d
from point_to_pixels import point_to_pixel
import copy
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

def chunks_from_pointcloud(pcd, T_pcd, positions, first_position, indices, R, overlap):

    points = np.asarray(pcd.points)

    pcd_chunks = []
    chunk_indices = []
    center_pos = []
    center_ids = []
    chunk_bounds = []

    distance = 0
    last_position = None
    for (position, index) in zip(positions, indices):
        if last_position is not None:
            distance += np.linalg.norm(position - last_position)
            if distance > (R[0]-overlap): # New chunk

                pos_pcd = position - first_position
                rot = np.linalg.inv(T_pcd[:3,:3])
                pos_pcd = rot @ pos_pcd

                ids = np.where(np.all(points > (pos_pcd - 0.5 * R), axis=1) & np.all(points < (pos_pcd + 0.5 * R), axis=1))[0]
                pcd_cut = pcd.select_by_index(ids)

                inlier_indices = get_statistical_inlier_indices(pcd_cut)
                pcd_cut_final = get_subpcd(pcd_cut, inlier_indices)
        
                pcd_chunks.append(pcd_cut_final)      
                chunk_indices.append(ids)
                center_pos.append(pos_pcd)
                center_ids.append(index)
                chunk_bounds.append(((pos_pcd - 0.5 * R), (pos_pcd + 0.5 * R)))
                
                distance = 0
        last_position = position

    return pcd_chunks, chunk_indices, center_pos, center_ids, chunk_bounds

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

def tarl_features_per_patch(dataset, pcd, center_id, T_pcd, center_position, global_indices, chunk_size, search_radius, adjacent_frames=11, min_z=4.4):

    concatenated_tarl_points = np.zeros((0, 3))
    concatenated_tarl_features = np.zeros((0, 96))

    center_index = global_indices.index(center_id)

    num_points = np.asarray(pcd.points).shape[0]

    for points_index in global_indices[max(0,center_index-adjacent_frames):min(len(global_indices)-1,center_index+adjacent_frames)]:
        
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
        min_position = center_position - (0.5 * np.array([chunk_size[0], chunk_size[1], min_z])) # min_z to cut away points below street -> artefacts

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


def image_based_features_per_patch(dataset, pcd, chunk_indices, T_pcd2world, global_indices, first_id, cams, cam_id, adjacent_frames=(8,5), hpr_radius=1000, num_dino_features = 384):

    first_index = global_indices.index(first_id)
    cam_indices = global_indices[max(0,first_index-adjacent_frames[0]):first_index+adjacent_frames[1]]

    num_points = np.asarray(pcd.points).shape[0]
    point2sam = (-1) * np.ones((num_points, len(cam_indices)), dtype=int) # -1 indicates no association
    point2dino = np.zeros((num_points, len(cam_indices), num_dino_features))

    for i, points_index in enumerate(cam_indices):

        # Load the calibration matrices
        T_lidar2world = dataset.get_pose(points_index)
        T_world2lidar = np.linalg.inv(T_lidar2world)
        T_lidar2cam, K = dataset.get_calibration_matrices(cams[cam_id])
        T_world2cam = T_lidar2cam @ T_world2lidar
        T_pcd2cam = T_world2cam @ T_pcd2world

        #hidden point removal
        pcd_camframe = copy.deepcopy(pcd).transform(T_pcd2cam)
        visible_indices = hidden_point_removal_o3d(np.asarray(pcd_camframe.points), camera=[0,0,0], radius_factor=hpr_radius)
        frame_indices = list(set(visible_indices) & set(chunk_indices))

        # Load the SAM label
        sam_masks = dataset.get_sam_mask(cams[cam_id], points_index)
        sam_labels = masks_to_image(sam_masks)

        # Load the DINOV2 feature map
        dinov2_feature_map = dataset.get_dinov2_features(cams[cam_id], points_index)
        dinov2_feature_map_zoomed = scipy.ndimage.zoom(dinov2_feature_map, (sam_labels.shape[0] / dinov2_feature_map.shape[0], sam_labels.shape[1] / dinov2_feature_map.shape[1], 1), order=0)

        #chunk generation
        map_visible = get_subpcd(pcd_camframe, frame_indices)
        points_to_pixels = point_to_pixel(np.asarray(map_visible.points), K, sam_labels.shape[0], sam_labels.shape[1])

        for point_id, pixel_id in points_to_pixels.items():
            pixel = pixel_id["pixels"]
            point2sam[frame_indices[point_id], i] = sam_labels[pixel[1], pixel[0]]
            point2dino[frame_indices[point_id], i, :] = dinov2_feature_map_zoomed[pixel[1], pixel[0], :]

    pcd_chunk = get_subpcd(pcd, chunk_indices)
    point2sam_chunk = point2sam[chunk_indices]
    point2dino_chunk = point2dino[chunk_indices]

    inlier_indices = get_statistical_inlier_indices(pcd_chunk)
    pcd_chunk_final = get_subpcd(pcd_chunk, inlier_indices)

    point2sam_chunk_final = point2sam_chunk[inlier_indices]

    point2dino_chunk_final = point2dino_chunk[inlier_indices]

    return point2sam_chunk_final, point2dino_chunk_final, pcd_chunk_final


def dinov2_mean(point2dino):
    # Compute mean of DINOV2 features over number of views
    point2dino_mean = np.zeros((point2dino.shape[0], point2dino.shape[2]))
    non_zero_mask = point2dino.any(axis=2)
    for i in range(point2dino.shape[0]):
        features = point2dino[i][non_zero_mask[i]]
        if features.shape[0] != 0:    
            point2dino_mean[i] = np.mean(features, axis=0)
    return point2dino_mean