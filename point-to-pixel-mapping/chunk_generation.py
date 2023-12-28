import numpy as np
from scipy.spatial.distance import cdist
import open3d as o3d
import os
from point_cloud_utils import transform_pcd, get_pcd, change_point_indices, get_statistical_inlier_indices, angle_between, get_subpcd
from image_utils import masks_to_image
from hidden_points_removal import hidden_point_removal_o3d
from point_to_pixels import point_to_pixel
import copy
import cv2
import scipy
from visualization_utils import color_pcd_by_labels

'''
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
'''
def subsample_positions(positions, voxel_size=1):
    positions = np.array(positions)
    min_vals = positions.min(axis=0)
    max_vals = positions.max(axis=0)

    centers = [np.arange(min_val, max_val + voxel_size, voxel_size) for min_val, max_val in zip(min_vals, max_vals)]
    grid = np.stack(np.meshgrid(*centers, indexing='ij'), -1).reshape(-1, 3)

    closest_pose_indices = np.argmin(cdist(grid, positions), axis=1)
    unique_indices = np.unique(closest_pose_indices)

    subsampled_indices = []
    for index in unique_indices:
        closest_pose = positions[index]
        distance = np.abs(grid[closest_pose_indices == index] - closest_pose)
        if np.all(distance < 0.5 * voxel_size, axis=1).any():
            subsampled_indices.append(index)

    return np.sort(subsampled_indices)

def chunks_from_pointcloud(pcd, T_pcd, positions, first_position, indices, R, overlap,labels=None,ground=False):

    points = np.asarray(pcd.points)

    pcd_chunks = []
    chunk_indices = []
    center_pos = []
    center_ids = []
    chunk_bounds = []
    
    if labels != None : 
     kitti_out = {'panoptic':[],'semantic':[],'instance':[]}
    else : 
        kitti_out = None 
    distance = 0
    last_position = None
    for (position, index) in zip(positions, indices):
        if last_position is not None:
            distance += np.linalg.norm(position - last_position)
            if distance > (R[0]-overlap): # New chunk

                pos_pcd = position - first_position
                rot = np.linalg.inv(T_pcd[:3,:3])
                pos_pcd = rot @ pos_pcd

                max_position = pos_pcd + (0.5 * R)
                min_position = pos_pcd - (0.5 * R) 
                

                ids = np.where(np.all(points > min_position, axis=1) & np.all(points < max_position, axis=1))[0]
                pcd_cut = pcd.select_by_index(ids)

                inlier_indices = get_statistical_inlier_indices(pcd_cut)
                pcd_cut_final = get_subpcd(pcd_cut, inlier_indices)
                
                
                if labels != None : 
                    if ground == False :
                        kitti_out['panoptic'].append(labels['panoptic_nonground'][ids][inlier_indices])
                        kitti_out['semantic'].append(labels['seg_nonground'][ids][inlier_indices])
                        kitti_out['instance'].append(labels['instance_nonground'][ids][inlier_indices])
                    else : 
                        kitti_out['panoptic'].append(labels['panoptic_ground'][ids][inlier_indices])
                        kitti_out['semantic'].append(labels['seg_ground'][ids][inlier_indices])
                        kitti_out['instance'].append(labels['instance_ground'][ids][inlier_indices])

                #pcd = color_pcd_by_labels(pcd_cut_final,labels['panoptic_nonground'][ids][inlier_indices])
                #o3d.visualization.draw_geometries([pcd])
                
                pcd_chunks.append(pcd_cut_final)      
                chunk_indices.append(ids)
                center_pos.append(pos_pcd)
                center_ids.append(index)
                chunk_bounds.append(((pos_pcd - 0.5 * R), (pos_pcd + 0.5 * R)))
                
                distance = 0
        last_position = position

    return pcd_chunks, chunk_indices, center_pos, center_ids, chunk_bounds, kitti_out

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

def tarl_features_per_patch(dataset, pcd, T_pcd, center_position, tarl_indices, chunk_size, search_radius=0.1):

    concatenated_tarl_points = np.zeros((0, 3))
    concatenated_tarl_features = np.zeros((0, 96))

    num_points = np.asarray(pcd.points).shape[0]

    max_position = center_position + 0.5 * chunk_size
    min_position = center_position - 0.5 * chunk_size

    for points_index in tarl_indices:
        
        # Load the TARL features & points
        tarl_features = dataset.get_tarl_features(points_index)
        coords = dataset.get_point_cloud(points_index)

        # Transform the coordinates 
        T_lidar2world = dataset.get_pose(points_index)
        T_local2global = np.linalg.inv(T_pcd) @ T_lidar2world
        coords = transform_pcd(coords, T_local2global)
        
        mask = np.where(np.all(coords > min_position, axis=1) & np.all(coords < max_position, axis=1))[0]

        coords, tarl_features = coords[mask], tarl_features[mask]
        concatenated_tarl_points = np.concatenate((concatenated_tarl_points, coords))
        concatenated_tarl_features = np.concatenate((concatenated_tarl_features, tarl_features))

    tarl_pcd = get_pcd(concatenated_tarl_points)
    tarl_tree = o3d.geometry.KDTreeFlann(tarl_pcd)
    tarl_features = np.zeros((num_points, 96))

    for i, point in enumerate(np.asarray(pcd.points)):
        [_, idx, _] = tarl_tree.search_radius_vector_3d(point, search_radius)
        features_in_radius = concatenated_tarl_features[idx]
  
        if not features_in_radius.shape[0]==0:
            tarl_features[i,:] = np.mean(features_in_radius, axis=0)
        else:
            continue

    return tarl_features

def get_indices_feature_reprojection(global_indices, first_id, adjacent_frames=(8,5)):
    
    first_index = global_indices.index(first_id)
    cam_indices_global = global_indices[max(0,first_index-adjacent_frames[0]):first_index+adjacent_frames[1]]
    indices = []
    for global_i in cam_indices_global:
        indices.append(global_indices.index(global_i))

    return cam_indices_global, indices

def is_perpendicular_and_upward(point, normal, boundary = 0.1):
    '''
    Args:
        point: 3D point
        normal: normal vector of 3D point
        boundary: boundary around pi/2 to be considered perpendicular
    Returns:
        True if point is perpendicular to normal and pointing upwards, False otherwise
    '''
    angle = np.abs(angle_between(point, normal))
    perpendicular = (angle > (np.pi / 2 - boundary) and  angle < (np.pi / 2 + boundary)) or (angle > (3 * np.pi / 2 - boundary) and  angle < (3 * np.pi / 2 + boundary))
    upward = (normal[2]*normal[2]) > (normal[0]*normal[0] + normal[1]*normal[1])

    return (perpendicular and upward)


def image_based_features_per_patch(dataset, pcd, chunk_indices, T_pcd2world, cam_indices, cams, cam_id, hpr_radius=1000, num_dino_features = 384, hpr_masks=None, dino=True, rm_perp=0.0):

    num_points = np.asarray(pcd.points).shape[0]
    point2sam = (-1) * np.ones((num_points, len(cam_indices)), dtype=int) # -1 indicates no association

    if rm_perp:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=200))

    if dino:
        point2dino = np.zeros((num_points, len(cam_indices), num_dino_features))

    if hpr_masks is not None:
        assert len(cam_indices) == hpr_masks.shape[0]

    for i, points_index in enumerate(cam_indices):

        # Load the calibration matrices
        T_lidar2world = dataset.get_pose(points_index)
        T_world2lidar = np.linalg.inv(T_lidar2world)
        T_lidar2cam, K = dataset.get_calibration_matrices(cams[cam_id])
        T_world2cam = T_lidar2cam @ T_world2lidar
        T_pcd2cam = T_world2cam @ T_pcd2world

        #hidden point removal
        pcd_camframe = copy.deepcopy(pcd).transform(T_pcd2cam)
        if hpr_masks is None:
            hpr_bounds = np.array([25,25,25])
            bound_indices = np.where(np.all(np.asarray(pcd_camframe.points) > -hpr_bounds, axis=1) & np.all(np.asarray(pcd_camframe.points) < hpr_bounds, axis=1))[0]
            pcd_camframe_hpr = get_subpcd(pcd_camframe, bound_indices)
            visible_indices = hidden_point_removal_o3d(np.asarray(pcd_camframe_hpr.points), camera=[0,0,0], radius_factor=hpr_radius)
            visible_indices = bound_indices[visible_indices]
        else:
            visible_indices = np.where(hpr_masks[i])[0]
        frame_indices = list(set(visible_indices) & set(chunk_indices))

        # Load the SAM label
        sam_masks = dataset.get_sam_mask(cams[cam_id], points_index)
        sam_labels = masks_to_image(sam_masks)

        # Load the DINOV2 feature map
        if dino:
            dinov2_feature_map = dataset.get_dinov2_features(cams[cam_id], points_index)
            dinov2_feature_map_zoomed = scipy.ndimage.zoom(dinov2_feature_map, (sam_labels.shape[0] / dinov2_feature_map.shape[0], sam_labels.shape[1] / dinov2_feature_map.shape[1], 1), order=0)

        #chunk generation
        if rm_perp:
            map_visible = get_subpcd(pcd_camframe, frame_indices, normals=True)
            T_cam2pcd_orientation = np.linalg.inv(copy.deepcopy(T_pcd2cam))
            T_cam2pcd_orientation[:3,3] = np.zeros(3) # We only want to fix the orientation, not the position
            map_visible_fixed_orientation = copy.deepcopy(map_visible).transform(T_cam2pcd_orientation)
            map_visible_points = np.asarray(map_visible_fixed_orientation.points)
            map_visible_normals = np.asarray(map_visible_fixed_orientation.normals)
        else:
            map_visible = get_subpcd(pcd_camframe, frame_indices)

        points_to_pixels = point_to_pixel(np.asarray(map_visible.points), K, sam_labels.shape[0], sam_labels.shape[1])


        for point_id, pixel_id in points_to_pixels.items():
            pixel = pixel_id["pixels"]
            label = sam_labels[pixel[1], pixel[0]]
            if rm_perp:
                valid = not is_perpendicular_and_upward(map_visible_points[point_id], map_visible_normals[point_id], boundary=rm_perp)
            else:
                valid = True
            if label and valid:
                point2sam[frame_indices[point_id], i] = label
            if dino and valid:
                point2dino[frame_indices[point_id], i, :] = dinov2_feature_map_zoomed[pixel[1], pixel[0], :]

    pcd_chunk = get_subpcd(pcd, chunk_indices)
    point2sam_chunk = point2sam[chunk_indices]
    
    if dino:
        point2dino_chunk = point2dino[chunk_indices]

    inlier_indices = get_statistical_inlier_indices(pcd_chunk)
    pcd_chunk_final = get_subpcd(pcd_chunk, inlier_indices)

    point2sam_chunk_final = point2sam_chunk[inlier_indices]

    if dino:
        point2dino_chunk_final = point2dino_chunk[inlier_indices]
        return point2sam_chunk_final, point2dino_chunk_final, pcd_chunk_final
    else:
        return point2sam_chunk_final, pcd_chunk_final


def dinov2_mean(point2dino):
    # Compute mean of DINOV2 features over number of views
    point2dino_mean = np.zeros((point2dino.shape[0], point2dino.shape[2]))
    non_zero_mask = point2dino.any(axis=2)
    for i in range(point2dino.shape[0]):
        features = point2dino[i][non_zero_mask[i]]
        if features.shape[0] != 0:
            point2dino_mean[i] = np.mean(features, axis=0)
    return point2dino_mean