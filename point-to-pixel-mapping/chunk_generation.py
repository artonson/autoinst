import numpy as np
from scipy.spatial.distance import cdist
import open3d as o3d
import os
from point_cloud_utils import transform_pcd, get_pcd, change_point_indices, get_statistical_inlier_indices, angle_between, get_subpcd, kDTree_1NN_feature_reprojection
from image_utils import masks_to_image
from hidden_points_removal import hidden_point_removal_o3d
from point_to_pixels import point_to_pixel
import umap
import copy
import cv2
import scipy
from visualization_utils import color_pcd_by_labels
from scipy.spatial import cKDTree
from tqdm import tqdm 
import time


import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

def subsample_positions(positions, voxel_size=1, batch_size=1000):
    positions = np.array(positions)
    min_vals = positions.min(axis=0)
    max_vals = positions.max(axis=0)

    centers = [np.arange(min_val, max_val + voxel_size, voxel_size) for min_val, max_val in zip(min_vals, max_vals)]
    grid = np.stack(np.meshgrid(*centers, indexing='ij'), -1).reshape(-1, 3)

    subsampled_indices = []
    total_batches = len(grid) // batch_size + (1 if len(grid) % batch_size != 0 else 0)
    with tqdm(total=total_batches) as pbar:
        for i in range(0, len(grid), batch_size):
            batch_grid = grid[i:i + batch_size]
            distances = cdist(batch_grid, positions)
            closest_pose_indices = np.argmin(distances, axis=1)
            unique_indices = np.unique(closest_pose_indices)
            for index in unique_indices:
                closest_pose = positions[index]
                distance = np.abs(batch_grid[closest_pose_indices == index] - closest_pose)
                if np.all(distance < 0.5 * voxel_size, axis=1).any():
                    subsampled_indices.append(index)
            pbar.update(1)

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
                        kitti_out['semantic'].append(labels['seg_nonground'][ids][inlier_indices])
                        kitti_out['instance'].append(labels['instance_nonground'][ids][inlier_indices])
                    else : 
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

def tarl_features_per_patch(dataset, pcd, T_pcd, center_position, tarl_indices, chunk_size, search_radius=0.1,norm=False):

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
            if norm : 
                tarl_features[i] /= np.linalg.norm(tarl_features[i])
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


def image_based_features_per_patch(dataset, pcd, chunk_indices, chunk_nc, voxel_size, T_pcd2world, cam_indices, cams, cam_ids: list, hpr_radius=1000, 
                                    num_dino_features = 384, hpr_masks=None, sam = True, dino=True, rm_perp=0.0,pcd_chunk=None):

    num_points_nc = np.asarray(chunk_nc.points).shape[0]

    pcd_chunk = get_subpcd(pcd, chunk_indices)
    inlier_indices_of_chunk = get_statistical_inlier_indices(pcd_chunk)
    chunk_and_inlier_indices = chunk_indices[inlier_indices_of_chunk]

    if rm_perp:
        pcd_chunk_final = get_subpcd(pcd_chunk, inlier_indices_of_chunk)
        pcd_chunk_final.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=200))

    if sam:
        point2sam_list = []
    if dino:
        point2dino_list = []

    for cam_id in cam_ids:

        if sam:
            point2sam_nc = (-1) * np.ones((num_points_nc, len(cam_indices)), dtype=int) # -1 indicates no association

        if dino:
            point2dino_nc = np.zeros((num_points_nc, len(cam_indices), num_dino_features))

        sam_mask_0 = dataset.get_sam_mask(cams[cam_id], 0)
        sam_label_0 = masks_to_image(sam_mask_0)
        label_shape = sam_label_0.shape

        if hpr_masks is not None:
            assert len(cam_indices) == hpr_masks.shape[0]

        for i, points_index in enumerate(cam_indices):
            start_loop = time.time()
            # Load the calibration matrices
            start = time.time() 
            T_lidar2world = dataset.get_pose(points_index)
            T_world2lidar = np.linalg.inv(T_lidar2world)
            T_lidar2cam, K = dataset.get_calibration_matrices(cams[cam_id])
            T_world2cam = T_lidar2cam @ T_world2lidar
            T_pcd2cam = T_world2cam @ T_pcd2world

            #hidden point removal
            new_pcd = o3d.geometry.PointCloud() 
            new_pcd.points  = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())
            pcd_camframe_world = copy.deepcopy(new_pcd).transform(dataset.get_pose(0))
            pcd_camframe = copy.deepcopy(pcd).transform(T_pcd2cam)
            
            if dataset.nuscenes == False : ##speedup currently only supported by KITTI 
                pts = np.asarray(pcd_chunk.points)
                min_x,min_y,min_z = pts[:,0].min(), pts[:,1].min(), pts[:,2].min()
                max_x,max_y,max_z = pts[:,0].max(), pts[:,1].max(), pts[:,2].max()
                min_bound = np.array([min_x,min_y,min_z]) 
                max_bound = np.array([max_x,max_y,max_z])  
                pcd_camframe_world.paint_uniform_color([0,0,1])
                #o3d.visualization.draw_geometries([pcd_camframe_world + pcd_chunk])
            
            
            if hpr_masks is None:
                
                hpr_bounds = np.array([25,25,25])
                if dataset.nuscenes == False : 
                    bound_indices = np.where(np.all(np.asarray(pcd_camframe_world.points) > min_bound, axis=1) & np.all(np.asarray(pcd_camframe_world.points) < max_bound, axis=1))[0] ##speedup currently only works for KITTI
                else : 
                    bound_indices = np.where(np.all(np.asarray(pcd_camframe.points) > -hpr_bounds, axis=1) & np.all(np.asarray(pcd_camframe.points) < hpr_bounds, axis=1))[0]
                pcd_camframe_hpr = get_subpcd(pcd_camframe, bound_indices)
                #o3d.visualization.draw_geometries([pcd_camframe_hpr])
        
                start = time.time()
                visible_indices = hidden_point_removal_o3d(np.asarray(pcd_camframe_hpr.points), camera=[0,0,0], radius_factor=hpr_radius)
                
                end_hpr = time.time() - start 
                #print("HPR takes ", end_hpr ," s")
                visible_indices = bound_indices[visible_indices]
            else:
                visible_indices = np.where(hpr_masks[i])[0]
            
            
            frame_indices = list(set(visible_indices) & set(chunk_and_inlier_indices))
            

            # Load the SAM label
            if sam:
                sam_masks = dataset.get_sam_mask(cams[cam_id], points_index)
                sam_labels = masks_to_image(sam_masks)
            

            # Load the DINOV2 feature map
            start = time.time()
            if dino:
                if num_dino_features < 384:
                    dinov2_original = dataset.get_dinov2_features(cams[cam_id], points_index)
                    dino_reshape = np.reshape(dinov2_original, (dinov2_original.shape[0] * dinov2_original.shape[1], dinov2_original.shape[2]))
                    fit = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=num_dino_features, metric='euclidean')
                    u = fit.fit_transform(dino_reshape)
                    dinov2_feature_map = np.reshape(u, (dinov2_original.shape[0], dinov2_original.shape[1], u.shape[1]))

                    
                elif num_dino_features == 384:
                    dinov2_feature_map = dataset.get_dinov2_features(cams[cam_id], points_index)
                else:
                    raise ValueError("num_dino_features must be <= 384")

                dino_factor_0 = dinov2_feature_map.shape[0] / label_shape[0]
                dino_factor_1 = dinov2_feature_map.shape[1] / label_shape[1]
            
            # Apply visibility to downsampled chunk used for normalized cuts
            visible_chunk = get_subpcd(pcd_camframe, frame_indices)
            #o3d.visualization.draw_geometries([chunk_nc])
            chunk_nc_camframe = copy.deepcopy(chunk_nc).transform(T_pcd2cam)

            visible_chunk_tree = o3d.geometry.KDTreeFlann(visible_chunk)
            nc_indices = []
            for j, point in enumerate(np.asarray(chunk_nc_camframe.points)):
                [_, idx, _] = visible_chunk_tree.search_knn_vector_3d(point, 1)
    
                if np.linalg.norm(point - np.asarray(visible_chunk.points)[idx[0]]) < voxel_size / 2:
                    nc_indices.append(j)

            visible_nc_camframe = get_subpcd(chunk_nc_camframe, nc_indices)
            #o3d.visualization.draw_geometries([visible_nc_camframe])

            # Project the points to pixels
            points_to_pixels = point_to_pixel(np.asarray(visible_nc_camframe.points), K, label_shape[0], label_shape[1])
            start = time.time()
            if rm_perp:
                T_cam2pcd = np.linalg.inv(copy.deepcopy(T_pcd2cam))
                visible_nc_pcdframe = copy.deepcopy(visible_nc_camframe).transform(T_cam2pcd)
                normals = np.zeros((len(visible_nc_pcdframe.points), 3))
                normals = kDTree_1NN_feature_reprojection(normals, visible_nc_pcdframe, np.asarray(pcd_chunk_final.normals), pcd_chunk_final, max_radius=voxel_size/2, no_feature_label=[0,0,0])
                visible_nc_pcdframe_points = np.asarray(visible_nc_pcdframe.points)
            
            for point_id, pixel_id in points_to_pixels.items():
                pixel = pixel_id["pixels"]

                if sam:
                    label = sam_labels[pixel[1], pixel[0]]
                else:
                    label = False

                if rm_perp:
                    valid = not is_perpendicular_and_upward(visible_nc_pcdframe_points[point_id], normals[point_id], boundary=rm_perp)
                else:
                    valid = True

                if label and valid:
                    point2sam_nc[nc_indices[point_id], i] = label
                if dino and valid:
                    dino_pixel_0 = int(dino_factor_0 * pixel[1])
                    dino_pixel_1 = int(dino_factor_1 * pixel[0])
                    point2dino_nc[nc_indices[point_id], i, :] = dinov2_feature_map[dino_pixel_0, dino_pixel_1, :]
            end = time.time() - start 
            end = time.time() - start_loop
            
            '''
            print("total loop time ", end ," s")
            print("Percentage HPR ", round(end_hpr/end,4)*100)
            print("Percentage vis ", round(end_vis/end,4)*100)
            print("Percentage load", round(end_load/end,4)* 100)
            print('-------------')
            '''

        if sam:
            point2sam_list.append(point2sam_nc)
        if dino:
            point2dino_list.append(point2dino_nc)
                
    if sam and dino:
        return point2sam_list, point2dino_list
    elif sam:
        return point2sam_list
    elif dino:
        return point2dino_list
    else:
        raise ValueError("Either sam or dino must be True")


def dinov2_mean(point2dino):
    # Compute mean of DINOV2 features over number of views
    point2dino_mean = np.zeros((point2dino.shape[0], point2dino.shape[2]))
    non_zero_mask = point2dino.any(axis=2)
    for i in range(point2dino.shape[0]):
        features = point2dino[i][non_zero_mask[i]]
        if features.shape[0] != 0:
            point2dino_mean[i] = np.mean(features, axis=0)
    return point2dino_mean