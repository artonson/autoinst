import numpy as np
from scipy.spatial.distance import cdist
import open3d as o3d
from utils.point_cloud.point_cloud_utils import (
    transform_pcd,
    get_pcd,
    get_statistical_inlier_indices,
    get_subpcd,
)

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from config import *


def subsample_positions(positions, voxel_size=1, batch_size=1000):
    positions = np.array(positions)
    min_vals = positions.min(axis=0)
    max_vals = positions.max(axis=0)

    centers = [
        np.arange(min_val, max_val + voxel_size, voxel_size)
        for min_val, max_val in zip(min_vals, max_vals)
    ]
    grid = np.stack(np.meshgrid(*centers, indexing="ij"), -1).reshape(-1, 3)

    subsampled_indices = []
    total_batches = len(grid) // batch_size + (1 if len(grid) % batch_size != 0 else 0)
    with tqdm(total=total_batches) as pbar:
        for i in range(0, len(grid), batch_size):
            batch_grid = grid[i : i + batch_size]
            distances = cdist(batch_grid, positions)
            closest_pose_indices = np.argmin(distances, axis=1)
            unique_indices = np.unique(closest_pose_indices)
            for index in unique_indices:
                closest_pose = positions[index]
                distance = np.abs(
                    batch_grid[closest_pose_indices == index] - closest_pose
                )
                if np.all(distance < 0.5 * voxel_size, axis=1).any():
                    subsampled_indices.append(index)
            pbar.update(1)

    return np.sort(subsampled_indices)


def is_point_inside_obb(point, obb):
    """
    Check if a point is inside the OBB by transforming the point into the OBB's local coordinate system.
    """
    # Translate the point to the OBB's local origin
    local_point = point - obb.center

    # Project the translated point onto the OBB's local axes
    for i in range(3):
        axis = np.array(obb.R[:, i])
        extent = obb.extent[i] / 2.0
        projection = np.dot(local_point, axis)
        if np.abs(projection) > extent:
            return False
    return True


def are_points_inside_obb(points, obb):
    """
    Vectorized check if multiple points are inside the OBB by transforming the points into the OBB's local coordinate system.

    Args:
    - points: Nx3 numpy array of points.
    - obb: An OBB object with attributes 'center', 'R', and 'extent'.

    Returns:
    - A boolean array indicating whether each point is inside the OBB.
    """
    # Translate the points to the OBB's local origin
    local_points = points - obb.center

    # Initialize a boolean array to keep track of points inside the OBB
    inside = np.ones(local_points.shape[0], dtype=bool)

    # Project the translated points onto the OBB's local axes and check extents
    for i in range(3):
        axis = np.array(obb.R[:, i])
        extent = obb.extent[i] / 2.0

        # Calculate the projection of each point onto the current axis
        projection = np.dot(local_points, axis)

        # Update 'inside' to False for points outside the OBB's extent on this axis
        inside &= np.abs(projection) <= extent

    return inside


def chunks_from_pointcloud(
    pcd,
    T_pcd,
    positions,
    first_position,
    indices,
    labels=None,
    ground=False,
):

    points = np.asarray(pcd.points)

    pcd_chunks = []
    chunk_indices = []
    center_pos = []
    center_ids = []
    chunk_bounds = []
    obbs = []

    if labels != None:
        kitti_out = {"panoptic": [], "semantic": [], "instance": []}
    else:
        kitti_out = None
    distance = 0
    last_position = None
    cnt = 0
    for position, index in tqdm(zip(positions, indices), total=len(positions)):
        if last_position is not None:
            distance += np.linalg.norm(position - last_position)
            if distance > (min(CHUNK_SIZE[0], CHUNK_SIZE[1]) - OVERLAP):  # New chunk

                pos_pcd = position - first_position
                rot = np.linalg.inv(T_pcd[:3, :3])
                pos_pcd = rot @ pos_pcd

                max_position = pos_pcd + (0.5 * CHUNK_SIZE)
                min_position = pos_pcd - (0.5 * CHUNK_SIZE)

                ids = np.where(
                        np.all(points > min_position, axis=1)
                        & np.all(points < max_position, axis=1)
                    )[0]
                obbs.append(0)
                pcd_cut = pcd.select_by_index(ids)

                cnt += 1

                inlier_indices = get_statistical_inlier_indices(pcd_cut)
                pcd_cut_final = get_subpcd(pcd_cut, inlier_indices)

                if labels != None:
                    if ground == False:
                        kitti_out["semantic"].append(
                            labels["seg_nonground"][ids][inlier_indices]
                        )
                        kitti_out["instance"].append(
                            labels["instance_nonground"][ids][inlier_indices]
                        )
                    else:
                        kitti_out["semantic"].append(
                            labels["seg_ground"][ids][inlier_indices]
                        )
                        kitti_out["instance"].append(
                            labels["instance_ground"][ids][inlier_indices]
                        )


                pcd_chunks.append(pcd_cut_final)
                chunk_indices.append(ids)
                center_pos.append(pos_pcd)
                center_ids.append(index)
                chunk_bounds.append(((pos_pcd - 0.5 * CHUNK_SIZE), (pos_pcd + 0.5 * CHUNK_SIZE)))

                distance = 0
        last_position = position

    return (
        pcd_chunks,
        chunk_indices,
        center_pos,
        center_ids,
        chunk_bounds,
        kitti_out,
        obbs,
    )


def indices_per_patch(
    T_pcd, center_positions, positions, first_position, global_indices 
):

    patchwise_indices = []

    for center in center_positions:
        indices = []

        for position, index in zip(positions, global_indices):

            pos_pcd = position - first_position
            rot = np.linalg.inv(T_pcd[:3, :3])
            pos_pcd = rot @ pos_pcd

            if np.linalg.norm(center - pos_pcd) < 0.5 * CHUNK_SIZE[1]:
                indices.append(index)
        patchwise_indices.append(indices)

    return patchwise_indices


def tarl_features_per_patch(
    dataset,
    pcd,
    T_pcd,
    center_position,
    tarl_indices,
):
    search_radius = MAJOR_VOXEL_SIZE/2.
    concatenated_tarl_points = np.zeros((0, 3))
    concatenated_tarl_features = np.zeros((0, 96))

    num_points = np.asarray(pcd.points).shape[0]

    max_position = center_position + 0.5 * CHUNK_SIZE
    min_position = center_position - 0.5 * CHUNK_SIZE

    for points_index in tarl_indices:

        # Load the TARL features & points
        tarl_features = dataset.get_tarl_features(points_index)
        coords = dataset.get_point_cloud(points_index)

        # Transform the coordinates
        T_lidar2world = dataset.get_pose(points_index)
        T_local2global = np.linalg.inv(T_pcd) @ T_lidar2world
        coords = transform_pcd(coords, T_local2global)

        mask = np.where(
                np.all(coords > min_position, axis=1)
                & np.all(coords < max_position, axis=1)
            )[0]

        coords, tarl_features = coords[mask], tarl_features[mask]
        concatenated_tarl_points = np.concatenate((concatenated_tarl_points, coords))
        concatenated_tarl_features = np.concatenate(
            (concatenated_tarl_features, tarl_features)
        )

    tarl_pcd = get_pcd(concatenated_tarl_points)
    tarl_tree = o3d.geometry.KDTreeFlann(tarl_pcd)
    tarl_features = np.zeros((num_points, 96))

    for i, point in enumerate(np.asarray(pcd.points)):
        [_, idx, _] = tarl_tree.search_radius_vector_3d(point, search_radius)
        features_in_radius = concatenated_tarl_features[idx]

        if not features_in_radius.shape[0] == 0:
            tarl_features[i, :] = np.mean(features_in_radius, axis=0)
            if TARL_NORM:
                tarl_features[i] /= np.linalg.norm(tarl_features[i])
        else:
            continue

    return tarl_features


def get_indices_feature_reprojection(global_indices, first_id, adjacent_frames=(8, 5)):

    first_index = global_indices.index(first_id)
    cam_indices_global = global_indices[
        max(0, first_index - adjacent_frames[0]) : first_index + adjacent_frames[1]
    ]
    indices = []
    for global_i in cam_indices_global:
        indices.append(global_indices.index(global_i))

    return cam_indices_global, indices




