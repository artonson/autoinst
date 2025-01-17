import open3d as o3d
import numpy as np
import os
from scipy.spatial.distance import cdist
from utils.point_cloud.point_cloud_utils import (
    remove_isolated_points,
    get_subpcd,
    get_statistical_inlier_indices,
    kDTree_1NN_feature_reprojection,
)
from utils.visualization_utils import generate_random_colors 
from utils.image.image_utils import (
    sam_label_distance,
    dinov2_mean,
    image_based_features_per_patch,
)

from utils.point_cloud.chunk_generation import (
    tarl_features_per_patch,
    get_indices_feature_reprojection,
)
from ncuts.normalized_cut import normalized_cut
import scipy
import copy
from config import *


def ncuts_chunk(
    dataset,
    chunk_downsample_dict,
    pcd_nonground_minor,
    T_pcd,
    sampled_indices_global,
    sequence=None,
    patchwise_indices=None,
):

    print("Start of sequence", sequence)
    first_id = patchwise_indices[sequence][0]
    center_id = chunk_downsample_dict['center_ids'][sequence]
    center_position = chunk_downsample_dict['center_positions'][sequence]
    chunk_indices = chunk_downsample_dict['indices'][sequence]

    cam_indices_global, _ = get_indices_feature_reprojection(
        sampled_indices_global, first_id, adjacent_frames=ADJACENT_FRAMES_CAM
    )
    tarl_indices_global, _ = get_indices_feature_reprojection(
        sampled_indices_global, center_id, adjacent_frames=ADJACENT_FRAMES_TARL
    )

    pcd_chunk = chunk_downsample_dict['pcd_nonground_chunks'][sequence]
    pcd_ground_chunk = chunk_downsample_dict['pcd_ground_chunks'][sequence]

    chunk_major = chunk_downsample_dict['pcd_nonground_chunks_major_downsampling'][sequence]

    points_major = np.asarray(chunk_major.points)
    num_points_major = points_major.shape[0]

    print(num_points_major, "points in downsampled chunk (major)")
    spatial_distance = cdist(points_major, points_major)
    mask = np.where(spatial_distance <= PROXIMITY_THRESHOLD, 1, 0)

    if CONFIG['alpha']:
        spatial_edge_weights = mask * np.exp(-CONFIG['alpha'] * spatial_distance)
    else:
        spatial_edge_weights = mask


    if CONFIG['beta'] and not CONFIG['gamma']:
        sam_features_major_list = image_based_features_per_patch(
            dataset,
            pcd_nonground_minor,
            chunk_indices,
            chunk_major,
            T_pcd,
            cam_indices_global,
            sam=True,
            dino=False,
        )

    elif CONFIG['gamma'] and not CONFIG['beta']:
        point2dino_list, vis_mask = image_based_features_per_patch(
            dataset,
            pcd_nonground_minor,
            chunk_indices,
            chunk_major,
            T_pcd,
            cam_indices_global,
            sam=False,
            dino=True,
            pcd_chunk=pcd_chunk,
        )
        dinov2_features_major_list = []
        for point2dino in point2dino_list:
            dinov2_features_major_list.append(dinov2_mean(point2dino))

    elif CONFIG['beta'] and CONFIG['gamma']:
        sam_features_major_list, point2dino_list = image_based_features_per_patch(
            dataset,
            pcd_nonground_minor,
            chunk_indices,
            chunk_major,
            T_pcd,
            cam_indices_global,
            sam=True,
            dino=True,
        )
        dinov2_features_major_list = []
        for point2dino in point2dino_list:
            dinov2_features_major_list.append(dinov2_mean(point2dino))

    sam_edge_weights = copy.deepcopy(mask)
    dinov2_edge_weights = copy.deepcopy(mask)

    if CONFIG['beta']:
        if len(sam_features_major_list) == 0:
            raise ValueError("The length should be longer than 0!")

        for sam_features_major in sam_features_major_list:
            sam_edge_weights_cam, _ = sam_label_distance(
                sam_features_major, spatial_distance, PROXIMITY_THRESHOLD, CONFIG['beta']
            )
            sam_edge_weights = sam_edge_weights * sam_edge_weights_cam

    if CONFIG['gamma']:
        if len(dinov2_features_major_list) == 0:
            raise ValueError("The length should be longer than 0!")

        for dinov2_features_major in dinov2_features_major_list:
            dinov2_distance = cdist(dinov2_features_major, dinov2_features_major)


            dinov2_edge_weights = dinov2_edge_weights * np.exp(-CONFIG['gamma'] * dinov2_distance)

    if CONFIG['theta']:
        tarl_features = tarl_features_per_patch(
            dataset,
            chunk_major,
            T_pcd,
            center_position,
            tarl_indices_global,
        )
        no_tarl_mask = ~np.array(tarl_features).any(1)
        tarl_distance = cdist(tarl_features, tarl_features)
        tarl_distance[no_tarl_mask] = 0
        tarl_distance[:, no_tarl_mask] = 0
        tarl_edge_weights = mask * np.exp(-CONFIG['theta'] * tarl_distance)
    else:
        tarl_edge_weights = mask

    A = (
        tarl_edge_weights
        * spatial_edge_weights
        * sam_edge_weights
        * dinov2_edge_weights
    )
    print("Adjacency Matrix built")
    # Remove isolated points
    chunk_major, A = remove_isolated_points(chunk_major, A)
    print(
        num_points_major - np.asarray(chunk_major.points).shape[0],
        "isolated points removed",
    )
    num_points_major = np.asarray(chunk_major.points).shape[0]

    print("Start of normalized Cuts")
    A = scipy.sparse.csr_matrix(A)
    grouped_labels = normalized_cut(
        A,
        num_points_major,
        np.arange(num_points_major),
        T=CONFIG['T'],
        split_lim=SPLIT_LIM,
    )


    random_colors = generate_random_colors(600)

    pcd_color = np.zeros((num_points_major, 3))

    for i, s in enumerate(grouped_labels):
        for j in s:
            pcd_color[j] = np.array(random_colors[i]) / 255

    pcd_chunk.paint_uniform_color([0, 0, 0])
    colors = kDTree_1NN_feature_reprojection(
        np.asarray(pcd_chunk.colors), pcd_chunk, pcd_color, chunk_major
    )
    pcd_chunk.colors = o3d.utility.Vector3dVector(colors)

    inliers = get_statistical_inlier_indices(pcd_ground_chunk)
    ground_inliers = get_subpcd(pcd_ground_chunk, inliers)
    mean_hight = np.mean(np.asarray(ground_inliers.points)[:, 2])
    in_idcs = np.where(
        np.asarray(ground_inliers.points)[:, 2] < (mean_hight + MEAN_HEIGHT)
    )[0]
    cut_hight = get_subpcd(ground_inliers, in_idcs)
    cut_hight.paint_uniform_color([0, 0, 0])
    merged_chunk = pcd_chunk + cut_hight
    
    inst_ground = chunk_downsample_dict['kitti_labels']["ground"]["instance"][sequence][inliers][in_idcs]
    seg_ground = chunk_downsample_dict['kitti_labels']["ground"]["semantic"][sequence][inliers][in_idcs]

    return merged_chunk,  pcd_chunk, cut_hight,inst_ground,seg_ground


def get_merge_pcds(out_folder_ncuts):
    point_clouds = []

    # List all files in the folder
    files = os.listdir(out_folder_ncuts)
    files.sort()

    # Filter files with a .pcd extension
    pcd_files = [file for file in files if file.endswith(".pcd")]
    print(pcd_files)
    # Load each point cloud and append to the list
    for pcd_file in pcd_files:

        file_path = os.path.join(out_folder_ncuts, pcd_file)
        point_cloud = o3d.io.read_point_cloud(file_path)
        point_clouds.append(point_cloud)
    return point_clouds


def kDTree_1NN_feature_reprojection_colors(
    features_to,
    pcd_to,
    features_from,
    pcd_from,
    labels=None,
    max_radius=None,
    no_feature_label=[1, 0, 0],
):
    """
    Args:
        pcd_from: point cloud to be projected
        pcd_to: point cloud to be projected to
        search_method: search method ("radius", "knn")
        search_param: search parameter (radius or k)
    Returns:
        features_to: features projected on pcd_to
    """
    from_tree = o3d.geometry.KDTreeFlann(pcd_from)
    labels_output = (
        np.ones(
            np.asarray(pcd_to.points).shape[0],
        )
        * -1
    )
    unique_colors = list(np.unique(np.asarray(pcd_from.colors), axis=0))

    for i, point in enumerate(np.asarray(pcd_to.points)):

        [_, idx, _] = from_tree.search_knn_vector_3d(point, 1)
        if max_radius is not None:
            if np.linalg.norm(point - np.asarray(pcd_from.points)[idx[0]]) > max_radius:
                features_to[i, :] = no_feature_label
                if labels is not None:
                    labels[i] = -1
                    labels_output[i] = -1
            else:
                features_to[i, :] = features_from[idx[0]]
                labels_output[i] = np.where(
                    (unique_colors == features_from[idx[0]]).all(axis=1)
                )[0]
        else:
            features_to[i, :] = features_from[idx[0]]
            labels_output[i] = np.where(
                (unique_colors == features_from[idx[0]]).all(axis=1)
            )[0]

    if labels is not None:
        return features_to, labels, labels_output
    else:
        return features_to, None, labels_output
