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



def ncuts_chunk(
    dataset,
    indices,
    pcd_nonground_chunks,
    pcd_ground_chunks,
    pcd_nonground_chunks_major_downsampling,
    pcd_nonground_minor,
    T_pcd,
    center_positions,
    center_ids,
    positions,
    first_position,
    sampled_indices_global,
    chunk_size,
    major_voxel_size=0.35,
    alpha=1,
    beta=0,
    gamma=0,
    theta=0,
    proximity_threshold=1,
    ncuts_threshold=0.03,
    cams=["cam2", "cam3"],
    cam_ids=[0],
    out_folder=None,
    ground_mode=True,
    sequence=None,
    patchwise_indices=None,
    adjacent_frames_cam=(16, 13),
    adjacent_frames_tarl=(10, 10),
    use_z=False,
    norm=False,
    split_lim=0.01,
    obb=None,
    mean_height=0.2,
):

    print("Start of sequence", sequence)
    first_id = patchwise_indices[sequence][0]
    center_id = center_ids[sequence]
    center_position = center_positions[sequence]
    chunk_indices = indices[sequence]

    cam_indices_global, _ = get_indices_feature_reprojection(
        sampled_indices_global, first_id, adjacent_frames=adjacent_frames_cam
    )
    tarl_indices_global, _ = get_indices_feature_reprojection(
        sampled_indices_global, center_id, adjacent_frames=adjacent_frames_tarl
    )

    if ground_mode == False:
        pcd_chunk = pcd_nonground_chunks[sequence]
        pcd_ground_chunk = pcd_ground_chunks[sequence]
    else:
        pcd_chunk = pcd_nonground_chunks
        chunk_major = pcd_nonground_chunks_major_downsampling[sequence]
        inliers = get_statistical_inlier_indices(chunk_major)
        ground_inliers = get_subpcd(chunk_major, inliers)
        mean_hight = np.mean(np.asarray(ground_inliers.points)[:, 2])
        in_idcs = np.where(
            np.asarray(ground_inliers.points)[:, 2] < (mean_hight + mean_height)
        )[0]
        chunk_major = get_subpcd(ground_inliers, in_idcs)

    chunk_major = pcd_nonground_chunks_major_downsampling[sequence]

    points_major = np.asarray(chunk_major.points)
    num_points_major = points_major.shape[0]

    print(num_points_major, "points in downsampled chunk (major)")
    spatial_distance = cdist(points_major, points_major)
    if use_z == True:
        spatial_distance = cdist(
            points_major[:, 2].reshape(-1, 1), points_major[:, 2].reshape(-1, 1)
        )
    mask = np.where(spatial_distance <= proximity_threshold, 1, 0)

    if alpha:
        spatial_edge_weights = mask * np.exp(-alpha * spatial_distance)
    else:
        spatial_edge_weights = mask


    if beta and not gamma:
        sam_features_major_list = image_based_features_per_patch(
            dataset,
            pcd_nonground_minor,
            chunk_indices,
            chunk_major,
            major_voxel_size,
            T_pcd,
            cam_indices_global,
            cams,
            cam_ids=cam_ids,
            hpr_radius=1000,
            sam=True,
            dino=False,
            rm_perp=0.0,
        )

    elif gamma and not beta:
        point2dino_list, vis_mask = image_based_features_per_patch(
            dataset,
            pcd_nonground_minor,
            chunk_indices,
            chunk_major,
            major_voxel_size,
            T_pcd,
            cam_indices_global,
            cams,
            cam_ids=cam_ids,
            hpr_radius=1000,
            num_dino_features=384,
            sam=False,
            dino=True,
            rm_perp=0.0,
            pcd_chunk=pcd_chunk,
            obb=obb,
        )
        dinov2_features_major_list = []
        for point2dino in point2dino_list:
            dinov2_features_major_list.append(dinov2_mean(point2dino))

    elif beta and gamma:
        sam_features_major_list, point2dino_list = image_based_features_per_patch(
            dataset,
            pcd_nonground_minor,
            chunk_indices,
            chunk_major,
            major_voxel_size,
            T_pcd,
            cam_indices_global,
            cams,
            cam_ids=cam_ids,
            hpr_radius=1000,
            num_dino_features=384,
            sam=True,
            dino=True,
            rm_perp=0.0,
            obb=obb,
        )
        dinov2_features_major_list = []
        for point2dino in point2dino_list:
            dinov2_features_major_list.append(dinov2_mean(point2dino))

    sam_edge_weights = copy.deepcopy(mask)
    dinov2_edge_weights = copy.deepcopy(mask)

    if beta:
        if len(sam_features_major_list) == 0:
            raise ValueError("The length should be longer than 0!")

        for sam_features_major in sam_features_major_list:
            sam_edge_weights_cam, _ = sam_label_distance(
                sam_features_major, spatial_distance, proximity_threshold, beta
            )
            sam_edge_weights = sam_edge_weights * sam_edge_weights_cam

    if gamma:
        if len(dinov2_features_major_list) == 0:
            raise ValueError("The length should be longer than 0!")

        for dinov2_features_major in dinov2_features_major_list:
            dinov2_distance = cdist(dinov2_features_major, dinov2_features_major)


            dinov2_edge_weights = dinov2_edge_weights * np.exp(-gamma * dinov2_distance)

    if theta:
        tarl_features = tarl_features_per_patch(
            dataset,
            chunk_major,
            T_pcd,
            center_position,
            tarl_indices_global,
            chunk_size,
            search_radius=major_voxel_size / 2,
            norm=norm,
            obb=obb,
        )
        no_tarl_mask = ~np.array(tarl_features).any(1)
        tarl_distance = cdist(tarl_features, tarl_features)
        tarl_distance[no_tarl_mask] = 0
        tarl_distance[:, no_tarl_mask] = 0
        tarl_edge_weights = mask * np.exp(-theta * tarl_distance)
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
        T=ncuts_threshold,
        split_lim=split_lim,
    )

    sorted_groups = sorted(grouped_labels, key=lambda x: len(x))
    num_points_top3 = np.sum([len(g) for g in sorted_groups[-3:]])
    top3_ratio = num_points_top3 / num_points_major

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

    if ground_mode == False:
        inliers = get_statistical_inlier_indices(pcd_ground_chunk)
        ground_inliers = get_subpcd(pcd_ground_chunk, inliers)
        mean_hight = np.mean(np.asarray(ground_inliers.points)[:, 2])
        in_idcs = np.where(
            np.asarray(ground_inliers.points)[:, 2] < (mean_hight + mean_height)
        )[0]
        cut_hight = get_subpcd(ground_inliers, in_idcs)
        cut_hight.paint_uniform_color([0, 0, 0])
        merged_chunk = pcd_chunk + cut_hight
    else:
        merged_chunk = pcd_chunk

    index_file = str(center_id).zfill(6) + ".pcd"
    file = os.path.join(out_folder, index_file)
    if ground_mode == False:
        return merged_chunk, file, pcd_chunk, cut_hight, inliers, in_idcs
    else:
        return merged_chunk, file


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
