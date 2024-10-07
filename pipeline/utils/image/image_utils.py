import cv2
from utils.visualization_utils import generate_random_colors
import numpy as np
import copy
import open3d as o3d 
from utils.image.point_to_pixels import (
    point_to_pixel
)

from utils.point_cloud.point_cloud_utils import (
    kDTree_1NN_feature_reprojection,
    angle_between,
    get_subpcd,
    get_statistical_inlier_indices
)

from utils.image.hidden_points_removal import (
    hidden_point_removal_o3d
)
from config import *

def is_perpendicular_and_upward(point, normal, boundary=0.1):
    """
    Args:
        point: 3D point
        normal: normal vector of 3D point
        boundary: boundary around pi/2 to be considered perpendicular
    Returns:
        True if point is perpendicular to normal and pointing upwards, False otherwise
    """
    angle = np.abs(angle_between(point, normal))
    perpendicular = (
        angle > (np.pi / 2 - boundary) and angle < (np.pi / 2 + boundary)
    ) or (angle > (3 * np.pi / 2 - boundary) and angle < (3 * np.pi / 2 + boundary))
    upward = (normal[2] * normal[2]) > (normal[0] * normal[0] + normal[1] * normal[1])

    return perpendicular and upward

def downsample(image, scale_factor):
    return cv2.resize(image, (int(scale_factor * image.shape[1]), int(scale_factor * image.shape[0])), interpolation = cv2.INTER_NEAREST)

def masks_to_image(masks):
    '''
    Function that takes an array of masks and returns an pixel-wise label map
    '''
    image_labels = np.zeros(masks[0]['segmentation'].shape)
    for i, mask in enumerate(masks):
        image_labels[mask['segmentation']] = i + 1

    return image_labels

def masks_to_colored_image(masks):
    '''
    Function that takes an array of masks and returns an pixel-wise colored label map
    '''
    colors = generate_random_colors(200)
    height, width = masks[0]["segmentation"].shape
    image_labels = np.zeros((height, width, 3))
    for i, mask in enumerate(masks):
        image_labels[mask['segmentation']] = colors[i]

    return image_labels

def sam_label_distance(sam_features, spatial_distance, proximity_threshold, beta):

    mask = np.where(spatial_distance <= proximity_threshold)
    
    # Initialize the distance matrix with zeros
    num_points, num_views = sam_features.shape
    distance_matrix = np.zeros((num_points, num_points))

    # Iterate over rows (points)
    for (point1, point2) in (zip(*mask)):
        view_counter = 0
        for view in range(num_views):
            instance_id1 = sam_features[point1, view]
            instance_id2 = sam_features[point2, view]

            if instance_id1 != -1 and instance_id2 != -1:
                view_counter += 1
                if instance_id1 != instance_id2:
                    distance_matrix[point1, point2] += 1
        if view_counter:
            distance_matrix[point1, point2] /= view_counter
            
    mask = np.where(spatial_distance <= proximity_threshold, 1, 0)
    label_distance = mask * np.exp(-beta * distance_matrix)

    return label_distance, mask

def image_based_features_per_patch(
    dataset,
    pcd,
    chunk_indices,
    chunk_nc,
    T_pcd2world,
    cam_indices,
    hpr_masks=None,
    sam=True,
    dino=True,
    rm_perp=0.0,
    pcd_chunk=None,
    vis=False,
):
    cams = ['cam2','cam3']
    orig_chunk_nc = copy.deepcopy(chunk_nc)
    num_points_nc = np.asarray(chunk_nc.points).shape[0]

    pcd_chunk = get_subpcd(pcd, chunk_indices)
    inlier_indices_of_chunk = get_statistical_inlier_indices(pcd_chunk)
    chunk_and_inlier_indices = chunk_indices[inlier_indices_of_chunk]

    visibility_mask = np.zeros(
        (np.asarray(chunk_nc.points).shape[0])
    )  ### for ablation eval

    if rm_perp:
        pcd_chunk_final = get_subpcd(pcd_chunk, inlier_indices_of_chunk)
        pcd_chunk_final.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=200)
        )

    if sam:
        point2sam_list = []
    if dino:
        point2dino_list = []

    for cam_id in CAM_IDS:

        if sam:
            point2sam_nc = (-1) * np.ones(
                (num_points_nc, len(cam_indices)), dtype=int
            ) 

        if dino:
            point2dino_nc = np.zeros(
                (num_points_nc, len(cam_indices), NUM_DINO_FEATURES)
            )

        image = dataset.get_image(cams[cam_id], 0)
        w, h = image.size
        label_shape = (h, w)
        print("label shape", label_shape)

        if hpr_masks is not None:
            assert len(cam_indices) == hpr_masks.shape[0]

        for i, points_index in enumerate(cam_indices):
            T_lidar2world = dataset.get_pose(points_index)
            T_world2lidar = np.linalg.inv(T_lidar2world)
            T_lidar2cam, K = dataset.get_calibration_matrices(cams[cam_id])
            T_world2cam = T_lidar2cam @ T_world2lidar
            T_pcd2cam = T_world2cam @ T_pcd2world

            # hidden point removal
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())
            pcd_camframe_world = copy.deepcopy(new_pcd).transform(dataset.get_pose(0))
            pcd_camframe = copy.deepcopy(pcd).transform(T_pcd2cam)

            pts = np.asarray(pcd_chunk.points)
            min_x, min_y, min_z = pts[:, 0].min(), pts[:, 1].min(), pts[:, 2].min()
            max_x, max_y, max_z = pts[:, 0].max(), pts[:, 1].max(), pts[:, 2].max()
            min_bound = np.array([min_x, min_y, min_z])
            max_bound = np.array([max_x, max_y, max_z])

            if hpr_masks is None:

                hpr_bounds = np.array([25, 25, 25])
                bound_indices = np.where(
                        np.all(
                            np.asarray(pcd_camframe_world.points) > min_bound, axis=1
                        )
                        & np.all(
                            np.asarray(pcd_camframe_world.points) < max_bound, axis=1
                        )
                    )[
                        0
                    ]  ##speedup currently only works for KITTI
               
                pcd_camframe_hpr = get_subpcd(pcd_camframe, bound_indices)

                try:
                    visible_indices = hidden_point_removal_o3d(
                        np.asarray(pcd_camframe_hpr.points),
                        camera=[0, 0, 0],
                        radius_factor=HPR_RADIUS,
                    )
                except:
                    print("hpr skip")
                    continue

                if vis:
                    visible_indices_visibility = hidden_point_removal_o3d(
                        np.asarray(pcd_camframe_hpr.points),
                        camera=[0, 0, 0],
                        radius_factor=20,
                    )

                    visible_indices_visibility = bound_indices[
                        visible_indices_visibility
                    ]

                visible_indices = bound_indices[visible_indices]

            else:
                visible_indices = np.where(hpr_masks[i])[0]

            frame_indices = list(set(visible_indices) & set(chunk_and_inlier_indices))
            if vis == True:
                frame_indices_vis = list(
                    set(visible_indices_visibility) & set(chunk_and_inlier_indices)
                )
            if len(frame_indices) == 0:
                print("out of view skip")
                continue

            # Load the SAM label
            if sam:
                sam_masks = dataset.get_sam_mask(cams[cam_id], points_index)
                sam_labels = masks_to_image(sam_masks)

            # Load the DINOV2 feature map
            if dino:
                if NUM_DINO_FEATURES < 384:
                    dinov2_original = dataset.get_dinov2_features(
                        cams[cam_id], points_index
                    )
                    dino_reshape = np.reshape(
                        dinov2_original,
                        (
                            dinov2_original.shape[0] * dinov2_original.shape[1],
                            dinov2_original.shape[2],
                        ),
                    )
                    fit = umap.UMAP(
                        n_neighbors=50,
                        min_dist=0.0,
                        n_components=NUM_DINO_FEATURES,
                        metric="euclidean",
                    )
                    u = fit.fit_transform(dino_reshape)
                    dinov2_feature_map = np.reshape(
                        u,
                        (
                            dinov2_original.shape[0],
                            dinov2_original.shape[1],
                            u.shape[1],
                        ),
                    )

                elif NUM_DINO_FEATURES == 384:
                    dinov2_feature_map = dataset.get_dinov2_features(
                        cams[cam_id], points_index
                    )
                else:
                    raise ValueError("num_dino_features must be <= 384")

                dino_factor_0 = dinov2_feature_map.shape[0] / label_shape[0]
                dino_factor_1 = dinov2_feature_map.shape[1] / label_shape[1]

            # Apply visibility to downsampled chunk used for normalized cuts
            visible_chunk = get_subpcd(pcd_camframe, frame_indices)
            if vis == True:
                visible_chunk_ablation = get_subpcd(pcd_camframe, frame_indices_vis)
            
            chunk_nc_camframe = copy.deepcopy(chunk_nc).transform(T_pcd2cam)

            visible_chunk_tree = o3d.geometry.KDTreeFlann(visible_chunk)
            nc_indices = []
            for j, point in enumerate(np.asarray(chunk_nc_camframe.points)):
                [_, idx, _] = visible_chunk_tree.search_knn_vector_3d(point, 1)

                if (
                    np.linalg.norm(point - np.asarray(visible_chunk.points)[idx[0]])
                    < MAJOR_VOXEL_SIZE / 2
                ):
                    nc_indices.append(j)

            if vis:
                visible_chunk_tree_ablation = o3d.geometry.KDTreeFlann(
                    visible_chunk_ablation
                )
                nc_indices_visbility = []
                for j, point in enumerate(np.asarray(chunk_nc_camframe.points)):
                    [_, idx, _] = visible_chunk_tree_ablation.search_knn_vector_3d(
                        point, 1
                    )

                    if (
                        np.linalg.norm(
                            point - np.asarray(visible_chunk_ablation.points)[idx[0]]
                        )
                        < MAJOR_VOXEL_SIZE / 2
                    ):
                        nc_indices_visbility.append(j)
                visibility_mask[nc_indices_visbility] = 1

            visible_nc_camframe = get_subpcd(chunk_nc_camframe, nc_indices)
            points_to_pixels = point_to_pixel(
                np.asarray(visible_nc_camframe.points),
                K,
                label_shape[0],
                label_shape[1],
            )
            if rm_perp:
                T_cam2pcd = np.linalg.inv(copy.deepcopy(T_pcd2cam))
                visible_nc_pcdframe = copy.deepcopy(visible_nc_camframe).transform(
                    T_cam2pcd
                )
                normals = np.zeros((len(visible_nc_pcdframe.points), 3))
                normals = kDTree_1NN_feature_reprojection(
                    normals,
                    visible_nc_pcdframe,
                    np.asarray(pcd_chunk_final.normals),
                    pcd_chunk_final,
                    max_radius=MAJOR_VOXEL_SIZE / 2,
                    no_feature_label=[0, 0, 0],
                )
                visible_nc_pcdframe_points = np.asarray(visible_nc_pcdframe.points)

            for point_id, pixel_id in points_to_pixels.items():
                pixel = pixel_id["pixels"]

                if sam:
                    label = sam_labels[pixel[1], pixel[0]]
                else:
                    label = False

                if rm_perp:
                    valid = not is_perpendicular_and_upward(
                        visible_nc_pcdframe_points[point_id],
                        normals[point_id],
                        boundary=rm_perp,
                    )
                else:
                    valid = True

                if label and valid:
                    point2sam_nc[nc_indices[point_id], i] = label
                if dino and valid:
                    dino_pixel_0 = int(dino_factor_0 * pixel[1])
                    dino_pixel_1 = int(dino_factor_1 * pixel[0])
                    point2dino_nc[nc_indices[point_id], i, :] = dinov2_feature_map[
                        dino_pixel_0, dino_pixel_1, :
                    ]

        if sam:
            point2sam_list.append(point2sam_nc)
        if dino:
            point2dino_list.append(point2dino_nc)

    if sam and dino:
        return point2sam_list, point2dino_list
    elif sam:
        return point2sam_list
    elif dino:
        return point2dino_list, visibility_mask
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
