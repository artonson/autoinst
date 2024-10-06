# Transformation on point clouds
import numpy as np
import open3d as o3d
import copy
from open3d.pipelines import registration
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor


def get_pcd(points: np.array):
    """
    Convert numpy array to open3d point cloud
    Args:
        points: 3D points in camera coordinate [npoints, 3] or [npoints, 4]
    Returns:
        pcd: open3d point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd


def transform_pcd(points, T):
    """
    Apply the perspective projection
    Args:
        points:              3D points in coordinate system 1 [npoints, 3]
        T:                   Transformation matrix in homogeneous coordinates [4, 4]
    Returns:
        points:              3D points in coordinate system 2 [npoints, 3]
    """
    pcd = get_pcd(points)
    transformed_pcd = pcd.transform(T)
    return np.asarray(transformed_pcd.points)


def filter_points_from_dict(points, filter_dict):
    """
    Filter points based on a dict
    Args:
        points:      3D points in camera coordinate [npoints, 3]
        filter_dict: dict that maps point indices to pixel coordinates
    Returns:
        points:    3D points in camera coordinate within image FOV [npoints, 3]
    """

    inds = np.array(list(filter_dict.keys()))
    return points[inds]


def filter_points_from_list(points, filter_list):
    """
    Filter points based on a dict
    Args:
        points:      3D points in camera coordinate [npoints, 3]
        filter_dict: dict that maps point indices to pixel coordinates
    Returns:
        points:    3D points in camera coordinate within image FOV [npoints, 3]
    """

    inds = np.array(list(filter_list))
    return points[inds]

def write_pcd(folder,name,pcd,seq=None,cur_idx=None):
    out_fn = f"{folder}{name}{seq}_{cur_idx}.pcd"
    if seq is None : 
        out_fn = f"{folder}{name}"
    o3d.io.write_point_cloud(
                out_fn,
                pcd,
                write_ascii=False,
                compressed=False,
                print_progress=True,
    )


def point_to_label(
    point_to_pixel: dict, label_map, label_is_color=True, label_is_instance=False
):
    """
    Args:
        point_to_pixel: dict that maps point indices to pixel coordinates
        label_map: label map of image
    Returns:
        point_to_label: dict that maps point indices to labels
    """
    point_to_label_dict = {}

    for index, point_data in point_to_pixel.items():
        pixel = point_data["pixels"]

        if label_is_color:
            color = label_map[pixel[1], pixel[0]]
            if color.tolist() == [70, 70, 70]:  # ignore unlabeled pixels
                continue
            point_to_label_dict[index] = tuple((color / 255))

        elif label_is_instance:
            instance_label = int(label_map[pixel[1], pixel[0]])
            if instance_label:  # ignore unlabeled pixels
                point_to_label_dict[index] = instance_label
            else:
                continue

        else:  # label is continuous feature vector
            point_to_label_dict[index] = label_map[pixel[1], pixel[0]]

    return point_to_label_dict


def change_point_indices(point_to_X: dict, indices: list):
    """
    Args:
        point_to_X: dict that maps point indices to X
        indices: list of indices
    Returns:
        point_to_X: dict that maps point indices to X
    """
    point_to_X_new = {}
    for old_index in point_to_X.keys():
        new_index = indices[old_index]
        point_to_X_new[new_index] = point_to_X[old_index]

    return point_to_X_new


def transformation_matrix(rotation: np.array, translation: np.array):
    """
    Create a transformation matrix from rotation and translation
    Args:
        rotation:    Rotation matrix [3, 3]
        translation: Translation vector [3, 1]
    Returns:
        T:           Transformation matrix in homogeneous coordinates [4, 4]
    """

    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def kDTree_1NN_feature_reprojection(
    features_to,
    pcd_to,
    features_from,
    pcd_from,
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

    for i, point in enumerate(np.asarray(pcd_to.points)):

        [_, idx, _] = from_tree.search_knn_vector_3d(point, 1)
        if max_radius is not None:
            if np.linalg.norm(point - np.asarray(pcd_from.points)[idx[0]]) > max_radius:
                features_to[i, :] = no_feature_label
            else:
                features_to[i, :] = features_from[idx[0]]
        else:
            features_to[i, :] = features_from[idx[0]]

    return features_to


def divide_indices_into_chunks(max_index, chunk_size=1000):
    chunks = []
    for start in range(0, max_index, chunk_size):
        end = min(start + chunk_size, max_index)
        chunks.append((start, end))
    return chunks

def intersect(pred_indices, gt_indices):
    intersection = np.intersect1d(pred_indices, gt_indices)
    return intersection.size / pred_indices.shape[0]


def remove_isolated_points(pcd, adjacency_matrix):

    isolated_mask = ~np.all(adjacency_matrix == 0, axis=1)
    adjacency_matrix = adjacency_matrix[isolated_mask][:, isolated_mask]
    pcd = pcd.select_by_index(np.where(isolated_mask == True)[0])

    return pcd, adjacency_matrix


def get_statistical_inlier_indices(pcd, nb_neighbors=20, std_ratio=2.0):
    _, inlier_indices = copy.deepcopy(pcd).remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return inlier_indices


def get_subpcd(pcd, indices, colors=False, normals=False):
    subpcd = o3d.geometry.PointCloud()
    subpcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[indices])
    if colors:
        subpcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[indices])
    if normals:
        subpcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[indices])
    return subpcd


def downsample_chunk(points):
    num_points_to_sample = 60000
    every_k_points = int(points.shape[0] / num_points_to_sample)
    if every_k_points == 0:
        every_k_points = 1
    indeces = uniform_down_sample_with_indices(points, every_k_points)

    points = points[indeces]
    return points


def downsample_chunk_data(points, ncuts_labels, kitti_labels, semantics):
    num_points_to_sample = 60000
    every_k_points = int(points.shape[0] / num_points_to_sample)
    if every_k_points == 0:
        every_k_points = 1
    indeces = uniform_down_sample_with_indices(points, every_k_points)

    points = points[indeces]
    return points, ncuts_labels[indeces], kitti_labels[indeces], semantics[indeces]

def intersect(pred_indices, gt_indices):
    intersection = np.intersect1d(pred_indices, gt_indices)
    return intersection.size / pred_indices.shape[0]

def process_batch(unique_pred, preds, labels, gt_idcs, threshold, new_ncuts_labels):
    pred_idcs = np.where(preds == unique_pred)[0]
    cur_intersect = np.sum(np.isin(pred_idcs, gt_idcs))
    if cur_intersect > threshold * len(pred_idcs):
        new_ncuts_labels[pred_idcs] = 0


def remove_semantics(labels, preds, threshold=0.8, num_threads=4):
    gt_idcs = np.where(labels == 0)[0]
    new_ncuts_labels = preds.copy()
    unique_preds = np.unique(preds)

    if num_threads is None:
        num_threads = min(len(unique_preds), 4)  # Default to 8 threads if not specified

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in tqdm(unique_preds):
            futures.append(
                executor.submit(
                    process_batch,
                    i,
                    preds,
                    labels,
                    gt_idcs,
                    threshold,
                    new_ncuts_labels,
                )
            )

        # Wait for all tasks to complete
        for future in tqdm(futures, total=len(futures), desc="Processing"):
            future.result()  # Get the result to catch any exceptions

    return new_ncuts_labels


def uniform_down_sample_with_indices(points, every_k_points):
    # Create a new point cloud for the downsampled output

    # List to hold the indices of the points that are kept
    indices = []
    # Iterate over the points and keep every k-th point
    for i in range(0, points.shape[0], every_k_points):
        indices.append(i)

    return indices



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

def merge_unite_gt(chunks):
    last_chunk = chunks[0]
    merge = o3d.geometry.PointCloud()
    merge += last_chunk

    for new_chunk in chunks[1:]:
        merge += new_chunk

    merge.remove_duplicated_points()
    return merge

def merge_unite_gt_labels(chunks, semantic_maps):
    # Assuming pcd1 and pcd2 are your Open3D point cloud objects
    last_chunk = chunks[0]
    merge = o3d.geometry.PointCloud()
    merge += last_chunk
    output_semantics = semantic_maps[0]

    j = 1
    for new_chunk in chunks[1:]:
        pcd1_tree = o3d.geometry.KDTreeFlann(merge)
        points_pcd1 = np.asarray(merge.points)
        points_new = np.asarray(new_chunk.points)
        keep_idcs = []
        for i, point in enumerate(points_new):
            # Find the nearest neighbor in pcd1
            [k, idx, _] = pcd1_tree.search_knn_vector_3d(point, 1)
            if k > 0:
                # Check if the nearest neighbor is an exact match (distance = 0)
                if np.allclose(points_pcd1[idx[0]], point):
                    pass
                else:
                    keep_idcs.append(i)

        new_chunk.points = o3d.utility.Vector3dVector(
            np.asarray(new_chunk.points)[keep_idcs]
        )
        output_semantics = np.hstack(
            (
                output_semantics.reshape(
                    -1,
                ),
                semantic_maps[j][keep_idcs].reshape(
                    -1,
                ),
            )
        )
        j += 1

        merge += new_chunk

    return output_semantics


def merge_chunks_unite_instances2(chunks: list, icp=False):
    merge = o3d.geometry.PointCloud()
    chunk_means = [np.mean(np.asarray(chunk.points), axis=0) for chunk in chunks]

    last_chunk = chunks[0]
    merge = o3d.geometry.PointCloud()
    merge += last_chunk

    for new_chunk in tqdm(chunks[1:]):

        new_chunk_center = np.asarray(new_chunk.points)

        x, y, z = (
            new_chunk_center[:, 0].mean(),
            new_chunk_center[:, 1].mean(),
            new_chunk_center[:, 2].mean(),
        )

        side_length = 40
        center_point = np.array(
            [x, y, z]
        )  # Replace x, y, z with your point's coordinates
        half_side = side_length / 2.0
        min_bound = center_point - half_side
        max_bound = center_point + half_side
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound, max_bound=max_bound
        )

        # Crop the point cloud
        extracted_pcd = merge.crop(aabb)

        points_1 = np.asarray(extracted_pcd.points)
        points_2 = np.asarray(new_chunk.points)

        colors_1 = np.asarray(extracted_pcd.colors)
        colors_2 = np.asarray(new_chunk.colors)

        unique_colors_1 = np.unique(colors_1, axis=0)
        unique_colors_2 = np.unique(colors_2, axis=0)

        instance2point_1 = {}
        for i in range(unique_colors_1.shape[0]):
            if not np.all(unique_colors_1[i] == 0.0):  # Streets are black
                instance2point_1[i] = {}
                inds = np.where(np.all(colors_1 == unique_colors_1[i], axis=1))[0]
                instance2point_1[i]["points"] = points_1[inds]
                instance2point_1[i]["inds"] = inds

        instance2point_2 = {}
        for i in range(unique_colors_2.shape[0]):
            if not np.all(unique_colors_2[i] == 0.0):  # Streets are black
                instance2point_2[i] = {}
                inds = np.where(np.all(colors_2 == unique_colors_2[i], axis=1))[0]
                instance2point_2[i]["points"] = points_2[inds]
                instance2point_2[i]["inds"] = inds

        id_pairs_iou = []
        for id_1, entries_1 in instance2point_1.items():
            points1 = entries_1["points"]
            min_bound = np.min(points1, axis=0)
            max_bound = np.max(points1, axis=0)
            association = []
            for id_2, entries_2 in instance2point_2.items():
                points2 = entries_2["points"]
                intersection = np.where(
                    np.all(points2 >= min_bound, axis=1)
                    & np.all(points2 <= max_bound, axis=1)
                )[0].shape[0]
                if intersection > 0:
                    union = len(np.unique(np.concatenate((points1, points2))))
                    iou = float(intersection) / float(union)
                    if iou > 0.01:
                        association.append((id_2, iou))
            if len(association) != 0:
                for association_id, iou in association:
                    id_pairs_iou.append((id_1, (association_id, iou)))

        ids_chunk_1 = []
        ids_chunk_2 = []
        ious = []
        for id1, (id2, iou) in id_pairs_iou:
            if id2 not in ids_chunk_2:
                ids_chunk_1.append(id1)
                ids_chunk_2.append(id2)
                ious.append(iou)
            else:
                i = ids_chunk_2.index(id2)
                if iou > ious[i]:
                    ious[i] = iou
                    ids_chunk_1[i] = id1

        for id1, id2 in zip(ids_chunk_1, ids_chunk_2):
            inds2 = instance2point_2[id2]["inds"]
            colors_2[inds2] = unique_colors_1[id1]

        new_chunk_recolored = o3d.geometry.PointCloud()
        new_chunk_recolored.points = new_chunk.points
        new_chunk_recolored.colors = o3d.utility.Vector3dVector(colors_2)
        last_chunk = new_chunk_recolored

        merge += new_chunk_recolored
        merge.remove_duplicated_points()

    return merge


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / (np.linalg.norm(vector) + 1e-6)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
