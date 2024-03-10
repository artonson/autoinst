import numpy as np
import torch
import open3d as o3d
import os
import copy
from PIL import Image
import json

from hidden_points_removal import (
    hidden_point_removal_o3d,
    hidden_point_removal_biasutti,
)
from point_cloud_utils import (
    transform_pcd,
    filter_points_from_dict,
    get_pcd,
    point_to_label,
    change_point_indices,
)
from point_to_pixels import point_to_pixel
from visualization_utils import (
    unite_pcd_and_img,
    color_pcd_with_labels,
    visualize_associations_in_img,
)
from merge_pointclouds import (
    build_associations,
    apply_associations_to_dict,
    merge_label_predictions,
    merge_pointclouds,
    build_associations_across_timesteps,
)
from merged_sequences import merged_sequence
from image_utils import masks_to_image
from point_cloud_utils import (
    get_pcd,
    transform_pcd,
    kDTree_1NN_feature_reprojection,
    remove_isolated_points,
    get_subpcd,
    get_statistical_inlier_indices,
    merge_chunks_unite_instances,
)
from dataset_utils import *


SCANNET_COLOR_MAP_20 = {
    -1: (0.0, 0.0, 0.0),
    0: (174.0, 199.0, 232.0),
    1: (152.0, 223.0, 138.0),
    2: (31.0, 119.0, 180.0),
    3: (255.0, 187.0, 120.0),
    4: (188.0, 189.0, 34.0),
    5: (140.0, 86.0, 75.0),
    6: (255.0, 152.0, 150.0),
    7: (214.0, 39.0, 40.0),
    8: (197.0, 176.0, 213.0),
    9: (148.0, 103.0, 189.0),
    10: (196.0, 156.0, 148.0),
    11: (23.0, 190.0, 207.0),
    12: (247.0, 182.0, 210.0),
    13: (219.0, 219.0, 141.0),
    14: (255.0, 127.0, 14.0),
    15: (158.0, 218.0, 229.0),
    16: (44.0, 160.0, 44.0),
    17: (112.0, 128.0, 144.0),
    18: (227.0, 119.0, 194.0),
    19: (82.0, 84.0, 163.0),
}
import pointops
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import lil_matrix
from sklearn.cluster import Birch
from chunk_generation import (
    subsample_positions,
    chunks_from_pointcloud,
    indices_per_patch,
    tarl_features_per_patch,
    image_based_features_per_patch,
    dinov2_mean,
    get_indices_feature_reprojection,
)

import pointops
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import lil_matrix
from sklearn.cluster import Birch


class Voxelize(object):
    def __init__(
        self,
        voxel_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "normal", "color", "label"),
        return_discrete_coord=False,
        return_min_coord=False,
    ):
        self.voxel_size = voxel_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_discrete_coord = return_discrete_coord
        self.return_min_coord = return_min_coord

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        discrete_coord = np.floor(
            data_dict["coord"] / np.array(self.voxel_size)
        ).astype(int)
        min_coord = discrete_coord.min(0) * np.array(self.voxel_size)
        discrete_coord -= discrete_coord.min(0)
        key = self.hash(discrete_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1])
            idx_unique = idx_sort[idx_select]
            if self.return_discrete_coord:
                data_dict["discrete_coord"] = discrete_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                if self.return_discrete_coord:
                    data_part["discrete_coord"] = discrete_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


voxelize = Voxelize(voxel_size=0.05, mode="train", keys=("coord", "color", "group"))


def color_pcd_by_labels(pcd, labels, colors=None, gt_labels=None):

    if colors == None:
        colors = generate_random_colors(2000)
    pcd_colored = copy.deepcopy(pcd)
    pcd_colors = np.zeros(np.asarray(pcd.points).shape)
    if gt_labels is None:
        unique_labels = list(np.unique(labels))
    else:
        unique_labels = list(np.unique(gt_labels))
    background_color = np.array([0, 0, 0])

    # for i in range(len(pcd_colored.points)):
    for i in unique_labels:
        if i == -1:
            continue
        idcs = np.where(labels == i)
        idcs = idcs[0]
        if i == 0:
            pcd_colors[idcs] = background_color
        else:
            pcd_colors[idcs] = np.array(colors[unique_labels.index(i)])

        # if labels[i] != (-1):
        #    pcd_colored.colors[i] = np.array(colors[labels[i]]) / 255
    pcd_colored.colors = o3d.utility.Vector3dVector(pcd_colors / 255)
    return pcd_colored


def sam3d(
    dataset,
    indices,
    pcd_nonground_minor,
    T_pcd2world,
    sampled_indices_global,
    sequence=None,
    patchwise_indices=None,
    pcd_chunk=None,
):

    print("Start of sequence", sequence)
    first_id = patchwise_indices[sequence][0]
    chunk_indices = indices[sequence]

    cam_indices_global, _ = get_indices_feature_reprojection(
        sampled_indices_global, first_id, adjacent_frames=(10, 10)
    )

    cams = ["cam2", "cam3"]
    cam_ids = [0, 1]

    # seg_dict, pts = image_based_features_per_patch(dataset, pcd_nonground_minor, chunk_indices,  major_voxel_size, T_pcd,
    #                                                    cam_indices_global, cams, cam_ids=[0,1],hpr_radius=1000, num_dino_features=384, sam=True, dino=True, rm_perp=0.0)

    colors = np.zeros_like(np.asarray(pcd_chunk.points))
    pcd_list = []

    for i, points_index in enumerate(cam_indices_global):
        for cam_id in cam_ids:

            T_lidar2world = dataset.get_pose(points_index)

            T_world2lidar = np.linalg.inv(T_lidar2world)
            T_lidar2cam, K = dataset.get_calibration_matrices(cams[cam_id])
            image = dataset.get_image(cams[cam_id], points_index)
            T_world2cam = T_lidar2cam @ T_world2lidar
            T_pcd2cam = T_world2cam @ T_pcd2world

            sam_mask = dataset.get_sam_mask(cams[cam_id], points_index)
            sam_labels = masks_to_image(sam_mask)

            # hidden point removal
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(
                np.asarray(pcd_chunk.points).copy()
            )

            pcd_camframe = copy.deepcopy(pcd_chunk).transform(T_pcd2cam)
            pcd_chunk.paint_uniform_color([0, 0, 1])
            cam_frame_chunk = copy.deepcopy(pcd_chunk).transform(T_pcd2cam)

            # hpr stuff

            # visible_indices = hidden_point_removal_o3d(np.asarray(pcd_camframe.points), camera=[0,0,0], radius_factor=50)
            diameter = np.linalg.norm(
                np.asarray(pcd_camframe.get_max_bound())
                - np.asarray(pcd_camframe.get_min_bound())
            )

            radius = 200
            _, visible_indices = pcd_camframe.hidden_point_removal(
                [0, 0, 0], diameter * radius
            )

            frame_indices = list(set(visible_indices))
            # frame_indices = list(set(visible_indices) & set(chunk_and_inlier_indices))

            visible_chunk = get_subpcd(pcd_camframe, visible_indices)
            # visible_chunk = pcd_camframe

            # Project the points to pixels
            width, height = image.size
            points_to_pixels = point_to_pixel(
                np.asarray(visible_chunk.points), K, height, width
            )

            colors = np.zeros_like(np.asarray(visible_chunk.points))
            all_idcs = list(points_to_pixels.keys())
            group_ids = np.zeros((colors.shape[0],))
            for point_id in points_to_pixels.keys():
                x, y = (
                    points_to_pixels[point_id]["pixels"][0],
                    points_to_pixels[point_id]["pixels"][1],
                )
                colors[point_id] = image.getpixel((x, y))
                colors[point_id] = colors[point_id] / 255.0
                group_ids[point_id] = sam_labels[y, x]

            visible_chunk.colors = o3d.utility.Vector3dVector(colors[all_idcs])
            cam_frame_local_world = copy.deepcopy(visible_chunk).transform(
                np.linalg.inv(T_pcd2cam)
            )
            visible_chunk.points = o3d.utility.Vector3dVector(
                np.asarray(cam_frame_local_world.points)[all_idcs]
            )

            if (len(all_idcs)) == 0:
                continue
            group_ids = group_ids[all_idcs]
            coords = np.asarray(visible_chunk.points)
            cols = np.asarray(visible_chunk.colors)

            # print(group_ids.shape)
            # print(cols.shape)
            # print(coords.shape)
            save_dict = voxelize(dict(coord=coords, color=cols, group=group_ids))
            # pcd_new = color_pcd_by_labels(visible_chunk,group_ids)
            # o3d.visualization.draw_geometries([pcd_new])
            # image.show()
            pcd_list.append(save_dict)

    th = 200
    while len(pcd_list) != 1:
        print(len(pcd_list), flush=True)
        new_pcd_list = []
        for indice in pairwise_indices(len(pcd_list)):
            # print(indice)
            pcd_frame = cal_2_scenes(
                pcd_list, indice, voxel_size=0.05, voxelize=voxelize
            )
            if pcd_frame is not None:
                new_pcd_list.append(pcd_frame)
        pcd_list = new_pcd_list
    seg_dict = pcd_list[0]
    seg_dict["group"] += 1
    seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(seg_dict["coord"])

    # pcd = color_pcd_by_labels(pcd,seg_dict["group"])
    # o3d.visualization.draw_geometries([pcd])

    scene_coord = torch.tensor(np.asarray(pcd_chunk.points)).cuda().contiguous().float()
    new_offset = torch.tensor(scene_coord.shape[0]).cuda().float()
    gen_coord = torch.tensor(seg_dict["coord"]).cuda().contiguous().float()
    offset = torch.tensor(gen_coord.shape[0]).cuda().float()
    gen_group = seg_dict["group"]
    indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
    indices = indices.cpu().numpy()
    group = gen_group[indices.reshape(-1)].astype(np.int16)
    mask_dis = dis.reshape(-1).cpu().numpy() > 0.2
    group[mask_dis] = 0
    group = group.astype(np.int16)
    torch.save(num_to_natural(group), "test.pth")

    pcd_chunk = color_pcd_by_labels(pcd_chunk, group)
    # o3d.visualization.draw_geometries([pcd_chunk])
    return pcd_chunk


def reproject_merged_pointcloud(pcd_merge, dataset, start_index, sequence_length):
    """
    Args:
        pcd_merge:          merged point cloud
        dataset:            dataset object
        start_index:        index of first point cloud
        sequence_length:    number of point clouds to merge
    Returns:
        feature_matrix:     feature matrix of merged point cloud
    """

    left_cam = "cam2"

    pose_merge = dataset.get_pose(start_index)
    pcd_merge_t0 = pcd_merge
    num_points = pcd_merge_t0.shape[0]

    feature_matrix = lil_matrix((num_points, 0), dtype=np.int32)

    for i in range(sequence_length):
        points_index = start_index + i

        T_lidar2leftcam, K_leftcam = dataset.get_calibration_matrices(left_cam)

        left_label_PIL = dataset.get_sam_mask(left_cam, points_index)
        left_label = masks_to_colored_image(left_label_PIL)

        pose_to = np.linalg.inv(dataset.get_pose(points_index))

        pcd_merge_velo_ti = transform_pcd(pcd_merge_t0, pose_to)

        pcd_merge_leftcamframe = transform_pcd(pcd_merge_velo_ti, T_lidar2leftcam)

        point_to_pixel_dict_leftcam = point_to_pixel(
            pcd_merge_leftcamframe, K_leftcam, left_label.shape[0], left_label.shape[1]
        )

        point_to_label_dict_leftcam = point_to_label(
            point_to_pixel_dict_leftcam, left_label
        )

        instance_ids = list(set(point_to_label_dict_leftcam.values()))

        features = lil_matrix((num_points, len(instance_ids)), dtype=np.int32)

        for index, instance_id in point_to_label_dict_leftcam.items():
            features[index, instance_ids.index(instance_id)] = 1

        old_shape = feature_matrix.get_shape()
        new_shape = (old_shape[0], old_shape[1] + len(instance_ids))
        new_feature_matrix = lil_matrix(new_shape, dtype=np.int32)
        new_feature_matrix[:, : old_shape[1]] = feature_matrix
        new_feature_matrix[:, old_shape[1] :] = features
        feature_matrix = new_feature_matrix

    return feature_matrix


def our_sam3d(dataset, first_id, center_id, pcd_chunk, seq_length=10):

    _, pcd_merge = merged_sequence(
        dataset,
        max(0, first_id - 10),
        center_id + seq_length - first_id,
        pcd_chunk,
        labelling=True,
    )
    # feature_matrix = reproject_merged_pointcloud(np.array(pcd_merge.points), dataset, max(0,first_id-10), center_id + seq_length - first_id)

    # tSVD = TruncatedSVD(n_components=75)
    # transformed_data = tSVD.fit_transform(feature_matrix)
    # print('transformed shape',transformed_data.shape)

    # Birch is fastest, results are also quite nice
    # birch_model = Birch(threshold=0.1, n_clusters=100)
    # birch_model.fit(transformed_data)

    # labels = birch_model.predict(transformed_data)

    # colorspace = plt.cm.rainbow(np.linspace(0, 1, len(set(labels))))[:, :3]
    # colors = [colorspace[i] for i in labels]
    # pcd_merge.colors = o3d.utility.Vector3dVector(np.array(colors))
    # o3d.visualization.draw_geometries([pcd_merge])

    _, gen_group = np.unique(np.asarray(pcd_merge.colors), axis=0, return_inverse=True)

    scene_coord = torch.tensor(np.asarray(pcd_chunk.points)).cuda().contiguous().float()
    new_offset = torch.tensor(scene_coord.shape[0]).cuda().float()
    gen_coord = torch.tensor(pcd_merge.points).cuda().contiguous().float()
    offset = torch.tensor(gen_coord.shape[0]).cuda().float()
    indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
    group = gen_group[indices.cpu().numpy().reshape(-1)].astype(np.int16)
    mask_dis = dis.reshape(-1).cpu().numpy() > 0.2
    group[mask_dis] = 0
    group = group.astype(np.int16)
    pcd_chunk = color_pcd_by_labels(pcd_chunk, group)
    ##o3d.visualization.draw_geometries([pcd_chunk])

    return pcd_chunk


def make_open3d_point_cloud(input_dict, voxelize, th):
    input_dict["group"] = remove_small_group(input_dict["group"], th)
    # input_dict = voxelize(input_dict)

    xyz = input_dict["coord"]
    if np.isnan(xyz).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def cal_group(input_dict, new_input_dict, match_inds, ratio=0.5):
    group_0 = input_dict["group"].astype(np.int64)
    group_1 = new_input_dict["group"].astype(np.int64)
    group_1[group_1 != -1] += group_0.max() + 1

    unique_groups, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups, group_0_counts))
    unique_groups, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups, group_1_counts))

    # Calculate the group number correspondence of overlapping points
    group_overlap = {}
    for i, j in match_inds:
        group_i = group_1[i]
        group_j = group_0[j]
        if group_i == -1:
            group_1[i] = group_0[j]
            continue
        if group_j == -1:
            continue
        if group_i not in group_overlap:
            group_overlap[group_i] = {}
        if group_j not in group_overlap[group_i]:
            group_overlap[group_i][group_j] = 0
        group_overlap[group_i][group_j] += 1

    # Update group information for point cloud 1
    for group_i, overlap_count in group_overlap.items():
        # for group_j, count in overlap_count.items():
        max_index = np.argmax(np.array(list(overlap_count.values())))
        group_j = list(overlap_count.keys())[max_index]
        count = list(overlap_count.values())[max_index]
        total_count = min(group_0_counts[group_j], group_1_counts[group_i]).astype(
            np.float32
        )
        # print(count / total_count)
        if count / total_count >= ratio:
            group_1[group_1 == group_i] = group_j
    return group_1


def cal_2_scenes(pcd_list, index, voxel_size, voxelize, th=50):
    if len(index) == 1:
        return pcd_list[index[0]]
    # print(index, flush=True)
    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]
    pcd0 = make_open3d_point_cloud(input_dict_0, voxelize, th)
    pcd1 = make_open3d_point_cloud(input_dict_1, voxelize, th)
    if pcd0 == None:
        if pcd1 == None:
            return None
        else:
            return input_dict_1
    elif pcd1 == None:
        return input_dict_0

    # Cal Dul-overlap
    pcd0_tree = o3d.geometry.KDTreeFlann(copy.deepcopy(pcd0))
    match_inds = get_matching_indices(pcd1, pcd0_tree, 1.5 * voxel_size, 1)
    pcd1_new_group = cal_group(input_dict_0, input_dict_1, match_inds)
    # print(pcd1_new_group)

    pcd1_tree = o3d.geometry.KDTreeFlann(copy.deepcopy(pcd1))
    match_inds = get_matching_indices(pcd0, pcd1_tree, 1.5 * voxel_size, 1)
    input_dict_1["group"] = pcd1_new_group
    pcd0_new_group = cal_group(input_dict_1, input_dict_0, match_inds)
    # print(pcd0_new_group)

    pcd_new_group = np.concatenate((pcd0_new_group, pcd1_new_group), axis=0)
    pcd_new_group = num_to_natural(pcd_new_group)
    pcd_new_coord = np.concatenate(
        (input_dict_0["coord"], input_dict_1["coord"]), axis=0
    )
    pcd_new_color = np.concatenate(
        (input_dict_0["color"], input_dict_1["color"]), axis=0
    )
    pcd_dict = dict(coord=pcd_new_coord, color=pcd_new_color, group=pcd_new_group)

    pcd_dict = voxelize(pcd_dict)
    return pcd_dict


def overlap_percentage(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    area_intersection = np.sum(intersection)

    area_mask1 = np.sum(mask1)
    area_mask2 = np.sum(mask2)

    smaller_area = min(area_mask1, area_mask2)

    return area_intersection / smaller_area


def remove_samll_masks(masks, ratio=0.8):
    filtered_masks = []
    skip_masks = set()

    for i, mask1_dict in enumerate(masks):
        if i in skip_masks:
            continue

        should_keep = True
        for j, mask2_dict in enumerate(masks):
            if i == j or j in skip_masks:
                continue
            mask1 = mask1_dict["segmentation"]
            mask2 = mask2_dict["segmentation"]
            overlap = overlap_percentage(mask1, mask2)
            if overlap > ratio:
                if np.sum(mask1) < np.sum(mask2):
                    should_keep = False
                    break
                else:
                    skip_masks.add(j)

        if should_keep:
            filtered_masks.append(mask1)

    return filtered_masks


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones_like(coord) if color is None else color
    )
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")


def remove_small_group(group_ids, th):
    unique_elements, counts = np.unique(group_ids, return_counts=True)
    result = group_ids.copy()
    for i, count in enumerate(counts):
        if count < th:
            result[group_ids == unique_elements[i]] = 0

    return result


def pairwise_indices(length):
    return [[i, i + 1] if i + 1 < length else [i] for i in range(0, length, 2)]


def num_to_natural(group_ids):
    """
    Change the group number to natural number arrangement
    """
    if np.all(group_ids == -1):
        return group_ids
    array = copy.deepcopy(group_ids).astype(np.int64)
    unique_values = np.unique(array[array != -1]).astype(np.int64)
    mapping = np.full(int(np.max(unique_values) + 2), -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array


def get_matching_indices(source, pcd_tree, search_voxel_size, K=None):
    match_inds = []
    for i, point in enumerate(source.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            # match_inds[i, j] = 1
            match_inds.append((i, j))
    return match_inds


def visualize_3d(data_dict, text_feat_path, save_path):
    text_feat = torch.load(text_feat_path)
    group_logits = np.einsum("nc,mc->nm", data_dict["group_feat"], text_feat)
    group_labels = np.argmax(group_logits, axis=-1)
    labels = group_labels[data_dict["group"]]
    labels[data_dict["group"] == -1] = -1
    visualize_pcd(data_dict["coord"], data_dict["color"], labels, save_path)


def visualize_pcd(coord, pcd_color, labels, save_path):
    # alpha = 0.5
    label_color = np.array([SCANNET_COLOR_MAP_20[label] for label in labels])
    # overlay = (pcd_color * (1-alpha) + label_color * alpha).astype(np.uint8) / 255
    label_color = label_color / 255
    save_point_cloud(coord, label_color, save_path)


def visualize_2d(img_color, labels, img_size, save_path):
    import matplotlib.pyplot as plt

    # from skimage.segmentation import mark_boundaries
    # from skimage.color import label2rgb
    label_names = [
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refridgerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "other",
    ]
    colors = np.array(list(SCANNET_COLOR_MAP_20.values()))[1:]
    segmentation_color = np.zeros((img_size[0], img_size[1], 3))
    for i, color in enumerate(colors):
        segmentation_color[labels == i] = color
    alpha = 1
    overlay = (img_color * (1 - alpha) + segmentation_color * alpha).astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(overlay)
    patches = [
        plt.plot([], [], "s", color=np.array(color) / 255, label=label)[0]
        for label, color in zip(label_names, colors)
    ]
    plt.legend(
        handles=patches,
        bbox_to_anchor=(0.5, -0.1),
        loc="upper center",
        ncol=4,
        fontsize="small",
    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def visualize_partition(coord, group_id, save_path):
    group_id = group_id.reshape(-1)
    num_groups = group_id.max() + 1
    group_colors = np.random.rand(num_groups, 3)
    group_colors = np.vstack((group_colors, np.array([0, 0, 0])))
    color = group_colors[group_id]
    save_point_cloud(coord, color, save_path)


def delete_invalid_group(group, group_feat):
    indices = np.unique(group[group != -1])
    group = num_to_natural(group)
    group_feat = group_feat[indices]
    return group, group_feat
