from dataset.kitti_odometry_dataset import (
    KittiOdometryDataset,
    KittiOdometryDatasetConfig,
)
from dataset.filters.filter_list import FilterList
from dataset.filters.kitti_gt_mo_filter import KittiGTMovingObjectFilter
from dataset.filters.range_filter import RangeFilter

import os
import numpy as np
import copy
import open3d as o3d

from aggregate_pointcloud import aggregate_pointcloud
from chunk_generation import subsample_positions, chunks_from_pointcloud
from visualization_utils import generate_random_colors_map
import numpy as np
import matplotlib.pyplot as plt
import random

# Generate 30 different colors
COLORS = plt.cm.viridis(np.linspace(0, 1, 30))
COLORS = list(list(col) for col in COLORS)
COLORS = [tuple(col[:3]) for col in COLORS]


def generate_random_colors(N):
    colors = set()  # Use a set to store unique colors
    while len(colors) < N:  # Keep generating colors until we have N unique ones
        # Generate a random color and add it to the set
        colors.add(
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        )

    return list(colors)  # Convert the set to a list before returning


def masks_to_colored_image(masks):
    """
    Function that takes an array of masks and returns a pixel-wise colored label map.
    Assumes each mask in masks is a dictionary with a "segmentation" key containing a binary mask.
    """

    # Assuming all masks are the same size, get the dimensions from the first mask
    height, width = masks[0]["segmentation"].shape
    image_labels = np.zeros(
        (height, width, 3), dtype=np.uint8
    )  # Use uint8 for image data

    colors = generate_random_colors(len(masks))  # Generate colors for each mask

    for i, mask in enumerate(masks):
        # Apply the color to the region specified by the mask
        for c in range(3):  # Iterate over color channels
            image_labels[:, :, c][mask["segmentation"]] = colors[i][c]

    return image_labels

def color_pcd_by_labels(pcd, labels, colors=None, gt_labels=None, semantics=False):

    if colors == None:
        colors = generate_random_colors(2000)
    pcd_colored = copy.deepcopy(pcd)
    pcd_colors = np.zeros(np.asarray(pcd.points).shape)
    if gt_labels is None:
        unique_labels = list(np.unique(labels))
    else:
        unique_labels = list(np.unique(gt_labels))

    background_color = np.array([0, 0, 0])

    for i in unique_labels:
        if i == -1:
            continue
        idcs = np.where(labels == i)
        idcs = idcs[0]
        if i == 0:
            pcd_colors[idcs] = background_color
        else:
            pcd_colors[idcs] = np.array(colors[unique_labels.index(i)])

    if semantics:
        pcd_colored.colors = o3d.utility.Vector3dVector(pcd_colors)
    else:
        pcd_colored.colors = o3d.utility.Vector3dVector(pcd_colors / 255)
    return pcd_colored


def create_kitti_odometry_dataset(
    dataset_path,
    sequence_num,
    cache=True,
    sam_folder_name="sam_pred_underseg",
    dinov2_folder_name="dinov2_features",
    correct_scan_calibration=True,
    range_min=3,
    range_max=25,
    ncuts_mode=True,
):
    if ncuts_mode:
        filters = FilterList(
            [
                KittiGTMovingObjectFilter(
                    os.path.join(
                        dataset_path, "sequences", "%.2d" % sequence_num, "labels"
                    )
                ),
                RangeFilter(range_min, range_max),
            ]
        )
    else:
        filters = FilterList([])

    config_filtered = KittiOdometryDatasetConfig(
        cache=cache,
        dataset_path=dataset_path,
        sam_folder_name=sam_folder_name,
        dinov2_folder_name=dinov2_folder_name,
        correct_scan_calibration=correct_scan_calibration,
        filters=filters,
        dist_threshold=None,
    )
    dataset = KittiOdometryDataset(config_filtered, sequence_num)
    return dataset


def create_kitti_odometry_dataset_no_filter(
    dataset_path,
    sequence_num,
    cache=True,
    sam_folder_name="sam_pred_underseg",
    dinov2_folder_name="dinov2_features",
    correct_scan_calibration=True,
    range_min=3,
    range_max=25,
    ncuts_mode=True,
):
    if ncuts_mode:
        filters = FilterList(
            [
                RangeFilter(range_min, range_max),
            ]
        )
    else:
        filters = FilterList([])

    config_filtered = KittiOdometryDatasetConfig(
        cache=cache,
        dataset_path=dataset_path,
        sam_folder_name=sam_folder_name,
        dinov2_folder_name=dinov2_folder_name,
        correct_scan_calibration=correct_scan_calibration,
        filters=filters,
        dist_threshold=None,
    )
    dataset = KittiOdometryDataset(config_filtered, sequence_num)
    return dataset

def process_and_save_point_clouds(
    dataset,
    ind_start,
    ind_end,
    ground_segmentation_method="patchwork",
    icp=True,
    minor_voxel_size=0.05,
    major_voxel_size=0.35,
    out_folder=None,
    sequence_num=7,
    cur_idx=0,
):

    if os.path.exists(out_folder) == False:
        os.makedirs(out_folder)

    pcd_ground, pcd_nonground, all_poses, T_pcd, labels = aggregate_pointcloud(
        dataset,
        ind_start,
        ind_end,
        ground_segmentation=ground_segmentation_method,
        icp=icp,
    )

    # Saving point clouds and poses
    sequence_num = str(sequence_num)
    if pcd_ground is not None:
        o3d.io.write_point_cloud(
            f"{out_folder}ground{sequence_num}_{cur_idx}.pcd",
            pcd_ground,
            write_ascii=False,
            compressed=False,
            print_progress=False,
        )
    o3d.io.write_point_cloud(
        f"{out_folder}non_ground{sequence_num}_{cur_idx}.pcd",
        pcd_nonground,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )

    np.savez(
        f"{out_folder}all_poses_" + str(sequence_num) + "_" + str(cur_idx) + ".npz",
        all_poses=all_poses,
        T_pcd=T_pcd,
    )
    np.savez(
        f"{out_folder}kitti_labels_" + str(sequence_num) + "_" + str(cur_idx) + ".npz",
        seg_ground=np.vstack(labels["seg_ground"]),
        seg_nonground=np.vstack(labels["seg_nonground"]),
        instance_ground=np.vstack(labels["instance_ground"]),
        instance_nonground=np.vstack(labels["instance_nonground"]),
    )

    return labels


def load_and_downsample_point_clouds(
    out_folder, sequence_num, minor_voxel_size=0.05, ground_mode=True, cur_idx=0
):
    # Load saved data
    with np.load(f"{out_folder}all_poses_{sequence_num}_{cur_idx}.npz") as data:
        all_poses = data["all_poses"]
        T_pcd = data["T_pcd"]
        first_position = T_pcd[:3, 3]

    # Load point clouds
    print("pcd load")
    if ground_mode is not None:
        pcd_ground = o3d.io.read_point_cloud(
            f"{out_folder}ground{sequence_num}_{cur_idx}.pcd"
        )
    pcd_nonground = o3d.io.read_point_cloud(
        f"{out_folder}non_ground{sequence_num}_{cur_idx}.pcd"
    )

    # Downsampling
    kitti_data1 = {}
    with np.load(
        f"{out_folder}kitti_labels_" + str(sequence_num) + "_" + str(cur_idx) + ".npz"
    ) as data:
        kitti_data1["seg_ground"] = data["seg_ground"]
        kitti_data1["seg_nonground"] = data["seg_nonground"]
        kitti_data1["instance_ground"] = data["instance_ground"]
        kitti_data1["instance_nonground"] = data["instance_nonground"]


    instances = np.hstack(
        (
            kitti_data1["instance_nonground"].reshape(
                -1,
            ),
            kitti_data1["instance_ground"].reshape(
                -1,
            ),
        )
    )

    semantics = np.hstack(
        (
            kitti_data1["seg_nonground"].reshape(
                -1,
            ),
            kitti_data1["seg_ground"].reshape(
                -1,
            ),
        )
    )
    colors = generate_random_colors_map(600)
    instance_non_ground_orig = color_pcd_by_labels(
        pcd_nonground,
        kitti_data1["instance_nonground"],
        colors=colors,
        gt_labels=instances,
    )
    
    instance_ground_orig = color_pcd_by_labels(
        pcd_ground, kitti_data1["instance_ground"], colors=colors, gt_labels=instances
    )

    
    semantic_non_ground_orig = color_pcd_by_labels(
        pcd_nonground,
        kitti_data1["seg_nonground"],
        colors=COLORS,
        gt_labels=semantics,
        semantics=True,
    )
    semantic_ground_orig = color_pcd_by_labels(
        pcd_ground,
        kitti_data1["seg_ground"],
        colors=COLORS,
        gt_labels=semantics,
        semantics=True,
    )
    kitti_data = {}
    pcd_ground_minor, _, _ = pcd_ground.voxel_down_sample_and_trace(
        minor_voxel_size, pcd_ground.get_min_bound(), pcd_ground.get_max_bound(), False
    )
    pcd_nonground_minor, _, _ = pcd_nonground.voxel_down_sample_and_trace(
        minor_voxel_size,
        pcd_nonground.get_min_bound(),
        pcd_nonground.get_max_bound(),
        False,
    )

    del pcd_ground, pcd_nonground

    print("downsample")
    # KDTree for finding nearest neighbors
    pcd_tree = o3d.geometry.KDTreeFlann(instance_non_ground_orig)
    instance_non_ground = o3d.geometry.PointCloud()
    instance_non_ground.points = pcd_nonground_minor.points

    # Map colors from original to downsampled point cloud
    new_colors = []
    new_labels = []
    for point in instance_non_ground.points:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        point_color = np.asarray(instance_non_ground_orig.colors)[idx[0]]
        
        new_colors.append(point_color)
        new_labels.append(kitti_data1["instance_nonground"][idx[0]])
    # import pdb; pdb.set_trace()

    instance_non_ground.colors = o3d.utility.Vector3dVector(new_colors)
    kitti_data["instance_nonground"] = new_labels

    pcd_tree = o3d.geometry.KDTreeFlann(instance_ground_orig)
    instance_ground = o3d.geometry.PointCloud()
    instance_ground.points = pcd_ground_minor.points

    # Map colors from original to downsampled point cloud
    new_colors = []
    new_labels = []
    for point in instance_ground.points:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        point_color = np.asarray(instance_ground_orig.colors)[idx[0]]
        new_colors.append(point_color)
        new_labels.append(kitti_data1["instance_ground"][idx[0]])

    instance_ground.colors = o3d.utility.Vector3dVector(new_colors)
    kitti_data["instance_ground"] = new_labels

    pcd_tree = o3d.geometry.KDTreeFlann(semantic_ground_orig)
    seg_ground = o3d.geometry.PointCloud()
    seg_ground.points = pcd_ground_minor.points

    # Map colors from original to downsampled point cloud
    new_colors = []
    new_labels = []
    for point in seg_ground.points:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        try:
            point_color = np.asarray(semantic_ground_orig.colors)[idx[0]]
            cur_label = kitti_data1["seg_ground"][idx[0]]
        except:
            import pdb

            pdb.set_trace()
        new_colors.append(point_color)
        new_labels.append(cur_label)

    seg_ground.colors = o3d.utility.Vector3dVector(new_colors)
    kitti_data["seg_ground"] = np.asarray(new_labels)

    pcd_tree = o3d.geometry.KDTreeFlann(semantic_non_ground_orig)
    seg_nonground = o3d.geometry.PointCloud()
    seg_nonground.points = pcd_nonground_minor.points

    # Map colors from original to downsampled point cloud
    new_colors = []
    new_labels = []
    for point in seg_nonground.points:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        point_color = np.asarray(semantic_non_ground_orig.colors)[idx[0]]
        cur_label = kitti_data1["seg_nonground"][idx[0]]
        new_colors.append(point_color)
        new_labels.append(cur_label)
    
    seg_nonground.colors = o3d.utility.Vector3dVector(new_colors)
    kitti_data["seg_nonground"] = np.asarray(new_labels)

    print("done downsample")
    return (
        pcd_ground_minor,
        pcd_nonground_minor,
        all_poses,
        T_pcd,
        first_position,
        kitti_data,
    )


def subsample_and_extract_positions(
    all_poses, voxel_size=1, ind_start=0, sequence_num=0, out_folder=None, cur_idx=0
):

    # Extracting positions from poses
    all_positions = [tuple(p[:3, 3]) for p in all_poses]

    # Performing subsampling
    sampled_indices_local = list(
        subsample_positions(all_positions, voxel_size=voxel_size)
    )
    sampled_indices_global = list(
        subsample_positions(all_positions, voxel_size=voxel_size) + ind_start
    )

    # Selecting a subset of poses and positions
    poses = np.array(all_poses)[sampled_indices_local]
    positions = np.array(all_positions)[sampled_indices_local]

    np.savez(
        f"{out_folder}subsampled_data{sequence_num}_{cur_idx}.npz",
        poses=poses,
        positions=positions,
        sampled_indices_global=sampled_indices_global,
        sampled_indices_local=sampled_indices_local,
    )

    return poses, positions, sampled_indices_local, sampled_indices_global


def chunk_and_downsample_point_clouds(
    dataset,
    pcd_nonground_minor,
    pcd_ground_minor,
    T_pcd,
    positions,
    first_position,
    sampled_indices_global,
    chunk_size=np.array([25, 25, 25]),
    overlap=3,
    major_voxel_size=0.35,
    kitti_labels=None,
):
    # Creating chunks
    (
        pcd_nonground_chunks,
        indices,
        center_positions,
        center_ids,
        chunk_bounds,
        kitti_out,
        obbs,
    ) = chunks_from_pointcloud(
        dataset,
        pcd_nonground_minor,
        T_pcd,
        positions,
        first_position,
        sampled_indices_global,
        chunk_size,
        overlap,
        labels=kitti_labels,
        chunk_size=chunk_size,
    )

    pcd_ground_chunks, indices_ground, _, _, _, kitti_out_ground, obbs = (
        chunks_from_pointcloud(
            dataset,
            pcd_ground_minor,
            T_pcd,
            positions,
            first_position,
            sampled_indices_global,
            chunk_size,
            overlap,
            labels=kitti_labels,
            ground=True,
            chunk_size=chunk_size,
        )
    )

    # Downsampling the chunks and printing information
    kitti_labels = {"nonground": kitti_out, "ground": kitti_out_ground}
    pcd_nonground_chunks_major_downsampling = []
    pcd_ground_chunks_major_downsampling = []
    for ground, nonground in zip(pcd_ground_chunks, pcd_nonground_chunks):
        downsampled_nonground = nonground.voxel_down_sample(voxel_size=major_voxel_size)
        downsampled_ground = ground.voxel_down_sample(voxel_size=major_voxel_size)
        print(
            "Downsampled from",
            np.asarray(nonground.points).shape,
            "to",
            np.asarray(downsampled_nonground.points).shape,
            "points (non-ground)",
        )
        print(
            "Downsampled from",
            np.asarray(ground.points).shape,
            "to",
            np.asarray(downsampled_ground.points).shape,
            "points (ground)",
        )

        pcd_nonground_chunks_major_downsampling.append(downsampled_nonground)
        pcd_ground_chunks_major_downsampling.append(downsampled_ground)

    return (
        pcd_nonground_chunks,
        pcd_ground_chunks,
        pcd_nonground_chunks_major_downsampling,
        pcd_ground_chunks_major_downsampling,
        indices,
        indices_ground,
        center_positions,
        center_ids,
        chunk_bounds,
        kitti_labels,
        obbs,
    )
