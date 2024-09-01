import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

src_path = os.path.abspath("../..")
import copy
import json
if src_path not in sys.path:
    sys.path.append(src_path)
from dataset.kitti_odometry_dataset import (
    KittiOdometryDataset,
    KittiOdometryDatasetConfig,
)
from dataset.filters.filter_list import FilterList
from dataset.filters.kitti_gt_mo_filter import KittiGTMovingObjectFilter
from dataset.filters.range_filter import RangeFilter
from dataset.filters.apply_pose import ApplyPose
import scipy
from scipy.spatial.distance import cdist
from tqdm import tqdm
from normalized_cut import normalized_cut
from ncuts_utils import (
    ncuts_chunk,
    kDTree_1NN_feature_reprojection_colors,
    get_merge_pcds,
)
from dataset_utils import *

from point_cloud_utils import (
    get_pcd,
    transform_pcd,
    kDTree_1NN_feature_reprojection,
    remove_isolated_points,
    get_subpcd,
    get_statistical_inlier_indices,
    merge_chunks_unite_instances,
)
from aggregate_pointcloud import aggregate_pointcloud

from visualization_utils import (
    generate_random_colors,
    color_pcd_by_labels,
    generate_random_colors_map,
)

from sam_label_distace import sam_label_distance

from chunk_generation import (
    subsample_positions,
    chunks_from_pointcloud,
    indices_per_patch,
    tarl_features_per_patch,
    image_based_features_per_patch,
    dinov2_mean,
    get_indices_feature_reprojection,
)

from point_cloud_utils import *
from visualization_utils import *
from visualization_utils import generate_random_colors_map
from metrics.metrics_class import Metrics

config_tarl_spatial_dino = {
    "name": "spatial_1.0_tarl_0.5_dino_0.1_t_0.005",
    "out_folder": "ncuts_data_tarl_dino_spatial/",
    "gamma": 0.1,
    "alpha": 1.0,
    "theta": 0.5,
    "T": 0.005,
    "gt": True,
}

config_tarl_spatial = {
    "name": "spatial_1.0_tarl_0.5_t_0.03",
    "out_folder": "ncuts_data_tarl_spatial/",
    "gamma": 0.0,
    "alpha": 1.0,
    "theta": 0.5,
    "T": 0.03,
    "gt": True,
}

config_spatial = {
    "name": "spatial_1.0_t_0.075",
    "out_folder": "ncuts_data_spatial/",
    "gamma": 0.0,
    "alpha": 1.0,
    "theta": 0.0,
    "T": 0.075,
    "gt": True,
}

config_maskpls_tarl_spatial = {
    "name": "maskpls_comp_",
    "out_folder": "maskpls_7/",
    "gamma": 0.0,
    "alpha": 0.0,
    "theta": 0.0,
    "T": 0.0,
    "gt": True,
}


config_maskpls_tarl_spatial_dino = {
    "name": "maskpls_no_filter_5_",
    "out_folder": "maskpls_no_filter_5/",
    "gamma": 0.0,
    "alpha": 0.0,
    "theta": 0.0,
    "T": 0.0,
    "gt": True,
}

start_chunk = 0
start_seq = 0
seqs = list(range(0, 11))
# seqs = [8, 10]
config = config_spatial
if 'maskpls' in config["name"]:
    from predict_maskpls import RefinerModel

print("Starting with config ", config)


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
            try:
                pcd_colors[idcs] = np.array(colors[unique_labels.index(i)])
            except:
                import pdb

                pdb.set_trace()

        # if labels[i] != (-1):
        #    pcd_colored.colors[i] = np.array(colors[labels[i]]) / 255
    pcd_colored.colors = o3d.utility.Vector3dVector(pcd_colors / 255)
    return pcd_colored


def generate_random_colors(N, seed=0):
    colors = set()  # Use a set to store unique colors
    while len(colors) < N:  # Keep generating colors until we have N unique ones
        # Generate a random color and add it to the set
        colors.add(
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        )

    return list(colors)  # Convert the set to a list before returning


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
    # for i in range(len(pcd_colored.points)):
    for i in unique_labels:
        if i == -1:
            continue
        idcs = np.where(labels == i)
        idcs = idcs[0]
        if i == 0 and semantics == False:
            pcd_colors[idcs] = background_color
        else:
            pcd_colors[idcs] = np.array(colors[unique_labels.index(i)])

    pcd_colored.colors = o3d.utility.Vector3dVector(pcd_colors / 255.0)
    return pcd_colored


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


def merge_unite_gt(chunks):
    last_chunk = chunks[0]
    merge = o3d.geometry.PointCloud()
    merge += last_chunk

    for new_chunk in chunks[1:]:
        merge += new_chunk

    merge.remove_duplicated_points()
    return merge


def create_folder(name):
    if os.path.exists(name) == False:
        os.makedirs(name)


def divide_indices_into_chunks(max_index, chunk_size=1000):
    chunks = []
    for start in range(0, max_index, chunk_size):
        end = min(start + chunk_size, max_index)
        chunks.append((start, end))
    return chunks


DATASET_PATH = os.path.join("/media/cedric/Datasets3/semantic_kitti/")
import shutil
import gc

minor_voxel_size = 0.05
major_voxel_size = 0.35
chunk_size = np.array([25, 25, 25])  # meters
overlap = 3  # meters
ground_segmentation_method = "patchwork"
NCUT_ground = False

out_folder = "pcd_preprocessed/semantics/"

out_folder_semantics = out_folder + "semantics/"
if os.path.exists(out_folder_semantics) == False:
    os.makedirs(out_folder_semantics)

out_folder_instances = out_folder + "instances/"
if os.path.exists(out_folder_instances) == False:
    os.makedirs(out_folder_instances)

out_folder_ncuts = out_folder + config["out_folder"]
if os.path.exists(out_folder_ncuts) == False:
    os.makedirs(out_folder_ncuts)

data_store_train = out_folder + "train/"
if os.path.exists(data_store_train) == False:
    os.makedirs(data_store_train)


color_dict_normalized = {
    0: [0.0, 0.0, 0.0],
    1: [0.0, 0.0, 1.0],
    10: [0.9607843137254902, 0.5882352941176471, 0.39215686274509803],
    11: [0.9607843137254902, 0.9019607843137255, 0.39215686274509803],
    13: [0.9803921568627451, 0.3137254901960784, 0.39215686274509803],
    15: [0.5882352941176471, 0.23529411764705882, 0.11764705882352941],
    16: [1.0, 0.0, 0.0],
    18: [0.7058823529411765, 0.11764705882352941, 0.3137254901960784],
    20: [1.0, 0.0, 0.0],
    30: [0.11764705882352941, 0.11764705882352941, 1.0],
    31: [0.7843137254901961, 0.1568627450980392, 1.0],
    32: [0.35294117647058826, 0.11764705882352941, 0.5882352941176471],
    40: [1.0, 0.0, 1.0],
    44: [1.0, 0.5882352941176471, 1.0],
    48: [0.29411764705882354, 0.0, 0.29411764705882354],
    49: [0.29411764705882354, 0.0, 0.6862745098039216],
    50: [0.0, 0.7843137254901961, 1.0],
    51: [0.19607843137254902, 0.47058823529411764, 1.0],
    52: [0.0, 0.5882352941176471, 1.0],
    60: [0.6666666666666666, 1.0, 0.5882352941176471],
    70: [0.0, 0.6862745098039216, 0.0],
    71: [0.0, 0.23529411764705882, 0.5294117647058824],
    72: [0.3137254901960784, 0.9411764705882353, 0.5882352941176471],
    80: [0.5882352941176471, 0.9411764705882353, 1.0],
    81: [0.0, 0.0, 1.0],
    99: [1.0, 1.0, 0.19607843137254902],
    252: [0.9607843137254902, 0.5882352941176471, 0.39215686274509803],
    256: [1.0, 0.0, 0.0],
    253: [0.7843137254901961, 0.1568627450980392, 1.0],
    254: [0.11764705882352941, 0.11764705882352941, 1.0],
    255: [0.35294117647058826, 0.11764705882352941, 0.5882352941176471],
    257: [0.9803921568627451, 0.3137254901960784, 0.39215686274509803],
    258: [0.7058823529411765, 0.11764705882352941, 0.3137254901960784],
    259: [1.0, 0.0, 0.0],
}

reverse_color_dict = {tuple(v): k for k, v in color_dict_normalized.items()}


def color_pcd_kitti(pcd, labels):
    unique_labels = np.unique(labels)
    pcd_colors = np.zeros_like(np.asarray(pcd.points))
    for i in unique_labels:
        idcs = np.where(labels == i)
        idcs = idcs[0]
        pcd_colors[idcs] = np.array(color_dict_normalized[int(i)])

    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    return pcd


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

    # merge.remove_duplicated_points()
    return output_semantics


alpha = config["alpha"]
theta = config["theta"]
colors = generate_random_colors_map(6000)
beta = 0.0
tarl_norm = False
gamma = config["gamma"]
proximity_threshold = 1.0
ncuts_threshold = config["T"]


exclude = [1, 4]
seqs = [7]
for seq in seqs:
    # maskpls = RefinerModel()
    if seq in exclude:
        continue
    print("Sequence", seq)
    SEQUENCE_NUM = seq
    dataset = create_kitti_odometry_dataset(
        DATASET_PATH, SEQUENCE_NUM, ncuts_mode=True, sam_folder_name="sam_pred_underseg"
    )
    chunks_idcs = divide_indices_into_chunks(len(dataset))

    data_store_folder = out_folder + str(SEQUENCE_NUM) + "/"
    if os.path.exists(data_store_folder) == False:
        os.makedirs(data_store_folder)

    data_store_folder_train_cur = data_store_train + str(SEQUENCE_NUM) + "/"
    if os.path.exists(data_store_folder_train_cur) == False:
        os.makedirs(data_store_folder_train_cur)

    print("STORE FOLDER", data_store_folder)

    for cur_idx, cidcs in enumerate(chunks_idcs[start_chunk:]):
        # if cur_idx == 0 :
        #         continue

        ind_start = cidcs[0]
        ind_end = cidcs[1]
        cur_idx = int(ind_start / 1000)
        if ind_end - ind_start < 200:  ##avoid use of small maps
            continue

        print("ind start", ind_start)
        print("ind end", ind_end)

        if "maskpls" in config["name"] and config["name"] != "maskpls_supervised":
            maskpls = RefinerModel(dataset="kitti")

        if config["name"] == "maskpls_supervised":
            print("Supervised model")
            maskpls = RefinerModelSupervised()

        if (
            os.path.exists(f"{out_folder}non_ground{SEQUENCE_NUM}_{cur_idx}.pcd")
            == False
        ):
            print("process poses")
            process_and_save_point_clouds(
                dataset,
                ind_start,
                ind_end,
                minor_voxel_size=minor_voxel_size,
                major_voxel_size=major_voxel_size,
                icp=False,
                out_folder=out_folder,
                sequence_num=SEQUENCE_NUM,
                ground_segmentation_method=ground_segmentation_method,
                cur_idx=cur_idx,
            )

        if (
            os.path.exists(
                f"{out_folder}pcd_nonground_minor{SEQUENCE_NUM}_{cur_idx}.pcd"
            )
            == False
        ):
            #    if True == True :
            print("load and downsample points")
            (
                pcd_ground_minor,
                pcd_nonground_minor,
                all_poses,
                T_pcd,
                first_position,
                kitti_labels,
            ) = load_and_downsample_point_clouds(
                out_folder,
                SEQUENCE_NUM,
                minor_voxel_size,
                ground_mode=ground_segmentation_method,
                cur_idx=cur_idx,
            )

            print("write pcds")
            print(pcd_ground_minor)
            o3d.io.write_point_cloud(
                f"{out_folder}pcd_ground_minor{SEQUENCE_NUM}_{cur_idx}.pcd",
                pcd_ground_minor,
                write_ascii=False,
                compressed=False,
                print_progress=True,
            )
            o3d.io.write_point_cloud(
                f"{out_folder}pcd_nonground_minor{SEQUENCE_NUM}_{cur_idx}.pcd",
                pcd_nonground_minor,
                write_ascii=False,
                compressed=False,
                print_progress=True,
            )
            print("write labels")
            np.savez(
                f"{out_folder}kitti_labels_preprocessed{SEQUENCE_NUM}_{cur_idx}.npz",
                instance_nonground=kitti_labels["instance_nonground"],
                instance_ground=kitti_labels["instance_ground"],
                seg_ground=kitti_labels["seg_ground"],
                seg_nonground=kitti_labels["seg_nonground"],
            )
        print("load pcd")
        pcd_ground_minor = o3d.io.read_point_cloud(
            f"{out_folder}pcd_ground_minor{SEQUENCE_NUM}_{cur_idx}.pcd"
        )
        pcd_nonground_minor = o3d.io.read_point_cloud(
            f"{out_folder}pcd_nonground_minor{SEQUENCE_NUM}_{cur_idx}.pcd"
        )

        print("load data")
        kitti_labels_orig = {}
        with np.load(
            f"{out_folder}kitti_labels_preprocessed{SEQUENCE_NUM}_{cur_idx}.npz"
        ) as data:
            kitti_labels_orig["instance_ground"] = data["instance_ground"]
            kitti_labels_orig["instance_nonground"] = data["instance_nonground"]
            kitti_labels_orig["seg_nonground"] = data["seg_nonground"]
            kitti_labels_orig["seg_ground"] = data["seg_ground"]

        with np.load(
            f"{out_folder}all_poses_" + str(SEQUENCE_NUM) + "_" + str(cur_idx) + ".npz"
        ) as data:
            all_poses = data["all_poses"]
            T_pcd = data["T_pcd"]
            first_position = T_pcd[:3, 3]

        print("pose downsample")
        if (
            os.path.exists(f"{out_folder}subsampled_data{SEQUENCE_NUM}_{cur_idx}.npz")
            == False
        ):
            poses, positions, sampled_indices_local, sampled_indices_global = (
                subsample_and_extract_positions(
                    all_poses,
                    ind_start=ind_start,
                    sequence_num=SEQUENCE_NUM,
                    out_folder=out_folder,
                    cur_idx=cur_idx,
                )
            )

        with np.load(
            f"{out_folder}subsampled_data{SEQUENCE_NUM}_{cur_idx}.npz"
        ) as data:
            poses = data["poses"]
            positions = data["positions"]
            sampled_indices_local = list(data["sampled_indices_local"])
            sampled_indices_global = list(data["sampled_indices_global"])

        print("chunk downsample")
        (
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
            _,
        ) = chunk_and_downsample_point_clouds(
            dataset,
            pcd_nonground_minor,
            pcd_ground_minor,
            T_pcd,
            positions,
            first_position,
            sampled_indices_global,
            chunk_size=chunk_size,
            overlap=overlap,
            major_voxel_size=major_voxel_size,
            kitti_labels=kitti_labels_orig,
        )

        print("finished downsample")

        out_folder_ncuts_cur = (
            out_folder_ncuts + str(SEQUENCE_NUM) + "_" + str(cur_idx) + "/"
        )
        out_folder_semantics_cur = (
            out_folder_semantics + str(SEQUENCE_NUM) + "_" + str(cur_idx) + "/"
        )
        
        out_folder_instances_cur = (
            out_folder_instances + str(SEQUENCE_NUM) + "_" + str(cur_idx) + "/"
        )

        create_folder(out_folder_ncuts_cur)
        if config["gt"]:
            create_folder(out_folder_semantics_cur)
            create_folder(out_folder_instances_cur)

        instances = np.hstack(
            (
                kitti_labels_orig["instance_nonground"].reshape(
                    -1,
                ),
                kitti_labels_orig["instance_ground"].reshape(
                    -1,
                ),
            )
        )
        patchwise_indices = indices_per_patch(
            T_pcd,
            center_positions,
            positions,
            first_position,
            sampled_indices_global,
            chunk_size,
        )
        out_data = []
        semantics_kitti = []
        for sequence in tqdm(range(start_seq, len(center_ids))):

            # try :
            print("sequence", sequence)
            if (
                config["name"] not in ["sam3d", "sam3d_fix", "3duis", "3duis_fix"]
                and "maskpls" not in config["name"]
            ):
                (
                    merged_chunk,
                    file_name,
                    pcd_chunk,
                    pcd_chunk_ground,
                    inliers,
                    inliers_ground,
                ) = ncuts_chunk(
                    dataset,
                    list(indices),
                    pcd_nonground_chunks,
                    pcd_ground_chunks,
                    pcd_nonground_chunks_major_downsampling,
                    pcd_nonground_minor,
                    T_pcd,
                    center_positions,
                    center_ids,
                    positions,
                    first_position,
                    list(sampled_indices_global),
                    chunk_size=chunk_size,
                    major_voxel_size=major_voxel_size,
                    cam_ids=[0],
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    theta=theta,
                    proximity_threshold=proximity_threshold,
                    out_folder=out_folder_ncuts,
                    ground_mode=False,
                    sequence=sequence,
                    patchwise_indices=patchwise_indices,
                    ncuts_threshold=ncuts_threshold,
                    mean_height=0.6,
                )
                name = file_name.split("/")[-1]

                cur_name = name.split(".")[0]
                pred_pcd = pcd_chunk + pcd_chunk_ground

                if config["gt"]:
                    inst_ground = kitti_labels["ground"]["instance"][sequence][inliers][
                        inliers_ground
                    ]
                    seg_ground = kitti_labels["ground"]["semantic"][sequence][inliers][
                        inliers_ground
                    ]

                    kitti_chunk_instance = color_pcd_by_labels(
                        copy.deepcopy(pcd_chunk),
                        kitti_labels["nonground"]["instance"][sequence].reshape(
                            -1,
                        ),
                        colors=colors,
                        gt_labels=kitti_labels_orig["instance_nonground"],
                    )
                    kitti_chunk_instance_ground = color_pcd_by_labels(
                        copy.deepcopy(pcd_chunk_ground),
                        inst_ground.reshape(
                            -1,
                        ),
                        colors=colors,
                        gt_labels=instances,
                    )
                    gt_pcd = kitti_chunk_instance + kitti_chunk_instance_ground
                    semantics_non_ground = color_pcd_kitti(
                        copy.deepcopy(pcd_chunk),
                        kitti_labels["nonground"]["semantic"][sequence].reshape(
                            -1,
                        ),
                    )
                    semantics_ground = color_pcd_kitti(
                        copy.deepcopy(pcd_chunk_ground),
                        seg_ground.reshape(
                            -1,
                        ),
                    )
                    kitti_semantics = np.hstack(
                        (
                            kitti_labels["nonground"]["semantic"][sequence].reshape(
                                -1,
                            ),
                            seg_ground.reshape(
                                -1,
                            ),
                        )
                    )
                    unique_colors, labels_kitti = np.unique(
                        np.asarray(gt_pcd.colors), axis=0, return_inverse=True
                    )

                    pts = np.asarray(gt_pcd.points)

                    instance_pcd = kitti_chunk_instance + kitti_chunk_instance_ground
                    semantics_pcd = semantics_non_ground + semantics_ground
                    # points,labels_ncuts,labels_kitti, downsampled_semantics = downsample_chunk_data(pts,labels_ncuts,labels_kitti,kitti_semantics)
                    print("output", data_store_folder + name.split(".")[0])

                    # if labels_ncuts.shape[0] != points.shape[0] != downsampled_semantics.shape[0] != labels_kitti.shape[0]:
                    #        AssertionError
\
                    o3d.io.write_point_cloud(
                        out_folder_semantics_cur + name,
                        semantics_pcd,
                        write_ascii=False,
                        compressed=False,
                        print_progress=False,
                    )
                    
                    o3d.io.write_point_cloud(
                        out_folder_instances_cur + name,
                        instance_pcd,
                        write_ascii=False,
                        compressed=False,
                        print_progress=False,
                    )


            elif "maskpls" in config["name"]:
                inliers = get_statistical_inlier_indices(pcd_ground_chunks[sequence])
                ground_inliers = get_subpcd(pcd_ground_chunks[sequence], inliers)
                mean_hight = np.mean(np.asarray(ground_inliers.points)[:, 2])
                in_idcs = np.where(
                    np.asarray(ground_inliers.points)[:, 2] < (mean_hight + 0.6)
                )[0]
                pcd_chunk_ground = get_subpcd(ground_inliers, in_idcs)
                pcd_chunk_ground.paint_uniform_color([0, 0, 0])

                input_pcd = pcd_nonground_chunks[sequence] + pcd_chunk_ground

                if "supervised" not in config["name"]:
                    print("unsupervised")
                    pred_pcd = maskpls.forward_and_project(input_pcd)
                else:
                    pred_pcd = maskpls.forward_and_project(input_pcd)
                # o3d.visualization.draw_geometries([pred_pcd])

            name = str(center_ids[sequence]).zfill(6) + ".pcd"

            unique_colors, labels_ncuts = np.unique(
                np.asarray(pred_pcd.colors), axis=0, return_inverse=True
            )

            o3d.io.write_point_cloud(
                out_folder_ncuts_cur + name,
                pred_pcd,
                write_ascii=False,
                compressed=False,
                print_progress=False,
            )
            
            

            gc.collect()

        merge_ncuts = merge_chunks_unite_instances2(
                get_merge_pcds(out_folder_ncuts_cur[:-1])
            )


        if config["gt"]:

            kitti_pcd = merge_unite_gt(get_merge_pcds(out_folder_semantics_cur[:-1]))
            map_instances = merge_unite_gt(
                get_merge_pcds(out_folder_instances_cur[:-1])
            )
            _, labels_semantics = np.unique(
                np.asarray(kitti_pcd.colors), axis=0, return_inverse=True
            )
            
            _, labels_instances = np.unique(
                np.asarray(map_instances.colors), axis=0, return_inverse=True
            )
            
            updated_labels_semantics = np.zeros(np.asarray(kitti_pcd.colors).shape[0])
            for label in np.unique(labels_semantics):
                idcs = np.where(labels_semantics == label)[0]
                col_cur = np.asarray(kitti_pcd.colors)[idcs][0]
                updated_labels_semantics[idcs] = reverse_color_dict[tuple(col_cur)]

            np.savez(
                data_store_folder
                + "kitti_semantic"
                + str(SEQUENCE_NUM)
                + "_"
                + str(cur_idx)
                + ".npz",
                labels=updated_labels_semantics,
            )

        if "maskpls" in config["name"]:

            with open(
                data_store_folder
                + config["name"]
                + "_confs"
                + str(SEQUENCE_NUM)
                + "_"
                + str(cur_idx)
                + ".json",
                "w",
            ) as fp:
                json.dump(maskpls.confs_dict, fp)
        
        
        
        new_labels_inst = labels_instances + (
                        updated_labels_semantics * np.unique(labels_instances).shape[0]
                    )
        
        
        metrics = Metrics(config['name'] + ' ' + str(seq))
        colors, labels_ncuts_all = np.unique(
            np.asarray(merge_ncuts.colors), axis=0, return_inverse=True
        )
        
        instance_preds = remove_semantics(labels_instances, labels_ncuts_all)
        
        out, aps_lstq_dict = metrics.update_stats(
                labels_ncuts_all,
                instance_preds,
                new_labels_inst,
        )
        
        o3d.io.write_point_cloud(
            data_store_folder
            + config["name"]
            + str(SEQUENCE_NUM)
            + "_"
            + str(cur_idx)
            + ".pcd",
            merge_ncuts,
            write_ascii=False,
            compressed=False,
            print_progress=False,
        )
        
        o3d.io.write_point_cloud(
            data_store_folder
            + "_instances_"
            + str(SEQUENCE_NUM)
            + "_"
            + str(cur_idx)
            + ".pcd",
            map_instances,
            write_ascii=False,
            compressed=False,
            print_progress=False,
        )
        
        
