import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

src_path = os.path.abspath("../..")
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
import hdbscan
from predict_maskpls import RefinerModel
from point_cloud_utils import *
from visualization_utils import *
from sam3d_util import *
from predict_maskpls import RefinerModel


config_tarl_spatial_dino = {
    "name": "spatial_1.0_tarl_0.5_dino_0.1_t_0.005",
    "out_folder": "ncuts_data_tarl_dino_spatial/",
    "gamma": 0.1,
    "alpha": 1.0,
    "theta": 0.5,
    "T": 0.005,
    "gt": False,
}

config_tarl_spatial = {
    "name": "spatial_1.0_tarl_0.5_t_0.03",
    "out_folder": "ncuts_data_tarl_spatial/",
    "gamma": 0.0,
    "alpha": 1.0,
    "theta": 0.5,
    "T": 0.03,
    "gt": False,
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

config_sam3d = {
    "name": "sam3d_fix",
    "out_folder": "ncuts_data_sam3d/",
    "gamma": 0.0,
    "alpha": 0.0,
    "theta": 0.0,
    "T": 0.0,
    "gt": False,
}

config_3duis = {
    "name": "3duis_fix",
    "out_folder": "ncuts_data_3duis_fixed/",
    "gamma": 0.0,
    "alpha": 0.0,
    "theta": 0.0,
    "T": 0.0,
    "gt": False,
}

config_maskpls = {
    "name": "maskpls_hdbscan",
    "out_folder": "refine_hdbscan/",
    "gamma": 0.0,
    "alpha": 0.0,
    "theta": 0.0,
    "T": 0.0,
    "gt": False,
}

start_chunk = 0
start_seq = 0
seqs = list(range(0, 11))
config = config_sam3d
if "3duis" in config["name"]:
    from utils_3duis import UIS3D_clustering

if "maskpls" in config["name"]:
    maskpls = RefinerModel(dataset="kitti")

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


def clustering_logic(pcd_nonground_corrected, method="hdbscan"):
    """
    Perform DBSCAN clustering on the point cloud data.

    :param cur_pcd: Current point cloud for clustering.
    :param pcd_all: All point cloud data.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: Cluster labels for each point in the point cloud.
    """

    pcd_nonground_downsampled = o3d.geometry.PointCloud()
    pts_downsampled = downsample_chunk(np.asarray(pcd_nonground_corrected.points))
    pcd_nonground_downsampled.points = o3d.utility.Vector3dVector(pts_downsampled)

    clustering = hdbscan.HDBSCAN(
        algorithm="best",
        alpha=1.0,
        approx_min_span_tree=True,
        gen_min_span_tree=True,
        leaf_size=100,
        metric="euclidean",
        min_cluster_size=10,
        min_samples=None,
    )
    clustering.fit(pts_downsampled)

    labels_not_road = clustering.labels_
    colors_gen = generate_random_colors(5000)

    labels_not_road = labels_not_road + 1
    # Reproject cluster labels to the original point cloud size
    cluster_labels = np.ones((len(pcd_nonground_corrected.points), 1)) * -1
    labels_non_ground = kDTree_1NN_feature_reprojection(
        cluster_labels,
        pcd_nonground_corrected,
        labels_not_road.reshape(-1, 1),
        pcd_nonground_downsampled,
    )
    colors = np.zeros((labels_non_ground.shape[0], 3))
    unique_labels = list(np.unique(labels_non_ground))

    for j in unique_labels:
        cur_idcs = np.where(labels_non_ground == j)[0]
        if j == 0:
            colors[cur_idcs] = np.array([0, 0, 0])
        else:
            colors[cur_idcs] = np.array(colors_gen[unique_labels.index(j)])

    pcd_nonground_corrected.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # o3d.visualization.draw_geometries([pcd_nonground_corrected])

    return pcd_nonground_corrected


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


DATASET_PATH = os.path.join("/media/cedric/Datasets2/semantic_kitti/")
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

out_folder_dbscan = out_folder + "dbscan_data/"
if os.path.exists(out_folder_dbscan) == False:
    os.makedirs(out_folder_dbscan)

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
        out_folder_dbscan_cur = (
            out_folder_dbscan + str(SEQUENCE_NUM) + "_" + str(cur_idx) + "/"
        )
        out_folder_instances_cur = (
            out_folder_instances + str(SEQUENCE_NUM) + "_" + str(cur_idx) + "/"
        )

        create_folder(out_folder_ncuts_cur)
        if config["gt"]:
            create_folder(out_folder_dbscan_cur)
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
                    pcd_nonground_hdbscan = clustering_logic(copy.deepcopy(pcd_chunk))
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
                    cluster_pcd = pcd_nonground_hdbscan + pcd_chunk_ground
                    # points,labels_ncuts,labels_kitti, downsampled_semantics = downsample_chunk_data(pts,labels_ncuts,labels_kitti,kitti_semantics)
                    print("output", data_store_folder + name.split(".")[0])

                    # if labels_ncuts.shape[0] != points.shape[0] != downsampled_semantics.shape[0] != labels_kitti.shape[0]:
                    #        AssertionError

                    o3d.io.write_point_cloud(
                        out_folder_dbscan_cur + name,
                        cluster_pcd,
                        write_ascii=False,
                        compressed=False,
                        print_progress=False,
                    )
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

            elif "sam3d" in config["name"]:
                inliers = get_statistical_inlier_indices(pcd_ground_chunks[sequence])
                ground_inliers = get_subpcd(pcd_ground_chunks[sequence], inliers)
                mean_hight = np.mean(np.asarray(ground_inliers.points)[:, 2])
                in_idcs = np.where(
                    np.asarray(ground_inliers.points)[:, 2] < (mean_hight + 0.6)
                )[0]
                pcd_chunk_ground = get_subpcd(ground_inliers, in_idcs)
                pcd_chunk_ground.paint_uniform_color([0, 0, 0])
                # try :
                pcd_chunk = sam3d(
                    dataset,
                    indices,
                    pcd_nonground_minor,
                    T_pcd,
                    sampled_indices_global,
                    patchwise_indices=patchwise_indices,
                    sequence=sequence,
                    pcd_chunk=pcd_nonground_chunks[sequence],
                )
                # except:

                # pcd_nonground_chunks[sequence].paint_uniform_color([0,0,0])
                pred_pcd = pcd_chunk + pcd_chunk_ground

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

                pred_pcd = maskpls.forward_and_project(input_pcd)

            else:  # '3duis'
                pcd_3duis = UIS3D_clustering(
                    pcd_nonground_chunks[sequence],
                    pcd_ground_chunks[sequence],
                    center_ids[sequence],
                    center_positions[sequence],
                    eps=0.4,
                    min_samples=10,
                )
                pred_pcd = pcd_3duis

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
            merge_dbscan = merge_chunks_unite_instances2(
                get_merge_pcds(out_folder_dbscan_cur[:-1])
            )
            kitti_pcd = merge_unite_gt(get_merge_pcds(out_folder_semantics_cur[:-1]))
            map_instances = merge_unite_gt(
                get_merge_pcds(out_folder_instances_cur[:-1])
            )
            _, labels_kitti_cur = np.unique(
                np.asarray(kitti_pcd.colors), axis=0, return_inverse=True
            )
            updated_labels = np.zeros(np.asarray(kitti_pcd.colors).shape[0])
            for label in np.unique(labels_kitti_cur):
                idcs = np.where(labels_kitti_cur == label)[0]
                col_cur = np.asarray(kitti_pcd.colors)[idcs][0]
                updated_labels[idcs] = reverse_color_dict[tuple(col_cur)]

            o3d.io.write_point_cloud(
                data_store_folder
                + "hdbscan"
                + str(SEQUENCE_NUM)
                + "_"
                + str(cur_idx)
                + ".pcd",
                merge_dbscan,
                write_ascii=False,
                compressed=False,
                print_progress=False,
            )
            np.savez(
                data_store_folder
                + "kitti_semantic"
                + str(SEQUENCE_NUM)
                + "_"
                + str(cur_idx)
                + ".npz",
                labels=updated_labels,
            )
            o3d.io.write_point_cloud(
                data_store_folder
                + "kitti_instances"
                + str(SEQUENCE_NUM)
                + "_"
                + str(cur_idx)
                + ".pcd",
                map_instances,
                write_ascii=False,
                compressed=False,
                print_progress=False,
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
