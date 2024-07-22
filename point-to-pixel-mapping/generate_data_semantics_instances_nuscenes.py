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
import gc
import scipy
from scipy.spatial.distance import cdist
from normalized_cut import normalized_cut
from ncuts_utils import (
    ncuts_chunk,
    kDTree_1NN_feature_reprojection_colors,
    get_merge_pcds,
)
from collections import Counter
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
from metrics.metrics_class import Metrics
from point_cloud_utils import *
from visualization_utils import *
from predict_maskpls import RefinerModel
import shutil


config_tarl_spatial_dino = {
    "name": "spatial_0.1_tarl_0.1_dino_0.1_t_0.03",
    "out_folder": "ncuts_data_tarl_dino_spatial_updated/",
    "gamma": 0.1,
    "alpha": 0.1,
    "theta": 0.1,
    "T": 0.03,
    "gt": True,
}

config_tarl_spatial = {
    "name": "spatial_1.0_tarl_0.5_t_0.03",
    "out_folder": "ncuts_data_tarl_spatial/",
    "gamma": 0.0,
    "alpha": 0.1,
    "theta": 0.1,
    "T": 0.03,
    "gt": True,
}

config_maskpls = {
    "name": "maskpls_",
    "out_folder": "ncuts_refined/",
    "gamma": 0.0,
    "alpha": 1.0,
    "theta": 0.5,
    "T": 0.03,
    "gt": False,
}


config_spatial = {
    "name": "spatial_0.1",
    "out_folder": "ncuts_data_spatial/",
    "gamma": 0.0,
    "alpha": 2.0,
    "theta": 0.0,
    "T": 0.02,
    "gt": True,
}

config_dino_spatial = {
    "name": "dino.1",
    "out_folder": "ncuts_data_dino/",
    "gamma": 0.1,
    "alpha": 0.1,
    "theta": 0.0,
    "T": 0.1,
    "gt": True,
}

config_maskpls = {
    "name": "maskpls_nuscenes_",
    "out_folder": "ncuts_data_maskpls_refined/",
    "gamma": 0.0,
    "alpha": 0.0,
    "theta": 0.0,
    "T": 0.0,
    "gt": True,
}


config = config_tarl_spatial
config["data_gen"] = True  ##for storing training refinement data
if "maskpls" in config["name"]:
    maskpls = RefinerModel(dataset="nuscenes")

# seq_limit = 5

print(config)
alpha = config["alpha"]
theta = config["theta"]
beta = 0.0
gamma = config["gamma"]
proximity_threshold = 1.0
tarl_norm = True
colors = generate_random_colors_map(6000)
ncuts_threshold = config["T"]

cams = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
cam_ids = [0]


dist_threshold = 5  # moving object filter threshold

minor_voxel_size = 0.05
major_voxel_size = 0.35
chunk_size = np.array([28, 10, 15])  # meters
overlap = 3  # meters
ground_segmentation_method = "patchwork"
min_pts = 20
NCUT_ground = False

sem_classes = [9, 10, 28, 30]
ignore_labels = [0, 24, 25, 26, 27, 31]

# dataset_type = "v1.0-mini"
dataset_type = "v1.0-trainval"

DATASET_PATH = "/media/cedric/Datasets21/nuScenes_mini/nuScenes/"

# seqs = list(range(0, 10))  ## only used for mini dataset
# seq_limit = 10  ## only use first two scenes
vis = False  ##flag for map vis after iteration
seq_limit = 5
seqs_trainval = [
    "scene-0003",
    "scene-0012",
    "scene-0013",
    "scene-0014",
    "scene-0015",
    "scene-0016",
    "scene-0017",
    "scene-0018",
    "scene-0035",
    "scene-0036",
    "scene-0038",
    "scene-0039",
    "scene-0092",
    "scene-0093",
    "scene-0094",
    "scene-0095",
    "scene-0096",
    "scene-0097",
    "scene-0098",
    "scene-0099",
    "scene-0100",
    "scene-0101",
    "scene-0102",
    "scene-0103",
    "scene-0104",
    "scene-0105",
    "scene-0106",
    "scene-0107",
    "scene-0108",
    "scene-0109",
    "scene-0110",
    "scene-0221",
    "scene-0268",
    "scene-0269",
    "scene-0270",
    "scene-0271",
    "scene-0272",
    "scene-0273",
    "scene-0274",
    "scene-0275",
    "scene-0276",
    "scene-0277",
    "scene-0278",
    "scene-0329",
    "scene-0330",
    "scene-0331",
    "scene-0332",
    "scene-0344",
    "scene-0345",
    "scene-0346",
    "scene-0519",
    "scene-0520",
    "scene-0521",
    "scene-0522",
    "scene-0523",
    "scene-0524",
    "scene-0552",
    "scene-0553",
    "scene-0554",
    "scene-0555",
    "scene-0556",
    "scene-0557",
    "scene-0558",
    "scene-0559",
    "scene-0560",
    "scene-0561",
    "scene-0562",
    "scene-0563",
    "scene-0564",
    "scene-0565",
    "scene-0625",
    "scene-0626",
    "scene-0627",
    "scene-0629",
    "scene-0630",
    "scene-0632",
    "scene-0633",
    "scene-0634",
    "scene-0635",
    "scene-0636",
    "scene-0637",
    "scene-0638",
    "scene-0770",
    "scene-0771",
    "scene-0775",
    "scene-0777",
    "scene-0778",
    "scene-0780",
    "scene-0781",
    "scene-0782",
    "scene-0783",
    "scene-0784",
    "scene-0794",
    "scene-0795",
    "scene-0796",
    "scene-0797",
    "scene-0798",
    "scene-0799",
    "scene-0800",
    "scene-0802",
    "scene-0904",
    "scene-0905",
    "scene-0906",
    "scene-0907",
    "scene-0908",
    "scene-0909",
    "scene-0910",
    "scene-0911",
    "scene-0912",
    "scene-0913",
    "scene-0914",
    "scene-0915",
    "scene-0916",
    "scene-0917",
    "scene-0919",
    "scene-0920",
    "scene-0921",
    "scene-0922",
    "scene-0923",
    "scene-0924",
    "scene-0925",
    "scene-0926",
    "scene-0927",
    "scene-0928",
    "scene-0929",
    "scene-0930",
    "scene-0931",
    "scene-0962",
    "scene-0963",
    "scene-0966",
    "scene-0967",
    "scene-0968",
    "scene-0969",
    "scene-0971",
    "scene-0972",
    "scene-1059",
    "scene-1060",
    "scene-1061",
    "scene-1062",
    "scene-1063",
    "scene-1064",
    "scene-1065",
    "scene-1066",
    "scene-1067",
    "scene-1068",
    "scene-1069",
    "scene-1070",
    "scene-1071",
    "scene-1072",
    "scene-1073",
]


if (
    dataset_type == "v1.0-trainval"
):  ##this script can be used for full eval with path change
    DATASET_PATH = "/media/cedric/Datasets21/nuScenes_train/"
    seqs = seqs_trainval


out_folder = "/media/cedric/Datasets21/pcd_preprocessed_nuscenes/" + dataset_type + "/"


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


def calc_num_instance(sem_classes, labels_instances, labels_sem):
    num_inst = len(np.unique(labels_instances)) - 1

    num_sem_inst = len(np.unique(labels_sem[np.isin(labels_sem, sem_classes)]))

    num_whole_inst = num_inst + num_sem_inst
    return num_whole_inst, labels_sem


def get_semantic_map(inst_labels, sem_labels, sem_classes=[49, 50, 51, 70, 72]):

    semantic_map = {}
    total_merges = 0

    inst_labels = inst_labels[np.isin(sem_labels, sem_classes)]
    sem_labels = sem_labels[np.isin(sem_labels, sem_classes)]
    for inst in np.unique(inst_labels)[1:]:
        intersect, area = np.unique(sem_labels[inst_labels == inst], return_counts=True)
        main_sem_label = intersect[np.argmax(area)]
        semantic_map[inst] = main_sem_label
        to_merge_sem_labels = intersect[intersect != main_sem_label]
        num_merges = len(to_merge_sem_labels)
        total_merges += num_merges
    return semantic_map


def get_num_splits(intersection, sem_classes, t_area, t_score, t_ratio):
    result = {
        "inst": [],
        "area": [],
        "score": [],
        "sem_labels": [],
        "num_splits": [],
        "sem_splits": [],
        "main_sem_label": [],
        "other_splits": [],
        "valid_sem_label": [],
        "sem-sem_splits": [],
    }
    for idx, inst in enumerate(intersection["inst"]):
        # print(f"inst: {inst}, area: {intersection['area'][inst]}, score: {intersection['score'][inst]}, sem_labels: {intersection['sem_labels'][inst]}, num_intersection: {intersection['num_intersection'][inst]}")
        area = np.sum(intersection["area"][idx])
        score = intersection["score"][idx]
        sem_label = intersection["sem_labels"][idx]
        if area > t_area and score < t_score and inst != 0:
            intersect_area = intersection["area"][idx]
            num_splits = np.count_nonzero(intersect_area / area > t_ratio) - 1
            valid_sem_label = sem_label[intersect_area / area > t_ratio]
            sem_sem_splits = np.count_nonzero(np.isin(valid_sem_label, sem_classes)) - 1
            other_splits = num_splits - sem_sem_splits
            result["inst"].append(inst)
            result["area"].append(area)
            result["score"].append(score)
            result["sem_labels"].append(sem_label)
            result["num_splits"].append(num_splits)
            result["sem_splits"].append(sem_label[intersect_area / area > t_ratio])
            result["main_sem_label"].append(sem_label[np.argmax(intersect_area)])
            result["other_splits"].append(other_splits)
            result["valid_sem_label"].append(valid_sem_label)
            result["sem-sem_splits"].append(sem_sem_splits)
        num_splits = np.sum(result["num_splits"])
        sem2sem_splits = np.sum(result["sem-sem_splits"])
        # print(f"inst: {inst}, area: {area}, score: {score}, sem_labels: {intersection['sem_labels'][idx]}, num_intersection: {intersection['num_intersection'][idx]}, num_splits: {num_splits}")
    return result, (num_splits, sem2sem_splits)


def create_sub_instances(inst_labels, sem_labels, result):
    new_inst_labels = inst_labels.copy()
    last_inst_id = np.max(inst_labels)
    for inst in np.unique(inst_labels):

        if inst in result["inst"]:
            idx = np.where(result["inst"] == inst)[0][0]
            sem_splits = result["sem_splits"][idx]
            main_sem_label = result["main_sem_label"][idx]
            to_split = sem_splits[sem_splits != main_sem_label]
            for split in to_split:
                last_inst_id += 1
                # print(f"Splitting instance {inst} with sem label {split} into {last_inst_id}")
                new_inst_labels[
                    np.logical_and(inst_labels == inst, sem_labels == split)
                ] = last_inst_id
    return new_inst_labels


def get_intersection(labels_sem, pred_labels, ignore_labels=[0, 1, 52, 99, 40, 48, 49]):
    intersection = {
        "inst": [],
        "sem_labels": [],
        "area": [],
        "num_intersection": [],
        "score": [],
    }
    for i in ignore_labels:
        labels_sem[labels_sem == i] = 0
    pred_labels = pred_labels[labels_sem != 0]
    labels_sem = labels_sem[labels_sem != 0]
    for label in np.unique(pred_labels):
        intersect, area = np.unique(
            labels_sem[pred_labels == label], return_counts=True
        )
        intersection["inst"].append(label)
        intersection["sem_labels"].append(intersect)
        intersection["area"].append(area)
        intersection["num_intersection"].append(len(intersect))
        max_area = np.max(area)
        # error_area = np.sum(area[area != max_area])
        score = max_area / np.sum(area)
        intersection["score"].append(score)
    return intersection


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

    if os.path.exists(name) == True:
        shutil.rmtree(name)
    os.makedirs(name)


def divide_indices_into_chunks(max_index, chunk_size=1000):
    chunks = []
    for start in range(0, max_index, chunk_size):
        end = min(start + chunk_size, max_index)
        chunks.append((start, end))
    return chunks


color_dict_normalized = {
    0: [0 / 255, 0 / 255, 0 / 255],
    1: [0 / 255, 0 / 255, 255 / 255],
    2: [245 / 255, 150 / 255, 100 / 255],
    3: [245 / 255, 230 / 255, 100 / 255],
    4: [250 / 255, 80 / 255, 100 / 255],
    5: [150 / 255, 60 / 255, 30 / 255],
    6: [255 / 255, 0 / 255, 0 / 255],
    7: [180 / 255, 30 / 255, 80 / 255],
    8: [255 / 255, 0 / 255, 0 / 255],
    9: [30 / 255, 30 / 255, 255 / 255],
    10: [200 / 255, 40 / 255, 255 / 255],
    11: [90 / 255, 30 / 255, 150 / 255],
    12: [255 / 255, 0 / 255, 255 / 255],
    13: [255 / 255, 150 / 255, 255 / 255],
    14: [75 / 255, 0 / 255, 75 / 255],
    15: [75 / 255, 0 / 255, 175 / 255],
    16: [0 / 255, 200 / 255, 255 / 255],
    17: [50 / 255, 120 / 255, 255 / 255],
    18: [0 / 255, 150 / 255, 255 / 255],
    19: [170 / 255, 255 / 255, 150 / 255],
    20: [0 / 255, 175 / 255, 0 / 255],
    21: [0 / 255, 60 / 255, 135 / 255],
    22: [80 / 255, 240 / 255, 150 / 255],
    23: [150 / 255, 240 / 255, 255 / 255],
    24: [0 / 255, 0 / 255, 255 / 255],
    25: [255 / 255, 255 / 255, 50 / 255],
    26: [245 / 255, 150 / 255, 100 / 255],
    27: [255 / 255, 0 / 255, 0 / 255],
    28: [200 / 255, 40 / 255, 255 / 255],
    29: [30 / 255, 30 / 255, 255 / 255],
    30: [90 / 255, 30 / 255, 150 / 255],
    31: [250 / 255, 80 / 255, 100 / 255],
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

data_store_train = out_folder + "train_" + config["name"] + "/"
if os.path.exists(data_store_train) == False:
    os.makedirs(data_store_train)


metrics_clustering = Metrics(name="hdbscan", min_points=min_pts)
metrics_ncuts = Metrics(name="ncuts", min_points=min_pts)
cnt = 0

for seq in tqdm(seqs):
    print("Sequence", seq)
    SEQUENCE_NUM = seq
    dataset = create_nuscenes_odometry_dataset(
        DATASET_PATH,
        seq,
        ncuts_mode=True,
        sam_folder_name="SAM",
        dinov2_folder_name="Dinov2",
        dist_threshold=dist_threshold,
        dataset_type=dataset_type,
        scene=seq,
    )

    chunks_idcs = divide_indices_into_chunks(len(dataset))

    data_store_folder = out_folder + str(SEQUENCE_NUM) + "/"
    if os.path.exists(data_store_folder) == False:
        os.makedirs(data_store_folder)

    data_store_folder_train_cur = data_store_train + str(SEQUENCE_NUM) + "/"
    if os.path.exists(data_store_folder_train_cur) == False:
        os.makedirs(data_store_folder_train_cur)

    print("STORE FOLDER", data_store_folder)

    for cur_idx, cidcs in enumerate([0]):
        # if cur_idx == 0 :
        #         continue

        ind_start = 0
        ind_end = len(dataset)

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
            # if True == True:  ##use this to allow variable chunk generation changes
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
            obbs,
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
        for sequence in range(0, len(center_ids)):

            # try:
            print("sequence", sequence)
            if "maskpls" not in config["name"]:
                (
                    merged_chunk,
                    file_name,
                    pcd_chunk,
                    pcd_chunk_ground,
                    inliers,
                    inliers_ground,
                ) = ncuts_chunk(
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
                    chunk_size=chunk_size,
                    major_voxel_size=major_voxel_size,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    theta=theta,
                    proximity_threshold=proximity_threshold,
                    ncuts_threshold=ncuts_threshold,
                    cams=cams,
                    cam_ids=cam_ids,
                    out_folder="",
                    ground_mode=False,
                    sequence=sequence,
                    patchwise_indices=patchwise_indices,
                    adjacent_frames_cam=(4, 5),
                    adjacent_frames_tarl=(5, 5),
                    norm=tarl_norm,
                    obb=obbs[sequence],
                )

                pred_pcd = pcd_chunk + pcd_chunk_ground
                inst_ground = kitti_labels["ground"]["instance"][sequence][inliers][
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
                name = str(center_ids[sequence]).zfill(6) + ".pcd"
                instance_pcd = kitti_chunk_instance + kitti_chunk_instance_ground
                o3d.io.write_point_cloud(
                    out_folder_instances_cur + name,
                    instance_pcd,
                    write_ascii=False,
                    compressed=False,
                    print_progress=False,
                )

                unique_colors, labels_ncuts = np.unique(
                    np.asarray(pred_pcd.colors), axis=0, return_inverse=True
                )

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
                    gt_pcd = kitti_chunk_instance + kitti_chunk_instance_ground

                    unique_colors, labels_kitti = np.unique(
                        np.asarray(gt_pcd.colors), axis=0, return_inverse=True
                    )
                    # o3d.visualization.draw_geometries([gt_pcd])
                    semantics_pcd = semantics_non_ground + semantics_ground
                    name = str(center_ids[sequence]).zfill(6) + ".pcd"
                    o3d.io.write_point_cloud(
                        out_folder_semantics_cur + name,
                        semantics_pcd,
                        write_ascii=False,
                        compressed=False,
                        print_progress=False,
                    )

            else:
                pcd_chunk = pcd_nonground_chunks[sequence]
                pcd_ground_chunk = pcd_ground_chunks[sequence]
                inliers = get_statistical_inlier_indices(pcd_ground_chunk)
                ground_inliers = get_subpcd(pcd_ground_chunk, inliers)
                mean_hight = np.mean(np.asarray(ground_inliers.points)[:, 2])
                inliers_ground = np.where(
                    np.asarray(ground_inliers.points)[:, 2] < (mean_hight + 0.2)
                )[0]
                pcd_chunk_ground = get_subpcd(ground_inliers, inliers_ground)
                pcd_chunk_ground.paint_uniform_color([0, 0, 0])
                merged_chunk = pcd_chunk + pcd_chunk_ground
                pred_pcd = maskpls.forward_and_project(merged_chunk)
                unique_colors, labels_ncuts = np.unique(
                    np.asarray(pred_pcd.colors), axis=0, return_inverse=True
                )
                # o3d.visualization.draw_geometries([pred_pcd])
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
                name = str(center_ids[sequence]).zfill(6) + ".pcd"
                instance_pcd = kitti_chunk_instance + kitti_chunk_instance_ground
                o3d.io.write_point_cloud(
                    out_folder_instances_cur + name,
                    instance_pcd,
                    write_ascii=False,
                    compressed=False,
                    print_progress=False,
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
                gt_pcd = kitti_chunk_instance + kitti_chunk_instance_ground

                unique_colors, labels_kitti = np.unique(
                    np.asarray(gt_pcd.colors), axis=0, return_inverse=True
                )

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

                # o3d.visualization.draw_geometries([gt_pcd])
                semantics_pcd = semantics_non_ground + semantics_ground
                name = str(center_ids[sequence]).zfill(6) + ".pcd"
                o3d.io.write_point_cloud(
                    out_folder_semantics_cur + name,
                    semantics_pcd,
                    write_ascii=False,
                    compressed=False,
                    print_progress=False,
                )

            pcd_nonground_hdbscan = clustering_logic(copy.deepcopy(pcd_chunk))
            cluster_pcd = pcd_nonground_hdbscan + pcd_chunk_ground
            print(pred_pcd)
            print(instance_pcd)
            print(cluster_pcd)

            name = str(center_ids[sequence]).zfill(6) + ".pcd"

            pts = np.asarray(pred_pcd.points)
            if config["data_gen"] == True:
                points, labels_ncuts, labels_kitti, downsampled_semantics = (
                    downsample_chunk_data(
                        pts, labels_ncuts, labels_kitti, kitti_semantics
                    )
                )
                print("output", data_store_folder + name.split(".")[0])

                if (
                    labels_ncuts.shape[0]
                    != points.shape[0]
                    != downsampled_semantics.shape[0]
                    != labels_kitti.shape[0]
                ):
                    AssertionError

            np.savez(
                data_store_folder_train_cur
                + name.split(".")[0]
                + str(cur_idx)
                + ".npz",
                pts=points,
                ncut_labels=labels_ncuts,
                kitti_labels=labels_kitti,
                cluster_labels=np.zeros_like(labels_ncuts),
                semantic=downsampled_semantics,
            )

            o3d.io.write_point_cloud(
                out_folder_ncuts_cur + name,
                pred_pcd,
                write_ascii=False,
                compressed=False,
                print_progress=False,
            )
            o3d.io.write_point_cloud(
                out_folder_dbscan_cur + name,
                cluster_pcd,
                write_ascii=False,
                compressed=False,
                print_progress=False,
            )

            del pred_pcd, pcd_chunk, pcd_chunk_ground
            gc.collect()

            # except:
            # cnt += 1
            # print("Skip due to convergence issue ", cnt)

        pcds_clustering = get_merge_pcds(out_folder_dbscan_cur)
        if len(pcds_clustering) == 0:
            continue

        merge_hdbscan = merge_chunks_unite_instances(
            get_merge_pcds(out_folder_dbscan_cur)
        )
        merge_ncuts = merge_chunks_unite_instances(get_merge_pcds(out_folder_ncuts_cur))

        map_instances = merge_unite_gt(get_merge_pcds(out_folder_instances_cur))
        _, labels_nuscenes = np.unique(
            np.asarray(map_instances.colors), axis=0, return_inverse=True
        )

        # o3d.visualization.draw_geometries(
        #    [
        #        merge_ncuts,
        #        map_instances.translate([0, 50, 0]),
        #        merge_hdbscan.translate([0, 100, 0]),
        #    ]
        # )
        # o3d.visualization.draw_geometries([kitti_pcd])

        if config["gt"]:
            kitti_pcd = merge_unite_gt(get_merge_pcds(out_folder_semantics_cur))
            _, labels_kitti_cur = np.unique(
                np.asarray(kitti_pcd.colors), axis=0, return_inverse=True
            )
            updated_labels = np.zeros(np.asarray(kitti_pcd.colors).shape[0])
            for label in np.unique(labels_kitti_cur):
                idcs = np.where(labels_kitti_cur == label)[0]
                # import pdb; pdb.set_trace()
                col_cur = np.asarray(kitti_pcd.colors)[idcs][0]

                updated_labels[idcs] = reverse_color_dict[tuple(col_cur)]

            np.savez(
                data_store_folder
                + "nuscenes_semantic_seq_"
                + str(SEQUENCE_NUM)
                + "_"
                + str(cur_idx)
                + ".npz",
                labels=updated_labels,
            )
            o3d.io.write_point_cloud(
                data_store_folder
                + "nuscenes_instances_seq_"
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
                + "hdbscan_seq_"
                + str(SEQUENCE_NUM)
                + "_"
                + str(cur_idx)
                + ".pcd",
                merge_hdbscan,
                write_ascii=False,
                compressed=False,
                print_progress=False,
            )

        _, labels_ncuts_all = np.unique(
            np.asarray(merge_ncuts.colors), axis=0, return_inverse=True
        )
        _, labels_hdbscan_all = np.unique(
            np.asarray(merge_hdbscan.colors), axis=0, return_inverse=True
        )

        labels_hdbscan_instances = remove_semantics(labels_nuscenes, labels_hdbscan_all)
        labels_ncuts_instances = remove_semantics(labels_nuscenes, labels_ncuts_all)

        print(labels_hdbscan_all.shape)
        print(labels_nuscenes.shape)
        metrics_clustering.add_stats(
            labels_hdbscan_all, labels_hdbscan_instances, labels_nuscenes
        )
        metrics_ncuts.add_stats(
            labels_ncuts_all, labels_ncuts_instances, labels_nuscenes
        )

        ####### semantic stuff for ncuts
        num_instance, sem_labels = calc_num_instance(
            sem_classes, labels_nuscenes, updated_labels
        )
        intersection = get_intersection(sem_labels, labels_ncuts_all)
        result, (num_splits, sem2sem_splits) = get_num_splits(
            intersection, sem_classes, 200, 0.9, 0.02
        )
        new_labels = create_sub_instances(labels_ncuts_all, sem_labels, result)
        semantic_map = get_semantic_map(new_labels, sem_labels, sem_classes)
        value_counts = Counter(semantic_map.values())
        num_merges = np.sum(list(value_counts.values())) - len(value_counts)

        metrics_ncuts.num_merges += num_merges
        metrics_ncuts.sem2sem_splits += sem2sem_splits
        metrics_ncuts.num_splits += num_splits

        ####### semantic stuff for hdbscan
        intersection = get_intersection(sem_labels, labels_hdbscan_all)
        result, (num_splits, sem2sem_splits) = get_num_splits(
            intersection, sem_classes, 200, 0.9, 0.02
        )
        new_labels = create_sub_instances(labels_hdbscan_all, sem_labels, result)
        semantic_map = get_semantic_map(new_labels, sem_labels, sem_classes)
        value_counts = Counter(semantic_map.values())
        num_merges = np.sum(list(value_counts.values())) - len(value_counts)

        metrics_clustering.num_merges += num_merges
        metrics_clustering.sem2sem_splits += sem2sem_splits
        metrics_clustering.num_splits += num_splits

        if vis == True:
            o3d.visualization.draw_geometries([merge_ncuts])
            o3d.visualization.draw_geometries([merge_hdbscan])
            o3d.visualization.draw_geometries([kitti_pcd])  # semantics
            o3d.visualization.draw_geometries([map_instances])

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
print("Total skipped due to convergence issue", cnt)
print(config)
metrics_clustering.compute_stats_final()
print(" Semantic Results HDBScan ---------------")
print("number of splits", metrics_clustering.num_splits)
print("sem sem splits", metrics_clustering.sem2sem_splits)
print("num merges", metrics_clustering.num_merges)
print("---------------------")


metrics_ncuts.compute_stats_final()
print(" Semantic Results NCuts ---------------")
print("number of splits", metrics_ncuts.num_splits)
print("sem sem splits", metrics_ncuts.sem2sem_splits)
print("num merges", metrics_ncuts.num_merges)
print("---------------------")
