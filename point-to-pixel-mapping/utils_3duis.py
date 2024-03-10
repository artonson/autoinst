import sklearn
from sklearn.cluster import DBSCAN, HDBSCAN
import hdbscan
import yaml
import scipy
import torch.nn as nn
import torch
import numpy as np
import open3d as o3d

from utils.UIS.utils import *
from utils.UIS.minkunet import *
from utils.UIS.collations import *
from utils.UIS.corr_utils import *
from utils.UIS.pcd_preprocess import *
from visualization_utils import *
from point_cloud_utils import (
    get_pcd,
    transform_pcd,
    remove_isolated_points,
    get_subpcd,
    get_statistical_inlier_indices,
    merge_chunks_unite_instances,
    kDTree_1NN_feature_reprojection,
    remove_semantics,
    get_merge_pcds,
    merge_chunks_unite_instances,
    merge_chunks_unite_instances2,
)
from chunk_generation import get_indices_feature_reprojection

config = "utils/UIS/instance_seg.yaml"
cfg = yaml.safe_load(open(config))
params = cfg

set_deterministic()
model = MinkUNet(in_channels=4, out_channels=96).type(torch.FloatTensor)
# checkpoint = torch.load(cfg['model']['checkpoint'], map_location=torch.device('cuda'))
checkpoint = torch.load(
    "utils/UIS/epoch199_model_segcontrast.pt", map_location=torch.device("cuda")
)
model.cuda()
# model.load_state_dict(checkpoint[cfg['model']['checkpoint_key']])
model.load_state_dict(checkpoint[cfg["model"]["checkpoint_key"]])
model.dropout = nn.Identity()


for param in model.parameters():
    param.require_grads = False


def uniform_down_sample_with_indices(points, every_k_points):
    # Create a new point cloud for the downsampled output

    # List to hold the indices of the points that are kept
    indices = []

    # Iterate over the points and keep every k-th point
    for i in range(0, points.shape[0], every_k_points):
        indices.append(i)

    return indices


def downsample_chunk(points):
    num_points_to_sample = 30000
    every_k_points = int(points.shape[0] / num_points_to_sample)
    indeces = uniform_down_sample_with_indices(points, every_k_points)

    return points[indeces]


def segcontrast_preprocessing(p, sem_labels, resolution=0.05, num_points="inf"):
    coord_p, feats_p, cluster_p = point_set_to_coord_feats(
        p, sem_labels, resolution, num_points
    )
    return coord_p, feats_p, cluster_p


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


def UIS3D_clustering(
    pcd_nonground_chunk,
    pcd_ground_chunk,
    center_id,
    center_position,
    eps=0.3,
    min_samples=10,
    tarl=False,
    height_thresh=0.6,
    roi_size=20,
):
    """
    Perform DBSCAN clustering on the point cloud data.

    :param cur_pcd: Current point cloud for clustering.
    :param pcd_all: All point cloud data.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: Cluster labels for each point in the point cloud.
    """

    inliers = get_statistical_inlier_indices(pcd_ground_chunk)
    ground_inliers = get_subpcd(pcd_ground_chunk, inliers)
    mean_hight = np.mean(np.asarray(ground_inliers.points)[:, 2])
    in_idcs = np.where(
        np.asarray(ground_inliers.points)[:, 2] < (mean_hight + height_thresh)
    )[0]
    cut_hight = get_subpcd(ground_inliers, in_idcs)
    cut_hight.paint_uniform_color([0, 0, 0])

    in_idcs = None

    # in_idcs = np.where(np.asarray(pcd_nonground_chunk.points)[:,2] > (mean_hight + 0.05))[0]
    # pcd_nonground_corrected = get_subpcd(pcd_nonground_chunk, in_idcs)
    pcd_nonground_corrected = pcd_nonground_chunk

    merge_orig = pcd_nonground_corrected + cut_hight

    pcd_nonground_downsampled = o3d.geometry.PointCloud()
    pts_downsampled = downsample_chunk(np.asarray(pcd_nonground_corrected.points))
    pcd_nonground_downsampled.points = o3d.utility.Vector3dVector(pts_downsampled)

    ground_downsampled = o3d.geometry.PointCloud()
    pts_downsampled_ground = downsample_chunk(np.asarray(cut_hight.points))
    ground_downsampled.points = o3d.utility.Vector3dVector(pts_downsampled_ground)

    # clustering = DBSCAN(eps=eps, min_samples=min_samples)
    # clustering = HDBSCAN(min_cluster_size=10).fit(pts_downsampled)
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

    merged_chunk = pcd_nonground_downsampled + ground_downsampled

    labels_nonground = clustering.labels_.reshape(-1, 1) + 2

    points = np.asarray(merged_chunk.points)
    labels = np.ones((points.shape[0], 1)) * -1

    ground_labels = np.ones(points.shape[0]) * -1
    non_ground_size = np.asarray(pcd_nonground_downsampled.points).shape[0]
    ground_labels[:non_ground_size] = 1
    labels[:non_ground_size] = labels_nonground
    pcd_cur = color_pcd_by_labels(merged_chunk, labels)
    # o3d.visualization.draw_geometries([pcd_cur])
    ins, num_pts = np.unique(labels, return_counts=True)

    mask = np.ones(labels.shape[0], dtype=bool)
    mask[non_ground_size:] = False
    points = np.concatenate((points, np.ones((points.shape[0], 1))), 1)
    mean_x = points[:, 0].mean()
    mean_y = points[:, 1].mean()
    mean_z = points[:, 2].mean()

    points[:, 0] -= mean_x
    points[:, 1] -= mean_y
    points[:, 2] -= mean_z

    print(np.unique(labels))
    coord_p, feats_p, cluster_p = segcontrast_preprocessing(points, labels)
    slc_full = np.zeros((points.shape[0],), dtype=int)
    pred_ins_full = np.ones((points.shape[0],), dtype=int)
    pred_ins_full[non_ground_size:] = 0
    for cluster in ins:
        cls_points = np.where(cluster_p == cluster)[0]

        if (
            cluster == -1 or len(cls_points) <= 10
        ):  ##ignore ground (-1) and small clusters
            continue
        # get cluster
        cluster_center = coord_p[cls_points].mean(axis=0)

        # crop a ROI around the cluster
        window_points = crop_region(
            coord_p, cluster_p, cluster, roi_size
        )  ##use original roi

        # skip when ROI is empty
        if not np.sum(window_points):
            continue

        # get closest point to the center
        center_dists = np.sqrt(
            np.sum((coord_p[window_points] - cluster_center) ** 2, axis=-1)
        )
        cluster_center = np.argmin(center_dists)

        # build input only with the ROI points
        x_forward = numpy_to_sparse_tensor(
            coord_p[window_points][np.newaxis, :, :],
            feats_p[window_points][np.newaxis, :, :],
        )

        # forward pass ROI
        model.eval()
        x_forward.F.requires_grad = True
        out = model(x_forward.sparse())
        out = out.slice(x_forward)

        # reset grads to compute saliency
        x_forward.F.grad = None

        # compute saliency for the point in the center
        slc = get_cluster_saliency(
            x_forward, out, np.where(cluster_p[window_points] == cluster)[0]
        )
        slc_ = slc.copy()

        # place the computed saliency into the full point cloud for comparison
        slc_full[window_points] = np.maximum(slc_full[window_points], slc)

        # build graph representation
        G = build_graph(
            out.F.detach().cpu().numpy(),
            slc[:, np.newaxis],
            coord_p[window_points],
            cluster_center,
            np.sum(cluster_p == cluster),
            params,
            ground_labels[window_points],
            np.where(cluster_p[window_points] == cluster)[0],
        )
        # perform graph cut
        # G = scipy.sparse.csr_matrix(G) -> try out this line
        ins_points = graph_cut(G)
        # create point-wise prediction matrix
        pred_ins = np.zeros((len(x_forward),)).astype(int)
        if len(ins_points) != 0:
            pred_ins[ins_points] = cluster

        # ignore assigned ground labels
        ins_ground = ground_labels[window_points] == -1
        pred_ins[ins_ground] = 0

        pred_ins_full[window_points] = np.maximum(
            pred_ins_full[window_points], pred_ins
        )
        # pred_ins_full[ins_ground] = 0

    # pcd_cur = color_pcd_by_labels(merged_chunk,pred_ins_full)
    # o3d.visualization.draw_geometries([pcd_cur])
    print(np.unique(pred_ins_full))
    colors_gen = generate_random_colors(500)

    # Reproject cluster labels to the original point cloud size
    cluster_labels = np.ones((len(merge_orig.points), 1)) * -1
    labels_orig = kDTree_1NN_feature_reprojection(
        cluster_labels, merge_orig, pred_ins_full.reshape(-1, 1), merged_chunk
    )
    colors = np.zeros((labels_orig.shape[0], 3))
    unique_labels = list(np.unique(labels_orig))

    for j in unique_labels:
        cur_idcs = np.where(labels_orig == j)[0]
        if j == 0:
            colors[cur_idcs] = np.array([0, 0, 0])

        else:
            colors[cur_idcs] = np.array(colors_gen[unique_labels.index(j)])

    merge_orig.colors = o3d.utility.Vector3dVector(colors / 255.0)

    return merge_orig
