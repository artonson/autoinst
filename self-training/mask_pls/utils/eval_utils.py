import random
import open3d as o3d
import numpy as np
import copy          

BACKGROUND_LABEL = 0


def merge_labels(gt_labels, pred_confs, ins,points,kitti_labels, topk=5):
    confs_dict = {}
    for clid in np.unique(ins):
        (cur_idcs,) = np.where(ins == clid)

        all_confs = pred_confs[cur_idcs]
        mean_confs = all_confs.mean()
        confs_dict[clid] = mean_confs
        
    diff = 30

    sorted_confs_dict = dict(
        sorted(
            confs_dict.items(),
            key=lambda item: item[1],
            reverse=True))
    topk_cluster_ids = list(sorted_confs_dict.keys())[:topk]
    all_labels_vis = np.ones((ins.shape[0],)) * BACKGROUND_LABEL
    
    for clid in topk_cluster_ids:
        (cur_idcs,) = np.where(ins == clid)
        if confs_dict[clid] > 0.4 : 
            all_labels_vis[cur_idcs] = clid
    
    pcd_conf = color_points_by_labels(points,all_labels_vis)
    pcd_gt = color_points_by_labels(points,gt_labels)
    pcd_pred = color_points_by_labels(points,ins)
    pcd_kitti = color_points_by_labels(points,kitti_labels)
    
    pcd_kitti.translate([0,-diff,0])
    pcd_gt.translate([0,diff,0])
    pcd_pred.translate([0,2*diff,0])
    
    
    new_gt_labels = copy.deepcopy(all_labels_vis)
    pred_labels = all_labels_vis
    gt_filtered = __filter_unsegmented(gt_labels)
    pred_filtered = __filter_unsegmented(pred_labels)

    pred_used = set()
    for gt_label in np.unique(gt_filtered):
        gt_indices = np.where(gt_labels == gt_label)
        overlap_flag = False
        if gt_label == BACKGROUND_LABEL:
            continue
        for pred_label in np.unique(pred_filtered):
            if pred_label in pred_used:
                continue

            pred_indices = np.where(pred_labels == pred_label)
            is_overlap = __is_overlapped_iou(pred_indices[0], gt_indices[0])

            if is_overlap:
                overlap_flag = True
                break

        if not overlap_flag:
            (background_idcs,) = np.where(new_gt_labels == BACKGROUND_LABEL)
            remaining_idcs = np.intersect1d(gt_indices, background_idcs)
            if remaining_idcs.shape[0] < 20 : 
                continue

            if gt_label not in list(np.unique(new_gt_labels)):
                new_gt_labels[remaining_idcs] = gt_label
            else:
                new_gt_labels[remaining_idcs] = max(
                    list(np.unique(new_gt_labels))) + 1
                    
    pcd_merged = color_points_by_labels(points,new_gt_labels)
    pcd_merged.translate([0,3 * diff,0])
    o3d.visualization.draw_geometries([pcd_conf,pcd_gt,pcd_pred,pcd_merged,pcd_kitti])
    return new_gt_labels.copy() + 3, all_labels_vis




def find_merge_candidates(points, small_segments, ins):
    # Placeholder for the actual implementation
    # This should return a dictionary mapping small segment IDs to their best merge candidates
    # Implement this based on your specific data and requirements
    merge_candidates = {}
    for small_segment in small_segments:
        # Find the closest segment in terms of average distance between points
        # This is a simplified example and should be replaced with your actual logic
        distances = np.linalg.norm(points[ins == small_segment] - points[:, None], axis=2).mean(axis=0)
        closest_segment = np.argmin(distances)
        merge_candidates[small_segment] = closest_segment
    return merge_candidates


def get_average(l):
    return sum(l) / len(l)



def generate_random_colors(N):
    colors = set()
    while len(colors) < N:
        new_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.add(new_color)

    return list(colors)


def color_points_by_labels(points, labels, unique_labels=None, shift=False):
    pcd = o3d.geometry.PointCloud()
    random_colors = generate_random_colors(100)

    pcd.points = o3d.utility.Vector3dVector(points)
    colors = []
    unique_labels = list(np.unique(labels))

    for i in range(labels.shape[0]):
        if labels[i] == 0:
            colors.append([0, 0, 0])
        else:
                colors.append(
                random_colors[unique_labels.index(int(labels[i]))])

        # colors.append(random_colors[int(labels[i])])

    pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors) / 255)

    return pcd


def __calc_tp(
    pred_labels,
    gt_labels
):
    """
    :param pred_labels: labels of points corresponding to segmented instances
    :param gt_labels: labels of points corresponding to ground truth instances
    :param tp_condition: helper function to calculate statistics
    :return: true positive received using pred_labels and gt_labels
    """
    true_positive = 0

    unique_gt_labels = __filter_unsegmented(np.unique(gt_labels))
    unique_pred_labels = __filter_unsegmented(np.unique(pred_labels))

    pred_used = set()
    tp_condition_function = __is_overlapped_iou

    for gt_label in unique_gt_labels:
        gt_indices = np.where(gt_labels == gt_label)
        gt_indices = gt_indices[0]
        for pred_label in unique_pred_labels:
            if pred_label in pred_used:
                continue

            pred_indices = np.where(pred_labels == pred_label)
            pred_indices = pred_indices[0]
            is_overlap = tp_condition_function(
                pred_indices, gt_indices, threshold=0.5)

            if is_overlap:
                true_positive += 1
                pred_used.add(pred_label)
                break

    return true_positive


def __filter_unsegmented(
        label_array):
    """
    :param label_array: labels of points corresponding to segmented instances
    :return: labels array where all unsegmented points (with label equal to metrics_base.metrics.constants.UNSEGMENTED_LABEL)
     are deleted
    """

    unsegmented_indices = np.where(
        label_array == BACKGROUND_LABEL
    )

    # label_array[unsegmented_indices] = BACKGROUND_LABEL
    label_array = np.delete(
        label_array,
        unsegmented_indices[0],
    )
    return label_array


def __iou(
    pred_indices,
    gt_indices,
) -> np.float64:
    intersection = np.intersect1d(pred_indices, gt_indices)
    union = np.union1d(pred_indices, gt_indices)

    return intersection.size / union.size


weighted_scores_pred = {
    'precision': [],
    "recall": [],
    "fscore": [],
    'tps': 0,
    'preds': 0,
    'unique_gts': 0}
weighted_scores_pseudo = {
    'precision': [],
    "recall": [],
    "fscore": [],
    'tps': 0,
    'preds': 0}
weighted_scores_cluster = {
    'precision': [],
    "recall": [],
    "fscore": [],
    'tps': 0,
    'preds': 0}
weighted_scores_maskpls = {
    'precision': [],
    "recall": [],
    "fscore": [],
    'tps': 0,
    'preds': 0}
weighted_scores_merged = {
    'precision': [],
    "recall": [],
    "fscore": [],
    'tps': 0,
    'preds': 0}


def __is_overlapped_iou(
    pred_indices,
    gt_indices,
    threshold=0.5,
) -> bool:
    """
    :param pred_indices: indices of points belonging to the given predicted label
    :param gt_indices: indices of points belonging to the given predicted label
    :param threshold: value at which instances will be selected as overlapped enough
    :return: true if IoU >= metrics_base.metrics.constants.IOU_THRESHOLD_FULL
    """
    if threshold is None:
        threshold = 0.5

    return __iou(pred_indices, gt_indices) >= threshold


def calculate_centroids(all_points, labels):
    unique_labels = list(np.unique(labels).astype(np.int8))
    centroids = []
    max_size = 0
    max_id = None
    idx = None
    for i, label in enumerate(unique_labels):
        points = np.asarray(all_points)[labels == label]
        centroid = np.mean(points, axis=0)
        centroids.append(centroid)
        if points.shape[0] > max_size:
            max_size = points.shape[0]
            max_id = label
            idx = i

    del centroids[idx]
    unique_labels.remove(max_id)  # remove ground label

    return np.array(centroids), unique_labels, max_id

# Step 2: Merge clusters based on centroid proximity


def merge_clusters(centroids, unique_labels, threshold):
    label_map = {label: label for label in unique_labels}
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distance = np.linalg.norm(centroids[i] - centroids[j])
            if distance < threshold:
                smaller_label = min(
                    label_map[unique_labels[i]], label_map[unique_labels[j]])
                label_map[unique_labels[i]] = smaller_label
                label_map[unique_labels[j]] = smaller_label
    return label_map