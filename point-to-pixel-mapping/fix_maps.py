import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
from metrics.metrics_class import Metrics
import copy
from concurrent.futures import ThreadPoolExecutor
import random
import json


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


# value = "spatial_1.0_tarl_0.5_dino_0.1_t_0.005"
value = "spatial_1.0_tarl_0.5_dino_0.1_t_0.005"
# value = "spatial_1.0_tarl_0.5_dino_0.1_t_0.005"
value = "3duis_fix"
value = "sam3d_fix"
# value = "hdbscan"
value = "spatial_1.0_tarl_0.5_t_0.04"
# value = "maskpls_tarl_spatial_7_"
#value = "maskpls_no_filter_"
# value = "maskpls_hdbscan_6_"
value = "spatial_1.0_t_0.075"


metrics_dict = {}
metrics_dict["ap0.25"] = []
metrics_dict["ap0.5"] = []
metrics_dict["ap"] = []
metrics_dict["p"] = []
metrics_dict["r"] = []
metrics_dict["f1"] = []
metrics_dict["lstq"] = []
metrics_dict["panoptic"] = []


seqs = [0, 2, 3, 5, 6, 7, 8, 9, 10]
map_num_dict_kitti = {0: 5, 2: 5, 3: 1, 5: 3, 6: 1, 7: 1, 8: 4, 9: 2, 10: 2}


import time

maskpls_base_dir = (
    "/home/cedric/unsup_3d_instances/point-to-pixel-mapping/pcd_preprocessed/semantics/"
)
maskpls = False
seqs = [6]

folder = '/home/cedric/unsup_3d_instances/point-to-pixel-mapping/pcd_preprocessed/semantics/'

for seq in tqdm(seqs, desc=" Sequence \n"):

    for div_num in tqdm(range(map_num_dict_kitti[seq]), desc="Subdivided Map \n"):
        metrics = Metrics(value + " seq " + str(seq) + " div_num" + str(div_num))
        inst_pred = o3d.io.read_point_cloud(
            f"{folder}/{seq}/kitti_instances_{seq}_{str(div_num)}.pcd"
        )
        _, labels_inst_all = np.unique(
            np.asarray(inst_pred.colors), axis=0, return_inverse=True
        )

        labels_sem = np.load(f"{folder}{seq}/kitti_semantic{seq}_{str(div_num)}.npz")["labels"]

        zero_idcs = np.where(labels_inst_all == 0)[0]
        new_labels_inst = labels_inst_all + (
            labels_sem * np.unique(labels_inst_all).shape[0]
        )

        new_labels_inst[zero_idcs] = 0

        pcd_pred = o3d.io.read_point_cloud(f"{folder}{seq}/{value}{seq}_{str(div_num)}.pcd")
        #pcd_pred = o3d.io.read_point_cloud(
        #    f"{maskpls_base_dir}{seq}/{value}{seq}_{str(div_num)}.pcd"
        #)

        colors, labels_ncuts_all = np.unique(
            np.asarray(pcd_pred.colors), axis=0, return_inverse=True
        )

        if maskpls == True:
            json_merge = (
                f"{maskpls_base_dir}{seq}/{value}_confs{seq}_{str(div_num)}.json"
            )
            with open(json_merge) as f:
                data_json = json.load(f)

            instance_preds = remove_semantics(labels_inst_all, labels_ncuts_all)

            label_to_confidence = {}
            pcd_cols = np.asarray(pcd_pred.colors)
            for label in list(np.unique(instance_preds)):
                idcs = np.where(instance_preds == label)[0]
                cur_color = pcd_cols[idcs[0]]
                key = (
                    str(int(cur_color[0] * 255))
                    + "|"
                    + str(int(cur_color[1] * 255))
                    + "|"
                    + str(int(cur_color[2] * 255))
                )
                label_to_confidence[label] = data_json[key]

            out, aps_lstq_dict = metrics.update_stats(
                labels_ncuts_all,
                instance_preds,
                new_labels_inst,
                confs=label_to_confidence,
            )

        else:
            instance_preds = remove_semantics(
                labels_inst_all, copy.deepcopy(labels_ncuts_all)
            )

            start = time.time()
            out, aps_lstq_dict = metrics.update_stats(
                labels_ncuts_all, instance_preds, new_labels_inst
            )

        print("---------")
        print(out)
        print(aps_lstq_dict)
        metrics_dict["panoptic"].append(out["panoptic"])
        metrics_dict["p"].append(out["precision"])
        metrics_dict["r"].append(out["recall"])
        metrics_dict["f1"].append(out["fScore"])
        metrics_dict["lstq"].append(aps_lstq_dict["lstq"])
        metrics_dict["ap"].append(aps_lstq_dict["ap"])
        metrics_dict["ap0.25"].append(aps_lstq_dict["0.25"])
        metrics_dict["ap0.5"].append(aps_lstq_dict["0.5"])
        print(metrics_dict)


with open("results_jsons/" + value + ".json", "rb") as f:
    json.dumps(metrics_dict, f)

print("All metrics averaged ")
for key in metrics_dict.keys():
    print("Key ", key)
    print("Key : ", sum(metrics_dict[key]) / float(len(metrics_dict[key])))