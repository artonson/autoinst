import numpy as np
import os
import sys
import open3d as o3d

src_path = os.path.abspath("../..")
import copy
import json
if src_path not in sys.path:
    sys.path.append(src_path)

from tqdm import tqdm
import gc
from ncuts.ncuts_utils import (
    ncuts_chunk,
    get_merge_pcds,
)
from dataset.dataset_utils import (
    chunk_and_downsample_point_clouds,
    color_pcd_by_labels,
    create_kitti_odometry_dataset,
    process_and_save_point_clouds,
    load_and_downsample_point_clouds,
    subsample_and_extract_positions, 
)

from utils.point_cloud.point_cloud_utils import (
    get_subpcd,
    get_statistical_inlier_indices,
    merge_chunks_unite_instances2,
    divide_indices_into_chunks,
    merge_unite_gt,
    remove_semantics,
    write_pcd,
    
)

from utils.visualization_utils import (
    color_pcd_by_labels,
    generate_random_colors_map,
)

from utils.point_cloud.chunk_generation import (
    indices_per_patch,
)

from utils.visualization_utils import generate_random_colors_map
from metrics.metrics_class import Metrics
from config import *


if 'maskpls' in CONFIG["name"]:
    from utils.maskpls.predict_maskpls import RefinerModel

print("Starting with config ", CONFIG)

def create_folder(name):
    if os.path.exists(name) == False:
        os.makedirs(name)

create_folder(OUT_FOLDER_NCUTS)
create_folder(OUT_FOLDER_INSTANCES)
create_folder(OUT_FOLDER_TRAIN)

seqs = [0]
for seq in seqs:
    if seq in exclude:
        continue
    print("Sequence", seq)
    dataset = create_kitti_odometry_dataset(
        DATASET_PATH, seq, ncuts_mode=True
    )
    chunks_idcs = divide_indices_into_chunks(len(dataset))

    data_store_folder = OUT_FOLDER + str(seq) + "/"
    create_folder(data_store_folder)

    data_store_folder_train_cur = OUT_FOLDER_TRAIN + str(seq) + "/"
    create_folder(data_store_folder_train_cur)

    print("STORE FOLDER", data_store_folder)

    for cur_idx, cidcs in enumerate(chunks_idcs[start_chunk:]):
        print(cur_idx)
        colors = generate_random_colors_map(6000)

        ind_start = cidcs[0]
        ind_end = cidcs[1]
        cur_idx = int(ind_start / 1000)
        if ind_end - ind_start < 200: #do not create small maps 
            continue

        print("ind start", ind_start)
        print("ind end", ind_end)

        if "maskpls" in CONFIG["name"]:
            maskpls = RefinerModel(dataset="kitti")

        
        process_and_save_point_clouds(
                dataset,
                ind_start,
                ind_end,
                sequence_num=seq,
                cur_idx=cur_idx,
            )
        
        load_and_downsample_point_clouds(
                OUT_FOLDER,
                seq,
                ground_mode=ground_segmentation_method,
                cur_idx=cur_idx,
            )
            
        print("load pcd")
        pcd_ground_minor = o3d.io.read_point_cloud(
            f"{OUT_FOLDER}pcd_ground_minor{seq}_{cur_idx}.pcd"
        )
        pcd_nonground_minor = o3d.io.read_point_cloud(
            f"{OUT_FOLDER}pcd_nonground_minor{seq}_{cur_idx}.pcd"
        )

        print("load data")
        kitti_labels_orig = {}
        with np.load(
            f"{OUT_FOLDER}kitti_labels_preprocessed{seq}_{cur_idx}.npz"
        ) as data:
            kitti_labels_orig["instance_ground"] = data["instance_ground"]
            kitti_labels_orig["instance_nonground"] = data["instance_nonground"]
            kitti_labels_orig["seg_nonground"] = data["seg_nonground"]
            kitti_labels_orig["seg_ground"] = data["seg_ground"]

        with np.load(
            f"{OUT_FOLDER}all_poses_" + str(seq) + "_" + str(cur_idx) + ".npz"
        ) as data:
            all_poses = data["all_poses"]
            T_pcd = data["T_pcd"]
            first_position = T_pcd[:3, 3]


        subsample_and_extract_positions(
                    all_poses,
                    ind_start=ind_start,
                    sequence_num=seq,
                    cur_idx=cur_idx,
                )

        with np.load(
            f"{OUT_FOLDER}subsampled_data{seq}_{cur_idx}.npz"
        ) as data:
            poses = data["poses"]
            positions = data["positions"]
            sampled_indices_local = list(data["sampled_indices_local"])
            sampled_indices_global = list(data["sampled_indices_global"])
        
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

        print("chunk downsample")
        chunk_downsample_dict = chunk_and_downsample_point_clouds(
            pcd_nonground_minor,
            pcd_ground_minor,
            T_pcd,
            positions,
            first_position,
            sampled_indices_global,
            kitti_labels=kitti_labels_orig,
        )
        print("finished downsample")

        out_folder_ncuts_cur = (
            OUT_FOLDER_NCUTS + str(seq) + "_" + str(cur_idx) + "/"
        )
        
        out_folder_instances_cur = (
            OUT_FOLDER_INSTANCES + str(seq) + "_" + str(cur_idx) + "/"
        )

        create_folder(out_folder_ncuts_cur)
        if CONFIG["gt"]:
            create_folder(out_folder_instances_cur)

        patchwise_indices = indices_per_patch(
            T_pcd,
            chunk_downsample_dict['center_positions'],
            positions,
            first_position,
            sampled_indices_global,
        )
        out_data = []
        semantics_kitti = []
        for sequence in tqdm(range(start_seq, len(chunk_downsample_dict['center_ids']))):
            
            name = str(chunk_downsample_dict['center_ids'][sequence]).zfill(6) + ".pcd" 
            print("sequence", sequence)
            if (
                CONFIG["name"] not in ["sam3d", "sam3d_fix", "3duis", "3duis_fix"]
                and "maskpls" not in CONFIG["name"]
            ):
                (
                    merged_chunk,
                    pcd_chunk,
                    pcd_chunk_ground,
                    inliers,
                    inliers_ground,
                ) = ncuts_chunk(
                    dataset,
                    chunk_downsample_dict,
                    pcd_nonground_minor,
                    T_pcd,
                    list(sampled_indices_global),
                    sequence=sequence,
                    patchwise_indices=patchwise_indices,
                )

                pred_pcd = pcd_chunk + pcd_chunk_ground

            elif "maskpls" in CONFIG["name"]:
                inliers = get_statistical_inlier_indices(chunk_downsample_dict['pcd_ground_chunks'][sequence])
                ground_inliers = get_subpcd(chunk_downsample_dict['pcd_ground_chunks'][sequence], inliers)
                mean_hight = np.mean(np.asarray(ground_inliers.points)[:, 2])
                inliers_ground = np.where(
                    np.asarray(ground_inliers.points)[:, 2] < (mean_hight + MEAN_HEIGHT)
                )[0]
                pcd_chunk_ground = get_subpcd(ground_inliers, inliers_ground)
                pcd_chunk_ground.paint_uniform_color([0, 0, 0])

                input_pcd = chunk_downsample_dict['pcd_nonground_chunks'][sequence] + pcd_chunk_ground

                if "supervised" not in CONFIG["name"]:
                    print("unsupervised")
                    pred_pcd = maskpls.forward_and_project(input_pcd)
                else:
                    pred_pcd = maskpls.forward_and_project(input_pcd)
            
            if CONFIG["gt"]:
                    inst_ground = chunk_downsample_dict['kitti_labels']["ground"]["instance"][sequence][inliers][
                        inliers_ground
                    ]

                    kitti_chunk_instance = color_pcd_by_labels(
                        copy.deepcopy(chunk_downsample_dict['pcd_nonground_chunks'][sequence]),
                        chunk_downsample_dict['kitti_labels']["nonground"]["instance"][sequence].reshape(
                            -1,
                        ),
                        colors=colors,
                        gt_labels=instances,
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
                    
                    unique_colors, labels_kitti = np.unique(
                        np.asarray(gt_pcd.colors), axis=0, return_inverse=True
                    )

                    pts = np.asarray(gt_pcd.points)

                    instance_pcd = kitti_chunk_instance + kitti_chunk_instance_ground
                    print("output", data_store_folder + name.split(".")[0])
                    
                    write_pcd(out_folder_instances_cur,name,instance_pcd)

            unique_colors, labels_ncuts = np.unique(
                np.asarray(pred_pcd.colors), axis=0, return_inverse=True
            )

            write_pcd(out_folder_ncuts_cur,name,pred_pcd)


            gc.collect()

        merge_ncuts = merge_chunks_unite_instances2(
                get_merge_pcds(out_folder_ncuts_cur[:-1])
            )


        if CONFIG["gt"]:

            map_instances = merge_unite_gt(
                get_merge_pcds(out_folder_instances_cur[:-1])
            )
         
            
            _, labels_instances = np.unique(
                np.asarray(map_instances.colors), axis=0, return_inverse=True
            )
            

        
        if "maskpls" in CONFIG["name"]:

            with open(
                data_store_folder
                + CONFIG["name"]
                + "_confs"
                + str(seq)
                + "_"
                + str(cur_idx)
                + ".json",
                "w",
            ) as fp:
                json.dump(maskpls.confs_dict, fp)
        
        zero_idcs = np.where(labels_instances == 0)[0]
        
        
        metrics = Metrics(CONFIG['name'] + ' ' + str(seq))
        colors, labels_ncuts_all = np.unique(
            np.asarray(merge_ncuts.colors), axis=0, return_inverse=True
        )
        
        write_pcd(data_store_folder,CONFIG['name'],merge_ncuts,seq,cur_idx)
        write_pcd(data_store_folder,'kitti_instances_',map_instances,seq,cur_idx)
        
        instance_preds = remove_semantics(labels_instances, copy.deepcopy(labels_ncuts_all))
        if 'maskpls' in CONFIG['name']:
            label_to_confidence = {}
            
            pcd_cols = np.asarray(merge_ncuts.colors)
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
                    label_to_confidence[label] = maskpls.confs_dict[key]

            out, aps_lstq_dict = metrics.update_stats(
                labels_ncuts_all,
                instance_preds,
                labels_instances,
                confs=label_to_confidence,
            )
        
        out, aps_lstq_dict = metrics.update_stats(
                labels_ncuts_all,
                instance_preds,
                labels_instances,
        )
