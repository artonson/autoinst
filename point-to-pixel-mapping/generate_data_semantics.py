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


config = config_spatial 
if 'maskpls' in config["name"]:
    from utils.maskpls.predict_maskpls import RefinerModel

print("Starting with config ", config)

def create_folder(name):
    if os.path.exists(name) == False:
        os.makedirs(name)


out_folder_instances = out_folder + "instances/"
if os.path.exists(out_folder_instances) == False:
    os.makedirs(out_folder_instances)

out_folder_ncuts = out_folder + config["out_folder"]
if os.path.exists(out_folder_ncuts) == False:
    os.makedirs(out_folder_ncuts)

data_store_train = out_folder + "train/"
if os.path.exists(data_store_train) == False:
    os.makedirs(data_store_train)


alpha = config["alpha"]
theta = config["theta"]
gamma = config["gamma"]
ncuts_threshold = config["T"]

seqs = [0]
for seq in seqs:
    if seq in exclude:
        continue
    print("Sequence", seq)
    dataset = create_kitti_odometry_dataset(
        DATASET_PATH, seq, ncuts_mode=True
    )
    chunks_idcs = divide_indices_into_chunks(len(dataset))

    data_store_folder = out_folder + str(seq) + "/"
    if os.path.exists(data_store_folder) == False:
        os.makedirs(data_store_folder)

    data_store_folder_train_cur = data_store_train + str(seq) + "/"
    if os.path.exists(data_store_folder_train_cur) == False:
        os.makedirs(data_store_folder_train_cur)

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

        if "maskpls" in config["name"] and config["name"] != "maskpls_supervised":
            maskpls = RefinerModel(dataset="kitti")

        if (
            os.path.exists(f"{out_folder}non_ground{seq}_{cur_idx}.pcd")
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
                sequence_num=seq,
                ground_segmentation_method=ground_segmentation_method,
                cur_idx=cur_idx,
            )

        if (
            os.path.exists(
                f"{out_folder}pcd_nonground_minor{seq}_{cur_idx}.pcd"
            )
            == False
        ):
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
                seq,
                minor_voxel_size,
                ground_mode=ground_segmentation_method,
                cur_idx=cur_idx,
            )

            print("write pcds")
            print(pcd_ground_minor)
            o3d.io.write_point_cloud(
                f"{out_folder}pcd_ground_minor{seq}_{cur_idx}.pcd",
                pcd_ground_minor,
                write_ascii=False,
                compressed=False,
                print_progress=True,
            )
            o3d.io.write_point_cloud(
                f"{out_folder}pcd_nonground_minor{seq}_{cur_idx}.pcd",
                pcd_nonground_minor,
                write_ascii=False,
                compressed=False,
                print_progress=True,
            )
            print("write labels")
            np.savez(
                f"{out_folder}kitti_labels_preprocessed{seq}_{cur_idx}.npz",
                instance_nonground=kitti_labels["instance_nonground"],
                instance_ground=kitti_labels["instance_ground"],
                seg_ground=kitti_labels["seg_ground"],
                seg_nonground=kitti_labels["seg_nonground"],
            )
        print("load pcd")
        pcd_ground_minor = o3d.io.read_point_cloud(
            f"{out_folder}pcd_ground_minor{seq}_{cur_idx}.pcd"
        )
        pcd_nonground_minor = o3d.io.read_point_cloud(
            f"{out_folder}pcd_nonground_minor{seq}_{cur_idx}.pcd"
        )

        print("load data")
        kitti_labels_orig = {}
        with np.load(
            f"{out_folder}kitti_labels_preprocessed{seq}_{cur_idx}.npz"
        ) as data:
            kitti_labels_orig["instance_ground"] = data["instance_ground"]
            kitti_labels_orig["instance_nonground"] = data["instance_nonground"]
            kitti_labels_orig["seg_nonground"] = data["seg_nonground"]
            kitti_labels_orig["seg_ground"] = data["seg_ground"]

        with np.load(
            f"{out_folder}all_poses_" + str(seq) + "_" + str(cur_idx) + ".npz"
        ) as data:
            all_poses = data["all_poses"]
            T_pcd = data["T_pcd"]
            first_position = T_pcd[:3, 3]

        print("pose downsample")
        if (
            os.path.exists(f"{out_folder}subsampled_data{seq}_{cur_idx}.npz")
            == False
        ):
            poses, positions, sampled_indices_local, sampled_indices_global = (
                subsample_and_extract_positions(
                    all_poses,
                    ind_start=ind_start,
                    sequence_num=seq,
                    out_folder=out_folder,
                    cur_idx=cur_idx,
                )
            )

        with np.load(
            f"{out_folder}subsampled_data{seq}_{cur_idx}.npz"
        ) as data:
            poses = data["poses"]
            positions = data["positions"]
            sampled_indices_local = list(data["sampled_indices_local"])
            sampled_indices_global = list(data["sampled_indices_global"])
        
        pcd_col = color_pcd_by_labels(pcd_nonground_minor,kitti_labels_orig['instance_nonground'],colors=colors)
        
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
            out_folder_ncuts + str(seq) + "_" + str(cur_idx) + "/"
        )
        
        out_folder_instances_cur = (
            out_folder_instances + str(seq) + "_" + str(cur_idx) + "/"
        )

        create_folder(out_folder_ncuts_cur)
        if config["gt"]:
            create_folder(out_folder_instances_cur)

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
            
            name = str(center_ids[sequence]).zfill(6) + ".pcd" 
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

                pred_pcd = pcd_chunk + pcd_chunk_ground

            elif "maskpls" in config["name"]:
                inliers = get_statistical_inlier_indices(pcd_ground_chunks[sequence])
                ground_inliers = get_subpcd(pcd_ground_chunks[sequence], inliers)
                mean_hight = np.mean(np.asarray(ground_inliers.points)[:, 2])
                inliers_ground = np.where(
                    np.asarray(ground_inliers.points)[:, 2] < (mean_hight + 0.6)
                )[0]
                pcd_chunk_ground = get_subpcd(ground_inliers, inliers_ground)
                pcd_chunk_ground.paint_uniform_color([0, 0, 0])

                input_pcd = pcd_nonground_chunks[sequence] + pcd_chunk_ground

                if "supervised" not in config["name"]:
                    print("unsupervised")
                    pred_pcd = maskpls.forward_and_project(input_pcd)
                else:
                    pred_pcd = maskpls.forward_and_project(input_pcd)
            
            if config["gt"]:
                    inst_ground = kitti_labels["ground"]["instance"][sequence][inliers][
                        inliers_ground
                    ]
                    seg_ground = kitti_labels["ground"]["semantic"][sequence][inliers][
                        inliers_ground
                    ]

                    kitti_chunk_instance = color_pcd_by_labels(
                        copy.deepcopy(pcd_nonground_chunks[sequence]),
                        kitti_labels["nonground"]["instance"][sequence].reshape(
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

                    o3d.io.write_point_cloud(
                        out_folder_instances_cur + name,
                        instance_pcd,
                        write_ascii=False,
                        compressed=False,
                        print_progress=False,
                    )
            

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

            map_instances = merge_unite_gt(
                get_merge_pcds(out_folder_instances_cur[:-1])
            )
         
            
            _, labels_instances = np.unique(
                np.asarray(map_instances.colors), axis=0, return_inverse=True
            )
            

        
        if "maskpls" in config["name"]:

            with open(
                data_store_folder
                + config["name"]
                + "_confs"
                + str(seq)
                + "_"
                + str(cur_idx)
                + ".json",
                "w",
            ) as fp:
                json.dump(maskpls.confs_dict, fp)
        
        zero_idcs = np.where(labels_instances == 0)[0]
        
        
        metrics = Metrics(config['name'] + ' ' + str(seq))
        colors, labels_ncuts_all = np.unique(
            np.asarray(merge_ncuts.colors), axis=0, return_inverse=True
        )
        
        o3d.io.write_point_cloud(
            data_store_folder
            + config["name"]
            + str(seq)
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
            + "kitti_instances_"
            + str(seq)
            + "_"
            + str(cur_idx)
            + ".pcd",
            map_instances,
            write_ascii=False,
            compressed=False,
            print_progress=False,
        )
        
        instance_preds = remove_semantics(labels_instances, copy.deepcopy(labels_ncuts_all))
        if 'maskpls' in config['name']:
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
