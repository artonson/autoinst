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
    create_kitti_odometry_dataset,
    process_and_save_point_clouds,
    load_and_downsample_point_clouds,
    subsample_and_extract_positions, 
    load_downsampled_pcds,
    load_subsampled_data,
    write_gt_chunk,
    store_train_chunks
)

from utils.point_cloud.point_cloud_utils import (
    merge_chunks_unite_instances2,
    divide_indices_into_chunks,
    merge_unite_gt,
    remove_semantics,
    write_pcd,
    get_corrected_ground,
    downsample_chunk_train,
    
)

from utils.visualization_utils import (
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

for seq in seqs:
    if seq in exclude:
        continue

    if TEST_MAP and seq > 0:
        break
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
            
        pcd_ground_minor,pcd_nonground_minor, kitti_labels_orig,instances, all_poses,T_pcd = load_downsampled_pcds(seq,cur_idx)
        first_position = T_pcd[:3, 3]

        subsample_and_extract_positions(
                    all_poses,
                    ind_start=ind_start,
                    sequence_num=seq,
                    cur_idx=cur_idx,
                )

        poses,positions,sampled_indices_local,sampled_indices_global = load_subsampled_data(seq,cur_idx)

        chunk_downsample_dict = chunk_and_downsample_point_clouds(
            pcd_nonground_minor,
            pcd_ground_minor,
            T_pcd,
            positions,
            first_position,
            sampled_indices_global,
            kitti_labels=kitti_labels_orig,
        )

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
            if "maskpls" not in CONFIG["name"]:
                (
                    merged_chunk,
                    pcd_chunk,
                    pcd_chunk_ground,
                    inst_ground,
                    seg_ground
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
                pcd_chunk_ground,inst_ground = get_corrected_ground(chunk_downsample_dict,sequence,MEAN_HEIGHT)
                pred_pcd = maskpls.forward_and_project(chunk_downsample_dict['pcd_nonground_chunks'][sequence] + pcd_chunk_ground)
                
            if CONFIG["gt"]:
                    gt_pcd = write_gt_chunk(out_folder_instances_cur,name,chunk_downsample_dict,
                                sequence,colors,instances,pcd_chunk_ground,inst_ground)
            
            if GEN_SELF_TRAIN_DATA :
                store_train_chunks(data_store_folder_train_cur,name,merged_chunk,gt_pcd,chunk_downsample_dict,sequence)
                continue

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
            maskpls.store_conf_dict(data_store_folder,CONFIG['name'],seq,cur_idx)
        
        zero_idcs = np.where(labels_instances == 0)[0]
        metrics = Metrics(CONFIG['name'] + ' ' + str(seq))
        colors, labels_ncuts_all = np.unique(
            np.asarray(merge_ncuts.colors), axis=0, return_inverse=True
        )

        write_pcd(data_store_folder,CONFIG['name'],merge_ncuts,seq,cur_idx)
        write_pcd(data_store_folder,'kitti_instances_',map_instances,seq,cur_idx)
        
        instance_preds = remove_semantics(labels_instances, copy.deepcopy(labels_ncuts_all))
        if 'maskpls' in CONFIG['name']:
            label_to_confidence = maskpls.label_to_conf(merge_ncuts,instance_preds)
    
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
        
        if TEST_MAP:
            break
