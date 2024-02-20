import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

src_path = os.path.abspath("../..")
if src_path not in sys.path:
    sys.path.append(src_path)
from dataset.kitti_odometry_dataset import KittiOdometryDataset, KittiOdometryDatasetConfig
from dataset.filters.filter_list import FilterList
from dataset.filters.kitti_gt_mo_filter import KittiGTMovingObjectFilter
from dataset.filters.range_filter import RangeFilter
from dataset.filters.apply_pose import ApplyPose

import scipy
from scipy.spatial.distance import cdist
from normalized_cut import normalized_cut
from ncuts_utils import ncuts_chunk,kDTree_1NN_feature_reprojection_colors, get_merge_pcds
from dataset_utils import * 
from point_cloud_utils import get_pcd, transform_pcd, kDTree_1NN_feature_reprojection, remove_isolated_points, get_subpcd, get_statistical_inlier_indices, merge_chunks_unite_instances
from aggregate_pointcloud import aggregate_pointcloud
from visualization_utils import generate_random_colors, color_pcd_by_labels,generate_random_colors_map
from sam_label_distace import sam_label_distance
from chunk_generation import subsample_positions, chunks_from_pointcloud, indices_per_patch, tarl_features_per_patch, image_based_features_per_patch, dinov2_mean, get_indices_feature_reprojection
from metrics.metrics_class import Metrics


def intersect(pred_indices, gt_indices):
        intersection = np.intersect1d(pred_indices, gt_indices)
        return intersection.size / pred_indices.shape[0]


def color_pcd_by_labels(pcd, labels,colors=None,gt_labels=None):
    
    if colors == None : 
        colors = generate_random_colors(2000)
    pcd_colored = copy.deepcopy(pcd)
    pcd_colors = np.zeros(np.asarray(pcd.points).shape)
    if gt_labels is None :
    	unique_labels = list(np.unique(labels)) 
    else: 
        unique_labels = list(np.unique(gt_labels))
    
    background_color = np.array([0,0,0])


    #for i in range(len(pcd_colored.points)):
    for i in unique_labels:
        if i == -1 : 
            continue
        idcs = np.where(labels == i)
        idcs = idcs[0]
        if i == 0 : 
            pcd_colors[idcs] = background_color
        else : 
            try : 
                pcd_colors[idcs] = np.array(colors[unique_labels.index(i)])
            except : 
                import pdb; pdb.set_trace()
        
        #if labels[i] != (-1):
        #    pcd_colored.colors[i] = np.array(colors[labels[i]]) / 255
    pcd_colored.colors = o3d.utility.Vector3dVector(pcd_colors/ 255)
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

def uniform_down_sample_with_indices(points,every_k_points):
    # Create a new point cloud for the downsampled output
    
    # List to hold the indices of the points that are kept
    indices = []
    
    
    # Iterate over the points and keep every k-th point
    for i in range(0, points.shape[0], every_k_points):
        indices.append(i)

    
    return indices



def downsample_chunk(points,chunk_preds_ncut,kitti_chunk_labels):
        num_points_to_sample = 60000
        every_k_points = int(kitti_chunk_labels.shape[0] /num_points_to_sample) 
        indeces = uniform_down_sample_with_indices(kitti_chunk_labels,every_k_points)

        points = points[indeces]
        chunk_preds_ncut = chunk_preds_ncut[indeces]
        kitti_chunk_labels = kitti_chunk_labels[indeces]
        return points,chunk_preds_ncut,kitti_chunk_labels

def divide_indices_into_chunks(max_index, chunk_size=200):
    chunks = []
    for start in range(0, max_index, chunk_size):
        end = min(start + chunk_size, max_index)
        chunks.append((start, end))
    return chunks


DATASET_PATH = os.path.join('/media/cedric/Datasets1/semantic_kitti/')
import shutil 
import gc 

minor_voxel_size = 0.05
major_voxel_size = 0.35
chunk_size = np.array([25, 25, 25]) #meters
overlap = 23 #meters
ground_segmentation_method = 'patchwork' 
NCUT_ground = False 
out_folder_ncuts = 'test_data/'

out_folder = 'pcd_preprocessed/train_data/'
if os.path.exists(out_folder) == False : 
        os.makedirs(out_folder)

alpha = 0.0
theta = 0.0
colors = generate_random_colors_map(6000)
beta = 0.0
tarl_norm = True 
gamma = 0.1
proximity_threshold = 1.0
ncuts_threshold = 0.08

seqs = list(range(1,3))

for seq in seqs : 
        print("Sequence",seq)
        SEQUENCE_NUM = seq
        dataset = create_kitti_odometry_dataset(DATASET_PATH,SEQUENCE_NUM,ncuts_mode=True)
        chunks_idcs = divide_indices_into_chunks(len(dataset))
        
        
        data_store_folder = out_folder + str(SEQUENCE_NUM) + "/"
        if os.path.exists(data_store_folder) == False :
                        os.makedirs(data_store_folder)
        print("STORE FOLDER",data_store_folder)
        
        for cur_idx,cidcs in enumerate(chunks_idcs):
                #if cur_idx == 0 : 
                #         continue
        
                ind_start = cidcs[0]
                ind_end = cidcs[1]
                
                print('ind start',ind_start)
                print('ind end',ind_end)
                
                if os.path.exists(f'{out_folder}non_ground{SEQUENCE_NUM}_{cur_idx}.pcd') == False : 
                        print('process poses')
                        process_and_save_point_clouds(dataset,ind_start,ind_end,minor_voxel_size=minor_voxel_size,
                                                major_voxel_size=major_voxel_size,icp=False,
                                                out_folder=out_folder,sequence_num=SEQUENCE_NUM,
                                                ground_segmentation_method=ground_segmentation_method,cur_idx=cur_idx)
                
                if os.path.exists(f'{out_folder}pcd_nonground_minor{SEQUENCE_NUM}_{cur_idx}.pcd') == False : 
                #    if True == True : 
                        print('load and downsample points')
                        pcd_ground_minor, pcd_nonground_minor,\
                        all_poses, T_pcd, first_position,kitti_labels = load_and_downsample_point_clouds(out_folder,SEQUENCE_NUM,minor_voxel_size,\
                                                                                ground_mode=ground_segmentation_method,cur_idx=cur_idx)
                        
                        print('write pcds')
                        print(pcd_ground_minor)
                        o3d.io.write_point_cloud(f'{out_folder}pcd_ground_minor{SEQUENCE_NUM}_{cur_idx}.pcd', pcd_ground_minor, write_ascii=False, compressed=False, print_progress=True)
                        o3d.io.write_point_cloud(f'{out_folder}pcd_nonground_minor{SEQUENCE_NUM}_{cur_idx}.pcd', pcd_nonground_minor, write_ascii=False, compressed=False, print_progress=True)
                        print('write labels')
                        np.savez(f'{out_folder}kitti_labels_preprocessed{SEQUENCE_NUM}_{cur_idx}.npz',
                                                                instance_nonground=kitti_labels['instance_nonground'],
                                                                instance_ground=kitti_labels['instance_ground'],
                                                                seg_ground = kitti_labels['seg_ground'],
                                                                seg_nonground=kitti_labels['seg_nonground']
                                                                )
                print("load pcd")
                pcd_ground_minor = o3d.io.read_point_cloud(f'{out_folder}pcd_ground_minor{SEQUENCE_NUM}_{cur_idx}.pcd')
                pcd_nonground_minor = o3d.io.read_point_cloud(f'{out_folder}pcd_nonground_minor{SEQUENCE_NUM}_{cur_idx}.pcd')
                
                print("load data")
                kitti_labels_orig = {}
                with np.load(f'{out_folder}kitti_labels_preprocessed{SEQUENCE_NUM}_{cur_idx}.npz') as data :
                        kitti_labels_orig['instance_ground'] = data['instance_ground']
                        kitti_labels_orig['instance_nonground'] = data['instance_nonground']
                        kitti_labels_orig['seg_nonground'] = data['seg_nonground']
                        kitti_labels_orig['seg_ground'] = data['seg_ground']
        
                with np.load(f'{out_folder}all_poses_' + str(SEQUENCE_NUM) + '_' + str(cur_idx) + '.npz') as data:
                        all_poses = data['all_poses']
                        T_pcd = data['T_pcd']
                        first_position = T_pcd[:3, 3]
                        
                print("pose downsample")
                if os.path.exists(f'{out_folder}subsampled_data{SEQUENCE_NUM}_{cur_idx}.npz') == False : 
                        poses, positions, \
                        sampled_indices_local, sampled_indices_global = subsample_and_extract_positions(all_poses,ind_start=ind_start,sequence_num=SEQUENCE_NUM,out_folder=out_folder,cur_idx=cur_idx)
                
                with np.load(f'{out_folder}subsampled_data{SEQUENCE_NUM}_{cur_idx}.npz') as data:
                        poses = data['poses']
                        positions = data['positions']
                        sampled_indices_local = list(data['sampled_indices_local'])
                        sampled_indices_global = list(
                         data['sampled_indices_global'])
                        
                
                
                print("chunk downsample")
                pcd_nonground_chunks, pcd_ground_chunks,\
                pcd_nonground_chunks_major_downsampling, pcd_ground_chunks_major_downsampling, \
                indices,indices_ground, center_positions, \
                center_ids, chunk_bounds, kitti_labels, _ = chunk_and_downsample_point_clouds(pcd_nonground_minor, pcd_ground_minor, T_pcd, positions, 
                                                                    first_position, sampled_indices_global, chunk_size=chunk_size, 
                                                                    overlap=overlap, major_voxel_size=major_voxel_size,kitti_labels=kitti_labels_orig)
                
                print("finished downsample")
                
                instances = np.hstack((kitti_labels_orig['instance_nonground'],kitti_labels_orig['instance_ground']))
                patchwise_indices = indices_per_patch(T_pcd, center_positions, positions, first_position, sampled_indices_global, chunk_size)
                out_data = []
                for sequence in range(0,len(center_ids))[:-1]:
                                        #try : 
                                        print("sequence",sequence)
                                        merged_chunk,file_name, pcd_chunk, pcd_chunk_ground,inliers, inliers_ground = ncuts_chunk(dataset,list(indices),pcd_nonground_chunks,pcd_ground_chunks,
                                                pcd_nonground_chunks_major_downsampling,
                                                pcd_nonground_minor,T_pcd,center_positions,center_ids,
                                                positions,first_position,list(sampled_indices_global),
                                                chunk_size=chunk_size,major_voxel_size=major_voxel_size,
                                                alpha=alpha,beta=beta,gamma=gamma,theta=theta,
                                                proximity_threshold=proximity_threshold,
                                                out_folder=out_folder_ncuts,ground_mode=False,sequence=sequence,
                                                patchwise_indices=patchwise_indices,ncuts_threshold=ncuts_threshold)
                                        
                                        '''
                                        try : 
                                                ground_out,file_name = ncuts_chunk(dataset,indices_ground,pcd_chunk_ground,None,
                                                        pcd_ground_chunks_major_downsampling,
                                                        pcd_ground_minor,T_pcd,center_positions,center_ids,
                                                        positions,first_position,sampled_indices_global,
                                                        chunk_size=chunk_size,major_voxel_size=major_voxel_size,
                                                        alpha=1.0,beta=0.0,gamma=1.0,theta=1.0,
                                                        proximity_threshold=1.0,
                                                        out_folder=out_folder_ncuts,ground_mode=True,sequence=sequence,
                                                        patchwise_indices=patchwise_indices,ncuts_threshold=0.001,norm=tarl_norm)
                                        except : 
                                                print("skip")
                                        '''
                                        
                                        
                                        #kitti_labels['ground']['panoptic'][sequence] = kitti_labels['ground']['panoptic'][sequence][inliers_ground]
                                        inst_ground = kitti_labels['ground']['instance'][sequence][inliers][inliers_ground]
                                        seg_ground = kitti_labels['ground']['semantic'][sequence][inliers][inliers_ground]
                                        
                                        name = file_name.split('/')[-1]
                                        
                                        cur_name = name.split('.')[0]
                                        
                                        #import pdb; pdb.set_trace()
                                        #kitti_chunk = color_pcd_by_labels(pcd_chunk,kitti_labels['nonground']['panoptic'][sequence].reshape(-1,),
                                        #                        colors=colors,gt_labels=kitti_labels_orig['panoptic_nonground'])
                                        
                                        kitti_chunk_instance = color_pcd_by_labels(pcd_chunk,kitti_labels['nonground']['instance'][sequence].reshape(-1,),
                                                                colors=colors,gt_labels=kitti_labels_orig['instance_nonground'])
                                        
                                        kitti_chunk_instance_ground = color_pcd_by_labels(pcd_chunk_ground,inst_ground.reshape(-1,),
                                                        colors=colors,gt_labels=instances)
                                        
                                        gt_pcd = kitti_chunk_instance + kitti_chunk_instance_ground
                                        pred_pcd = pcd_chunk  + pcd_chunk_ground
                                        
                                        kitti_semantics = np.hstack((kitti_labels['nonground']['semantic'][sequence].reshape(-1,),seg_ground.reshape(-1,)))
                                        
                                        unique_colors, labels_ncuts = np.unique(np.asarray(pred_pcd.colors), axis=0, return_inverse=True)
                                        unique_colors, labels_kitti = np.unique(np.asarray(gt_pcd.colors),axis=0, return_inverse=True)
                                        
                                        
                                        pts = np.asarray(gt_pcd.points) 
                                        points,labels_ncuts,labels_kitti = downsample_chunk(pts,labels_ncuts,labels_kitti)
                                        print("output",data_store_folder + name.split('.')[0])
                                        
                                        if labels_ncuts.shape[0] != points.shape[0] : 
                                                AssertionError 
                                        np.savez(data_store_folder + name.split('.')[0] +  '.npz',pts=points,ncut_labels=labels_ncuts,
                                                kitti_labels=labels_kitti,cluster_labels=np.zeros_like(labels_ncuts),semantic=kitti_semantics)
                                        
                                        del points, labels_ncuts, labels_kitti 
                                        del pred_pcd, gt_pcd, kitti_chunk_instance, pcd_chunk, pcd_chunk_ground
                                        gc.collect()
                                        #except : 
                                        #        print("error",center_ids[sequence])
                                        
                                        #kitti_chunk_instance_ground = color_pcd_by_labels(pcd_chunk_ground,kitti_labels['ground']['instance'][sequence].reshape(-1,))
                                        #o3d.io.write_point_cloud(out_dbscan + name, pcd_dbscan + pcd_chunk_ground, write_ascii=False, compressed=False, print_progress=False)
                                        #o3d.io.write_point_cloud(out_kitti + name, kitti_chunk + pcd_chunk_ground, write_ascii=False, compressed=False, print_progress=False)
                                        #o3d.io.write_point_cloud(out_kitti_instance + name, kitti_chunk_instance + pcd_chunk_ground, write_ascii=False, compressed=False, print_progress=False)
                                        
        
        
        
