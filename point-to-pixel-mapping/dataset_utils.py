from dataset.kitti_odometry_dataset import KittiOdometryDataset, KittiOdometryDatasetConfig
from dataset.filters.filter_list import FilterList
from dataset.filters.kitti_gt_mo_filter import KittiGTMovingObjectFilter
from dataset.filters.range_filter import RangeFilter
import os 
import numpy as np 
import open3d as o3d
from aggregate_pointcloud import aggregate_pointcloud
from chunk_generation import subsample_positions, chunks_from_pointcloud
from visualization_utils import * 

def create_kitti_odometry_dataset(dataset_path, sequence_num, cache=True, sam_folder_name="sam_pred_medium", 
                                correct_scan_calibration=True, range_min=3, range_max=25,ncuts_mode=True):
    

    if ncuts_mode : 
        filters = FilterList(
            [
                KittiGTMovingObjectFilter(
                    os.path.join(dataset_path, "sequences", "%.2d" % sequence_num, "labels")
                ),
                RangeFilter(range_min, range_max),
            ]
        )
    else : 
        filters = FilterList([])
        
    config_filtered = KittiOdometryDatasetConfig(
        cache=cache,
        dataset_path=dataset_path,
        sam_folder_name=sam_folder_name,
        correct_scan_calibration=correct_scan_calibration,
        filters=filters,
    )

    dataset = KittiOdometryDataset(config_filtered, sequence_num)
    return dataset

def process_and_save_point_clouds(dataset, ind_start, ind_end, ground_segmentation_method="patchwork", icp=True, 
                                minor_voxel_size=0.05, major_voxel_size=0.35, out_folder=None, sequence_num=7):
    
    if os.path.exists(out_folder) == False : 
            os.makedirs(out_folder)
    
    pcd_ground, pcd_nonground, all_poses, T_pcd,kitti_labels = aggregate_pointcloud(dataset, ind_start, 
                                                ind_end, ground_segmentation=ground_segmentation_method, 
                                                icp=icp)

    
    # Saving point clouds and poses
    if pcd_ground is not None : 
        o3d.io.write_point_cloud(f'{out_folder}ground{sequence_num}.pcd', pcd_ground, write_ascii=False, compressed=False, print_progress=False)
    o3d.io.write_point_cloud(f'{out_folder}non_ground{sequence_num}.pcd', pcd_nonground, write_ascii=False, compressed=False, print_progress=False)
    
    np.savez(f'{out_folder}all_poses.npz', all_poses=all_poses, T_pcd=T_pcd)
    np.savez(f'{out_folder}kitti_labels.npz',seg_ground=np.vstack(kitti_labels['seg_ground']),
            seg_nonground=np.vstack(kitti_labels['seg_nonground']),
            instance_ground=np.vstack(kitti_labels['instance_ground']),
            instance_nonground=np.vstack(kitti_labels['instance_nonground']),
            panoptic_ground=np.vstack(kitti_labels['panoptic_ground']),
            panoptic_nonground=np.vstack(kitti_labels['panoptic_nonground']))
    
    return kitti_labels
    


def load_and_downsample_point_clouds(out_folder, sequence_num, minor_voxel_size=0.05,ground_mode=True):
    # Load saved data
    with np.load(f'{out_folder}all_poses.npz') as data:
        all_poses = data['all_poses']
        T_pcd = data['T_pcd']
        first_position = T_pcd[:3, 3]

    # Load point clouds
    if ground_mode is not None : 
        pcd_ground = o3d.io.read_point_cloud(f'{out_folder}ground{sequence_num}.pcd')
    pcd_nonground = o3d.io.read_point_cloud(f'{out_folder}non_ground{sequence_num}.pcd')

    # Downsampling
    
    #pcd_ground_minor = pcd_ground.voxel_down_sample(voxel_size=minor_voxel_size)
    pcd_ground_minor, trace_ground, _ = pcd_ground.voxel_down_sample_and_trace(minor_voxel_size, pcd_ground.get_min_bound(), 
                                                                        pcd_ground.get_max_bound(), False)

    
    #pcd_nonground_minor = pcd_nonground.voxel_down_sample(voxel_size=minor_voxel_size)
    pcd_nonground_minor, trace_nonground, _ = pcd_nonground.voxel_down_sample_and_trace(minor_voxel_size, pcd_nonground.get_min_bound(), 
                                                                        pcd_nonground.get_max_bound(), False)

    #flattened_trace = trace_nonground.flatten()
    #trace_nonground = np.unique(flattened_trace)
    
    #flattened_traceground = trace_ground.flatten()
    #trace_ground= np.unique(flattened_traceground)
    
    
    kitti_data = {}
    with np.load(f'{out_folder}kitti_labels.npz') as data : 
        kitti_data['panoptic_ground'] = data['panoptic_ground'][trace_ground]
        kitti_data['panoptic_nonground'] = data['panoptic_nonground'][trace_nonground]
        kitti_data['seg_ground'] = data['seg_ground'][trace_ground]
        kitti_data['seg_nonground'] = data['seg_nonground'][trace_nonground]
        kitti_data['instance_ground'] = data['instance_ground'][trace_ground]
        kitti_data['instance_nonground'] = data['instance_nonground'][trace_nonground]
        
    #o3d.visualization.draw_geometries([color_pcd_by_labels(pcd_nonground_minor,kitti_data['panoptic_nonground'])])
    panoptic_non_ground = color_pcd_by_labels(pcd_nonground_minor,kitti_data['panoptic_nonground'])
    _, kitti_data['panoptic_nonground'] = np.unique(np.asarray(panoptic_non_ground.colors), axis=0, return_inverse=True)
    
    return pcd_ground_minor, pcd_nonground_minor, all_poses, T_pcd, first_position,kitti_data

def subsample_and_extract_positions(all_poses, voxel_size=1, ind_start=0):

    # Extracting positions from poses
    all_positions = [tuple(p[:3, 3]) for p in all_poses]

    # Performing subsampling
    sampled_indices_local = list(subsample_positions(all_positions, voxel_size=voxel_size))
    sampled_indices_global = list(subsample_positions(all_positions, voxel_size=voxel_size) + ind_start)

    # Selecting a subset of poses and positions
    poses = np.array(all_poses)[sampled_indices_local]
    positions = np.array(all_positions)[sampled_indices_local]

    return poses, positions, sampled_indices_local, sampled_indices_global

def chunk_and_downsample_point_clouds(pcd_nonground_minor, pcd_ground_minor, T_pcd, positions, first_position, 
                                    sampled_indices_global, chunk_size=np.array([25, 25, 25]), 
                                    overlap=3, major_voxel_size=0.35,kitti_labels=None):
    # Creating chunks
    pcd_nonground_chunks, indices, center_positions, center_ids, chunk_bounds, kitti_out = chunks_from_pointcloud(pcd_nonground_minor, T_pcd, positions, 
                                                                            first_position, sampled_indices_global, chunk_size, overlap,labels=kitti_labels)
    
    pcd_ground_chunks, indices_ground, _, _, _ , kitti_out_ground = chunks_from_pointcloud(pcd_ground_minor, T_pcd, positions, 
                                                        first_position, sampled_indices_global, chunk_size, overlap,labels=kitti_labels,ground=True)
        
    # Downsampling the chunks and printing information
    kitti_labels = {'nonground':kitti_out,'ground':kitti_out_ground}
    pcd_nonground_chunks_major_downsampling = []
    pcd_ground_chunks_major_downsampling = []
    for (ground,nonground) in zip(pcd_ground_chunks,pcd_nonground_chunks):
        downsampled_nonground = nonground.voxel_down_sample(voxel_size=major_voxel_size)
        downsampled_ground = ground.voxel_down_sample(voxel_size=major_voxel_size)
        print("Downsampled from", np.asarray(nonground.points).shape, "to", np.asarray(downsampled_nonground.points).shape, "points (non-ground)")
        print("Downsampled from", np.asarray(ground.points).shape, "to", np.asarray(downsampled_ground.points).shape, "points (ground)")
        
        
        pcd_nonground_chunks_major_downsampling.append(downsampled_nonground)
        pcd_ground_chunks_major_downsampling.append(downsampled_ground)

    return pcd_nonground_chunks,pcd_ground_chunks,pcd_nonground_chunks_major_downsampling, \
        pcd_ground_chunks_major_downsampling, \
        indices, indices_ground,center_positions, center_ids, chunk_bounds, kitti_labels
