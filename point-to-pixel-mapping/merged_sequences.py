from dataset_utils import get_data_from_dataset
from image_utils import downsample
from hidden_points_removal import hidden_point_removal_o3d, hidden_point_removal_biasutti
from point_cloud_utils import transform_pcd, filter_points_from_dict, filter_points_from_list, get_pcd, point_to_label, change_point_indices
from point_to_pixels import point_to_pixel
from visualization_utils import color_pcd_with_labels, visualize_associations_in_img
from merge_pointclouds import build_associations, apply_associations_to_dict, merge_label_predictions, merge_pointclouds, build_associations_across_timesteps
import copy 
import open3d as o3d 
import numpy as np 
from visualization_utils import * 

def merged_sequence(dataset, start_index, sequence_length, pcd_chunk,labelling=True):
    '''
    Args:
        dataset:            dataset object
        start_index:        index of first point cloud
        sequence_length:    number of point clouds to merge
        labelling:          if True, the point clouds are merged with labels
    Returns:
        label_history:      list of propagated label maps of each time step (empty if labelling=False)
        pcd_merge:          merged point cloud
    '''

    left_cam = "cam2"
    right_cam = "cam3"
    first_pose = dataset.get_pose(start_index)
    pcd_merge = None
    label_history = []

    if labelling:
        last_label = None

        for i in range(sequence_length):
            points_index = start_index + i
            if points_index >= len(dataset): 
                break 

            pcd, images, labels, T_matrices, K_matrices, T_lidar2world = get_data_from_dataset(dataset, points_index, left_cam, right_cam,pcd_chunk)
            
            pcd_world = transform_pcd(copy.deepcopy(pcd),T_lidar2world)
            
            pcd_camframes = (transform_pcd(copy.deepcopy(pcd), T_matrices[0]), transform_pcd(copy.deepcopy(pcd), T_matrices[1]))    
            
            #test_pcd = o3d.geometry.PointCloud()
            #test_pcd.points = o3d.utility.Vector3dVector(pcd)
            #pcd_chunk.paint_uniform_color([0,0,1])
            #o3d.visualization.draw_geometries([test_pcd,pcd_chunk])

            hpr_mask_camframes = (hidden_point_removal_o3d(pcd_camframes[0], camera=[0,0,0], radius_factor=400), 
                                hidden_point_removal_o3d(pcd_camframes[1], camera=[0,0,0], radius_factor=400))


            pcd_camframes_hpr = (pcd_camframes[0][hpr_mask_camframes[0]], pcd_camframes[1][hpr_mask_camframes[1]])

            point_to_pixel_dicts = (point_to_pixel(pcd_camframes_hpr[0], K_matrices[0], images[0].shape[0], images[0].shape[1]),
                                    point_to_pixel(pcd_camframes_hpr[1], K_matrices[1], images[1].shape[0], images[1].shape[1]))

            point_to_label_dicts_camframes = (point_to_label(point_to_pixel_dicts[0], labels[0]),
                                            point_to_label(point_to_pixel_dicts[1], labels[1]))

            point_to_label_dicts_camframes_mapped = (change_point_indices(point_to_label_dicts_camframes[0], hpr_mask_camframes[0]),
                                                    change_point_indices(point_to_label_dicts_camframes[1], hpr_mask_camframes[1]))

            associations_rl = build_associations(point_to_label_dicts_camframes_mapped[1], point_to_label_dicts_camframes_mapped[0])

            point_to_label_dict_rightcam_associated = apply_associations_to_dict(point_to_label_dicts_camframes_mapped[1], associations_rl)

            merged_labels_rl = merge_label_predictions(point_to_label_dict_rightcam_associated, point_to_label_dicts_camframes_mapped[0], method='iou')

            # After id's of left and right cam of this time step are now associated, we need to associate this time step to the last one
            if last_label is not None: 
                last_leftlabel_downsampled = downsample(last_label, 0.25)
                leftlabel_downsampled = downsample(labels[0], 0.25)
                association_t0_t1 = build_associations_across_timesteps(leftlabel_downsampled, last_leftlabel_downsampled)
                merged_labels_rl = apply_associations_to_dict(merged_labels_rl, association_t0_t1)

                last_label = visualize_associations_in_img(labels[0], association_t0_t1)
            else:
                # If this is the first time step, we just keep the labels
                last_label = labels[0]

            label_history.append(last_label)
            
            #o3d.visualization.draw_geometries([pcd_merge,pcd_chunk])

            
            pcd_merged_labels_rl = filter_points_from_dict(pcd_world, merged_labels_rl)
            pcd_merged_labels_rl = color_pcd_with_labels(pcd_merged_labels_rl, merged_labels_rl)
            
            
            #test_pcd = o3d.geometry.PointCloud()
            #test_pcd.points = o3d.utility.Vector3dVector(pcd)
            #pcd_chunk.paint_uniform_color([0,0,1])
            #o3d.visualization.draw_geometries([test_pcd,pcd_chunk,pcd_merged_labels_rl])

            if pcd_merge is None:
                pcd_merge = pcd_merged_labels_rl
                #o3d.visualization.draw_geometries([pcd_merge,pcd_chunk])
            else:
                pose = dataset.get_pose(points_index)
                pcd_merge = merge_pointclouds(pcd_merge, pcd_merged_labels_rl, first_pose, pose)
                
        unique_colors, labels_cur = np.unique(np.asarray(pcd_merge.colors), axis=0, return_inverse=True)
        pcd_merge = color_pcd_by_labels(pcd_merge,labels_cur + 3)
        
    else : 
        for i in range(sequence_length):
            points_index = start_index + i
            if points_index >= len(dataset): 
                break 

            pcd, images, labels, T_matrices, K_matrices, T_lidar2world = get_data_from_dataset(dataset, points_index, left_cam, right_cam,pcd_chunk)
            
            pcd_world = transform_pcd(copy.deepcopy(pcd),T_lidar2world)
            
            pcd_camframes = (transform_pcd(copy.deepcopy(pcd), T_matrices[0]), transform_pcd(copy.deepcopy(pcd), T_matrices[1]))    
            
            #test_pcd = o3d.geometry.PointCloud()
            #test_pcd.points = o3d.utility.Vector3dVector(pcd)
            #pcd_chunk.paint_uniform_color([0,0,1])
            #o3d.visualization.draw_geometries([test_pcd,pcd_chunk])

            hpr_mask_camframes = (hidden_point_removal_o3d(pcd_camframes[0], camera=[0,0,0], radius_factor=400), 
                                hidden_point_removal_o3d(pcd_camframes[1], camera=[0,0,0], radius_factor=400))


            pcd_camframes_hpr = (pcd_camframes[0][hpr_mask_camframes[0]], pcd_camframes[1][hpr_mask_camframes[1]])

            point_to_pixel_dicts = (point_to_pixel(pcd_camframes_hpr[0], K_matrices[0], images[0].shape[0], images[0].shape[1]),
                                    point_to_pixel(pcd_camframes_hpr[1], K_matrices[1], images[1].shape[0], images[1].shape[1]))
            
            point_to_pixel_dicts_mapped = (change_point_indices(point_to_pixel_dicts[0], hpr_mask_camframes[0]),
                                        change_point_indices(point_to_pixel_dicts[1], hpr_mask_camframes[1]))
            
            merged_points = set(point_to_pixel_dicts_mapped[0].keys()) | set(point_to_pixel_dicts_mapped[1].keys())
            filtered_pcd = get_pcd(filter_points_from_list(pcd_world, merged_points))

            if pcd_merge is None:
                pcd_merge = filtered_pcd
            else:
                pose = dataset.get_pose(points_index)
                pcd_merge = merge_pointclouds(pcd_merge, filtered_pcd, first_pose, pose)

    
            
    return label_history, pcd_merge