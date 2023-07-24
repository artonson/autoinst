from dataset_utils import get_data_from_dataset
from image_utils import downsample
from hidden_points_removal import hidden_point_removal_o3d, hidden_point_removal_biasutti
from point_cloud_utils import transform_pcd, filter_points_from_dict, get_pcd, point_to_label, change_point_indices
from point_to_pixels import point_to_pixel
from visualization_utils import color_pcd_with_labels, visualize_associations_in_img
from merge_pointclouds import build_associations, apply_associations_to_dict, merge_label_predictions, merge_pointclouds, build_associations_across_timesteps

def merged_sequence(dataset, start_index, sequence_length):
    left_cam = "cam2"
    right_cam = "cam3"
    first_pose = dataset.get_pose(start_index)

    pcd_merge = None
    last_label = None
    label_history = []

    for i in range(sequence_length):
        points_index = start_index + i

        pcd, images, labels, T_matrices, K_matrices = get_data_from_dataset(dataset, points_index, left_cam, right_cam)
        
        pcd_camframes = (transform_pcd(pcd, T_matrices[0]), transform_pcd(pcd, T_matrices[1]))

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
        
        
        pcd_merged_labels_rl = filter_points_from_dict(pcd, merged_labels_rl)
        pcd_merged_labels_rl = color_pcd_with_labels(pcd_merged_labels_rl, merged_labels_rl)

        if pcd_merge is None:
            pcd_merge = pcd_merged_labels_rl
        else:
            pose = dataset.get_pose(points_index)
            pcd_merge = merge_pointclouds(pcd_merge, pcd_merged_labels_rl, first_pose, pose)
        

    return label_history, pcd_merge