import numpy as np
from point_cloud_utils import transform_pcd, get_pcd
import open3d as o3d

def build_associations(point_to_label_1: dict, point_to_label_2: dict):
    '''
    Args:
        point_to_id1: dict that maps point indices to instance label color
        point_to_id2: dict that maps point indices to instance label color
    Returns:
        associations: dict that maps original label colors from dict1 to associated label colors from dict2
    '''

    unique_ids1 = set(point_to_label_1.values())
    unique_ids2 = set(point_to_label_2.values())

    label_to_point_1 = {}
    label_to_point_2 = {}

    for id1 in unique_ids1:
        label_to_point_1[id1] = [pixel for pixel, value in point_to_label_1.items() if value == id1]

    for id2 in unique_ids2:
        label_to_point_2[id2] = [pixel for pixel, value in point_to_label_2.items() if value == id2]

    # Create a dictionary to store the associations
    associations = {}

    for id1 in unique_ids1:
        # If no association will be found, we just keep the original ID
        best_id2 = None
        best_iou = 0.0

        indices1 = label_to_point_1[id1]

        for id2 in unique_ids2:
            indices2 = label_to_point_2[id2]
            intersection = len(set(indices1) & set(indices2))
            union = len(set(indices1) | set(indices2))
            iou = intersection / union

            if iou > best_iou:
                best_id2 = id2
                best_iou = iou
        
        associations[id1] = best_id2

    return associations


def apply_associations_to_dict(point_to_label: dict, associations: dict):
    '''
    Args:
        point_to_label: dict that maps point indices to instance label color
        associations: dict that maps original label colors to associated instance label colors
    Returns:
        point_to_label_associated: dict that maps point indices to ssociated instance label colors
    '''

    point_to_label_associated = {}
    
    for index, color in point_to_label.items():
        
        association = associations.get(color)
        if association is not None:
            point_to_label_associated[index] = association
    
    return point_to_label_associated

def merge_label_predictions(point_to_label_1: dict, point_to_label_2: dict, method="iou"):
    '''
    Args:
        point_to_label_1:   dict that maps point indices to instance label color
        point_to_label_2:   dict that maps point indices to instance label color
        method:             method to merge the labels (iou or greedy)
    Returns:
        merged_dict:        dict that maps point indices to merged instance label color

    Important: point_to_label_1 and point_to_label_2 must have been associated beforehand, i.e. instance label colors 
    '''

    unique_ids1 = set(point_to_label_1.values())
    unique_ids2 = set(point_to_label_2.values())

    label_to_point_1 = {}
    label_to_point_2 = {}

    for id1 in unique_ids1:
        label_to_point_1[id1] = [pixel for pixel, value in point_to_label_1.items() if value == id1]

    for id2 in unique_ids2:
        label_to_point_2[id2] = [pixel for pixel, value in point_to_label_2.items() if value == id2]

    # Create a new dictionary to store the merged associations
    merged_dict = {}

    # Combine the keys from both dictionaries
    all_points = set(point_to_label_1.keys()) | set(point_to_label_2.keys())

    if method == "iou":

        for point in all_points:
            # Get the instance IDs assigned to the current key/index
            instance_id1 = point_to_label_1.get(point)
            instance_id2 = point_to_label_2.get(point)

            if instance_id1 is None:
                # If the key/index is not present in dict1, use the instance ID from dict2
                merged_dict[point] = instance_id2
            elif instance_id2 is None:
                # If the key/index is not present in dict2, use the instance ID from dict1
                merged_dict[point] = instance_id1
            elif instance_id1 == instance_id2:
                # If both instance IDs are the same, use either one
                merged_dict[point] = instance_id1
            else:
                # Maximize IoU
                indices1 = label_to_point_1[instance_id1]
                indices2 = label_to_point_2[instance_id2]

                union = len(set(indices1) | set(indices2))
                
                # Does this criterion make sense? It just picks the instance ID with more points
                iou_id1 = len(indices1) / union
                iou_id2 = len(indices2) / union

                # Assign the instance ID with the higher IoU to the merged dictionary
                if iou_id1 >= iou_id2:
                    merged_dict[point] = instance_id1
                else:
                    merged_dict[point] = instance_id2

    elif method == "greedy":

        for point in all_points:
            # Get the instance IDs assigned to the current key/index
            instance_id1 = point_to_label_1.get(point)
            instance_id2 = point_to_label_2.get(point)

            if instance_id1 is None:
                # If the key/index is not present in dict1, use the instance ID from dict2
                if point not in merged_dict:
                    merged_dict[point] = instance_id2
            elif instance_id2 is None:
                # If the key/index is not present in dict2, use the instance ID from dict1
                if point not in merged_dict:
                    merged_dict[point] = instance_id1
            elif instance_id1 == instance_id2:
                # If both instance IDs are the same, use either one
                if point not in merged_dict:
                    merged_dict[point] = instance_id1
            else:
                if point not in merged_dict:
                    # If they are different, we assign all points with instance_id2 to instance_id1
                    indices1 = label_to_point_1[instance_id1]
                    indices2 = label_to_point_2[instance_id2]

                    union = (set(indices1) | set(indices2))

                    for index in union:
                        if index not in merged_dict:
                            merged_dict[index] = instance_id1

    return merged_dict


def merge_pointclouds(pcd_to, pcd_from, pose_to: np.array, pose_from: np.array):
    '''
    Merges two pointclouds by transforming the second pointcloud into the coordinate system of the first pointcloud.
    Args:
        pcd_to:                 Pointcloud to which the second pointcloud will be merged
        pcd_from:               Pointcloud which will be merged into the first pointcloud
        pose_to (np.array):     Pose of the first pointcloud, that projects from timestep i to timestep 0 (kitty defintion)
        pose_from (np.array):   Pose of the second pointcloud, that projects from timestep j to timestep 0 (kitty defintion)
    Returns:
        merged_pcd:          Merged pointcloud
    '''
    points_to = np.asarray(pcd_to.points)
    points_from = np.asarray(pcd_from.points)

    # Transform points_from into coordinate system of t=0
    #points_from_t0 = transform_pcd(points_from, pose_from)

    # Transform points_from_t0 into coordinate system of points_to
    #pose_to_inv = np.linalg.inv(pose_to)
    #points_from_transformed = transform_pcd(points_from_t0, pose_to_inv)

    merged_points = np.vstack([points_to, points_from])
    merged_pcd = get_pcd(merged_points)

    if pcd_to.has_colors() and pcd_from.has_colors():
        merged_colors = np.vstack([np.asarray(pcd_to.colors), np.asarray(pcd_from.colors)])
        merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    return merged_pcd


def build_associations_across_timesteps(label1: np.array, label2: np.array, upper_threshold=0.9, lower_threshold=0.4):
    '''
    Args:
        label1: np.array that contains the label predictions for timestep t
        label2: np.array that contains the label predictions for timestep t-1
        upper_threshold: float that specifies the upper threshold for the IoU
        lower_threshold: float that specifies the lower threshold for the IoU
    Returns:
        associations: dict that maps label colors timestep t to label colors from timestep t-1
    '''
    
    pixel_to_id1 = {}
    pixel_to_id2 = {}

    for i in range(label1.shape[0]):
        for j in range(label1.shape[1]):
            pixel_to_id1[(i,j)] = tuple(label1[i,j] / 255.0)
            pixel_to_id2[(i,j)] = tuple(label2[i,j] / 255.0)

    # Calculate the intersection over union (IoU) for each unique pair of instance IDs
    unique_ids1 = set(pixel_to_id1.values())
    unique_ids2 = set(pixel_to_id2.values())

    id_to_pixel1 = {}
    id_to_pixel2 = {}

    for id1 in unique_ids1:
        id_to_pixel1[id1] = [pixel for pixel, value in pixel_to_id1.items() if value == id1]

    for id2 in unique_ids2:
        id_to_pixel2[id2] = [pixel for pixel, value in pixel_to_id2.items() if value == id2]


    # Create a dictionary to store the associations
    associations = {}

    for id1 in unique_ids1:
        # If no association will be found, we just keep the original ID
        best_id2 = None
        best_iou = 0.0

        indices1 = id_to_pixel1[id1]

        for id2 in unique_ids2:
            indices2 = id_to_pixel2[id2]
            intersection = len(set(indices1) & set(indices2))
            union = len(set(indices1) | set(indices2))
            iou = intersection / union

            if iou > best_iou:
                best_id2 = id2
                best_iou = iou
        
        indices_best_id2 = id_to_pixel2[best_id2]

        # We keep the ID of the instance that is smaller -> allows to integrate new class IDs that come up in the background
        intersection = set(indices1) & set(indices_best_id2)
        if len(intersection) / len(indices1) > upper_threshold and len(intersection) / len(indices_best_id2) < lower_threshold:
            associations[id1] = id1
        else:
            associations[id1] = best_id2

    return associations