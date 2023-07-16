def build_associations(dict1: dict, dict2: dict):
    '''
    Args:
        dict1: dict that maps point indices to instance label color
        dict2: dict that maps point indices to instance label color
    Returns:
        associations: dict that maps original label colors from dict1 to associated label colors from dict2
    '''

    # Create a dictionary to store the associations
    associations = {}

    # Calculate the intersection over union (IoU) for each unique pair of instance IDs
    unique_ids1 = set(dict1.values())
    unique_ids2 = set(dict2.values())

    for id1 in unique_ids1:
        # If no association will be found, we just keep the original ID
        best_id2 = None
        best_iou = 0.0


        for id2 in unique_ids2:
            indices1 = [index for index, value in dict1.items() if value == id1]
            indices2 = [index for index, value in dict2.items() if value == id2]
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
        
        association = associations[color]
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
                indices1 = [index for index, value in point_to_label_1.items() if value == instance_id1]
                indices2 = [index for index, value in point_to_label_2.items() if value == instance_id2]

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
                    indices1 = [index for index, value in point_to_label_1.items() if value == instance_id1]
                    indices2 = [index for index, value in point_to_label_2.items() if value == instance_id2]

                    union = (set(indices1) | set(indices2))

                    for index in union:
                        if index not in merged_dict:
                            merged_dict[index] = instance_id1

    return merged_dict