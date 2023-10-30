import numpy as np

def sam_label_distance(sam_features, spatial_distance, proximity_threshold, beta):

    mask = np.where(spatial_distance <= proximity_threshold)
    
    # Initialize the distance matrix with zeros
    num_points, num_views = sam_features.shape
    distance_matrix = np.zeros((num_points, num_points))

    # Iterate over rows (points)
    for (point1, point2) in (zip(*mask)):
        view_counter = 0
        for view in range(num_views):
            instance_id1 = sam_features[point1, view]
            instance_id2 = sam_features[point2, view]

            if instance_id1 != -1 and instance_id2 != -1:
                view_counter += 1
                if instance_id1 != instance_id2:
                    distance_matrix[point1, point2] += 1
        if view_counter:
            distance_matrix[point1, point2] /= view_counter
            
    mask = np.where(spatial_distance <= proximity_threshold, 1, 0)
    label_distance = mask * np.exp(-beta * distance_matrix)

    return label_distance, mask