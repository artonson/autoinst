import numpy as np

def sam_label_distance(sam_features, spatial_distance, proximity_threshold, num_points, beta):

    mask = np.where(spatial_distance <= proximity_threshold)
    not_ordered_dist_mat_emb = np.sum(np.abs(sam_features[mask[0]] - sam_features[mask[1]]), axis=1)

    embedding_dist_matrix = np.zeros(spatial_distance.shape)
    for k, (i, j) in enumerate(zip(*mask)):
        embedding_dist_matrix[i, j] = not_ordered_dist_mat_emb[k][0, 0]

    mask = np.where(spatial_distance <= proximity_threshold, 1, 0)

    label_distance = mask * np.exp(-beta * embedding_dist_matrix)

    return label_distance, mask