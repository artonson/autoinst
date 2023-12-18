import open3d as o3d
import numpy as np
import os
from scipy.spatial.distance import cdist
from sam_label_distace import sam_label_distance
from chunk_generation import get_indices_feature_reprojection
from point_cloud_utils import get_pcd, transform_pcd, remove_isolated_points, get_subpcd, get_statistical_inlier_indices, merge_chunks_unite_instances, kDTree_1NN_feature_reprojection
from aggregate_pointcloud import aggregate_pointcloud
from visualization_utils import generate_random_colors, color_pcd_by_labels
from sam_label_distace import sam_label_distance
from chunk_generation import subsample_positions, chunks_from_pointcloud, indices_per_patch, tarl_features_per_patch, image_based_features_per_patch, dinov2_mean, get_indices_feature_reprojection
from normalized_cut import normalized_cut

def ncuts_chunk(dataset,indices,pcd_nonground_chunks, pcd_ground_chunks, 
                        pcd_nonground_chunks_major_downsampling, 
                        pcd_nonground_minor,
                        T_pcd, 
                        center_positions,center_ids, 
                        positions, first_position, sampled_indices_global, chunk_size, 
                        major_voxel_size=0.35, alpha=1, beta=0, gamma=0, 
                        theta=0,proximity_threshold=1, ncuts_threshold=0.08,
                        out_folder='test_data/',ground_mode=True,sequence=None,
                        patchwise_indices=None):
        
                print("start of sequence",sequence)
                first_id = patchwise_indices[sequence][0]
                center_id = center_ids[sequence]
                center_position = center_positions[sequence]
                chunk_indices = indices[sequence]

                cam_indices_global, hpr_mask_indices = get_indices_feature_reprojection(sampled_indices_global, first_id, adjacent_frames=(16,13)) 
                
                pcd_chunk = pcd_nonground_chunks[sequence]
                if ground_mode == False  : 
                        pcd_ground_chunk = pcd_ground_chunks[sequence]
                chunk_major = pcd_nonground_chunks_major_downsampling[sequence]

                points_major = np.asarray(chunk_major.points)
                num_points_major = points_major.shape[0]   

                print(num_points_major, "points in downsampled chunk (major)")

                tarl_features = tarl_features_per_patch(dataset, chunk_major, center_id, T_pcd, center_position, sampled_indices_global, chunk_size, major_voxel_size)

                cams = ["cam2", "cam3"]

                sam_features_minor, chunk_minor = image_based_features_per_patch(dataset, pcd_nonground_minor, chunk_indices, 
                                                                        T_pcd, cam_indices_global, cams, cam_id=0, 
                                                                        hpr_radius=2000, dino=False, rm_perp=0.0)
                #point2dino
                #dinov2_features_minor = dinov2_mean(point2dino)
                
                sam_features_major = -1 * np.ones((num_points_major, sam_features_minor.shape[1]))
                #dinov2_features_major = np.zeros((num_points_major, dinov2_features_minor.shape[1])) 

                sam_features_major = kDTree_1NN_feature_reprojection(sam_features_major, chunk_major, sam_features_minor, chunk_minor)
                #dinov2_features_major = kDTree_1NN_feature_reprojection(dinov2_features_major, chunk_major, dinov2_features_minor, chunk_minor)

                zero_rows = np.sum(~np.array(sam_features_major).any(1))
                ratio = zero_rows / num_points_major

                if ratio > 0.3:
                        print("The ratio of points without image-based features is", ratio, ". Skipping this chunk.")
                        return

                spatial_distance = cdist(points_major, points_major)
                #dinov2_distance = cdist(dinov2_features_major, dinov2_features_major)
                tarl_distance = cdist(tarl_features, tarl_features)
                sam_edge_weights, mask = sam_label_distance(sam_features_major, spatial_distance, proximity_threshold, beta)
                spatial_edge_weights = mask * np.exp(-alpha * spatial_distance)
                #dinov2_edge_weights = mask * np.exp(-gamma * dinov2_distance)
                tarl_edge_weights = mask * np.exp(-theta * tarl_distance)

                A = sam_edge_weights
                print("Adjacency Matrix built")

                # Remove isolated points
                chunk_major, A = remove_isolated_points(chunk_major, A)
                print(num_points_major - np.asarray(chunk_major.points).shape[0], "isolated points removed")
                num_points_major = np.asarray(chunk_major.points).shape[0]

                print("Start of normalized Cuts")
                grouped_labels = normalized_cut(A, np.arange(num_points_major), T = ncuts_threshold)
                num_groups = len(grouped_labels)
                print("There are", num_groups, "cut regions")

                sorted_groups = sorted(grouped_labels, key=lambda x: len(x))
                num_points_top3 = np.sum([len(g) for g in sorted_groups[-3:]])
                top3_ratio = num_points_top3 / num_points_major
                print("Ratio of points in top 3 groups:", top3_ratio)

                random_colors = generate_random_colors(600)

                pcd_color = np.zeros((num_points_major, 3))
                

                for i, s in enumerate(grouped_labels):
                        for j in s:
                                pcd_color[j] = np.array(random_colors[i]) / 255


                pcd_chunk.paint_uniform_color([0, 0, 0])
                colors = kDTree_1NN_feature_reprojection(np.asarray(pcd_chunk.colors), pcd_chunk, pcd_color, chunk_major)
                pcd_chunk.colors = o3d.utility.Vector3dVector(colors)

                
                if ground_mode == False : 
                        inliers = get_statistical_inlier_indices(pcd_ground_chunk)
                        ground_inliers = get_subpcd(pcd_ground_chunk, inliers)
                        mean_hight = np.mean(np.asarray(ground_inliers.points)[:,2])
                        cut_hight = get_subpcd(ground_inliers, np.where(np.asarray(ground_inliers.points)[:,2] < (mean_hight + 0.2))[0])
                        cut_hight.paint_uniform_color([0, 0, 0])
                        merged_chunk = pcd_chunk + cut_hight
                else : 
                        merged_chunk = pcd_chunk 

                index_file = str(center_id).zfill(6) + '.pcd'
                file = os.path.join(out_folder, index_file)
                return merged_chunk,file
               
def get_merge_pcds(out_folder_ncuts):
        point_clouds = []

        # List all files in the folder
        files = os.listdir(out_folder_ncuts)
        files.sort()

        # Filter files with a .pcd extension
        pcd_files = [file for file in files if file.endswith(".pcd")][:3]
        print(pcd_files)
        # Load each point cloud and append to the list
        for pcd_file in pcd_files:
                file_path = os.path.join(out_folder_ncuts, pcd_file)
                point_cloud = o3d.io.read_point_cloud(file_path)
                point_clouds.append(point_cloud)
        return point_clouds
        
                
def kDTree_1NN_feature_reprojection_colors(features_to, pcd_to, features_from, pcd_from, 
                                labels=None,max_radius=None, no_feature_label=[1,0,0]):
    '''
    Args:
        pcd_from: point cloud to be projected
        pcd_to: point cloud to be projected to
        search_method: search method ("radius", "knn")
        search_param: search parameter (radius or k)
    Returns:
        features_to: features projected on pcd_to
    '''
    from_tree = o3d.geometry.KDTreeFlann(pcd_from)
    labels_output = np.ones(np.asarray(pcd_to.points).shape[0],) * -1
    unique_colors = list(np.unique(np.asarray(pcd_from.colors),axis=0)) 
    
    for i, point in enumerate(np.asarray(pcd_to.points)):

        [_, idx, _] = from_tree.search_knn_vector_3d(point, 1)
        if max_radius is not None:
            if np.linalg.norm(point - np.asarray(pcd_from.points)[idx[0]]) > max_radius:
                features_to[i,:] = no_feature_label
                if labels is not None : 
                    labels[i] = -1
                    labels_output[i] = -1 
            else:
                features_to[i,:] = features_from[idx[0]]
                labels_output[i] = np.where((unique_colors == features_from[idx[0]]).all(axis=1))[0]
        else:
            features_to[i,:] = features_from[idx[0]]
            labels_output[i] = np.where((unique_colors == features_from[idx[0]]).all(axis=1))[0]
        
        
    if labels is not None : 
        return features_to,labels, labels_output
    else : 
        return features_to, None, labels_output