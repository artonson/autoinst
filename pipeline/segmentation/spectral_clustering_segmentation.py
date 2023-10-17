import numpy as np
import open3d as o3d

from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from segmentation.abstract_segmentation import AbstractSegmentation
from reprojection.reproject_pointcloud import reproject_points_to_label, merge_associations

class SpectralClusteringSegmentation(AbstractSegmentation):
    def __init__(self, dataset, n_clusters, lower_bound, upper_bound):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __masks_to_image(self, masks):
        image_labels = np.zeros(masks[0]['segmentation'].shape)
        for i, mask in enumerate(masks):
            image_labels[mask['segmentation']] = i + 1

        return image_labels

    def __get_feature_diff(self, dist_matrix, association_matrix, num_points, proximity_threshold, beta):
        mask = np.where(dist_matrix <= proximity_threshold)
        not_ordered_dist_mat_emb = np.sum(np.abs(association_matrix[mask[0]] - association_matrix[mask[1]]), axis=1)

        embedding_dist_matrix = np.zeros(dist_matrix.shape)
        for k, (i, j) in enumerate(zip(*mask)):
            embedding_dist_matrix[i, j] = not_ordered_dist_mat_emb[k][0, 0]
        
        mask = np.where(dist_matrix <= proximity_threshold, 1, 0)
        feature_diff = (mask - np.eye(num_points)) * np.exp(-beta * embedding_dist_matrix)

        return feature_diff

    def segment_instances(self, index):
        points_full = self.dataset.get_point_cloud(index)
        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(points_full)
        labels_full = np.zeros(points_full.shape[0])
        T_pcd = self.dataset.get_pose(index)

        pcd = pcd_full
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,0] > self.lower_bound)[0])
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,0] < self.upper_bound)[0])
        pcd = pcd.voxel_down_sample(voxel_size=0.25)
        points = np.asarray(pcd.points)
        num_points = points.shape[0]

        cam_name = "cam2"
        point_to_label_reprojections = []

        start_ind = max(index-5, 0)
        for points_index in range(start_ind, start_ind+10):
            masks = self.dataset.get_image_instances(cam_name, points_index)
            label = self.__masks_to_image(masks)

            T_world2lidar = np.linalg.inv(self.dataset.get_pose(points_index))
            T_lidar2cam, K = self.dataset.get_calibration_matrices(cam_name)
            T_world2cam = T_lidar2cam @ T_world2lidar
            
            point_to_label_reprojections.append(reproject_points_to_label(np.array(pcd.points), T_pcd, label, T_world2cam, K, hidden_point_removal=True))


        association_matrix = merge_associations(point_to_label_reprojections, len(pcd.points))

        proximity_threshold = 2 # meters that points can be apart froim each other and still be considered neighbors
        alpha = 6.0 # weight of the spatial proximity term
        beta = 1.0 # weight of the feature similarity term

        dist_matrix = cdist(points, points)

        mask = np.where(dist_matrix <= proximity_threshold, 1, 0)

        feature_diff = self.__get_feature_diff(dist_matrix, association_matrix, num_points, proximity_threshold, beta)

        A = np.exp(-alpha * (mask * dist_matrix)) * feature_diff

        sc = SpectralClustering(n_clusters=self.n_clusters, affinity="precomputed", n_init=100, assign_labels="discretize")
        labels = sc.fit_predict(A)  

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        i=0
        for point in np.asarray(pcd_full.points):
            if point[0] > self.lower_bound and point[0] < self.upper_bound:
                [_, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
                labels_full[i] = labels[idx[0]] + 1
            else:
                labels_full[i] = 0
            i+=1

        return labels_full