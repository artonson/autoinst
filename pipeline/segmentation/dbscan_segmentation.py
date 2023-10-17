import numpy as np
import open3d as o3d

from sklearn.cluster import DBSCAN

from segmentation.abstract_segmentation import AbstractSegmentation

class DbscanSegmentation(AbstractSegmentation):
    def __init__(self, dataset):
        self.dataset = dataset

    def segment_instances(self, index):
        points = self.dataset.get_point_cloud(index)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        _, inliers = pcd.segment_plane(distance_threshold=0.2,
                                                ransac_n=3,
                                                num_iterations=2000)


        plane_cloud = pcd.select_by_index(inliers)
        plane_cloud.paint_uniform_color([1.0, 0, 0])

        noneplane_cloud = pcd.select_by_index(inliers, invert=True)
        noneplane_cloud.paint_uniform_color([0, 0, 1.0])

        labels = np.zeros(points.shape[0])
        labels[inliers] = 1 # road

        not_road_points = np.delete(points, inliers, axis=0)
        clustering = DBSCAN(eps=0.7, min_samples=20).fit(not_road_points)
        labels_not_road = clustering.labels_

        last_row_ind = 0
        for i, label in enumerate(labels_not_road):
            for j, row in enumerate(points[last_row_ind:]):
                if np.all(row == not_road_points[i]):
                    last_row_ind = last_row_ind + j
                    if label == -1: # not labelled
                        labels[last_row_ind] = 0
                    else: # labelled as not road
                        labels[last_row_ind] = label + 2
                    break

        return labels