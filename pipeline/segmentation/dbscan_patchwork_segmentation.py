import numpy as np
import open3d as o3d

from sklearn.cluster import DBSCAN

from segmentation.abstract_segmentation import AbstractSegmentation
import os,sys

lib_path = os.path.expanduser('~') + '/unsup_3d_instances/pipeline/segmentation/voxel_clustering_dependencies/build/'  
sys.path.insert(0, lib_path+ "patchworkpp")
import pypatchworkpp

class DbscanPatchworkSegmentation(AbstractSegmentation):
    def __init__(self, dataset):
        self.dataset = dataset
        params = pypatchworkpp.Parameters()
        params.verbose = False

        self.PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    def segment_instances(self, index):
        capr = None
        hash_table = None
        cluster_indices = None
        cluster_id = None
        points = self.dataset.get_point_cloud(index,intensity=True)  #patchwork ++ requires points 
        #points = np.fromfile('segmentation/test.bin',dtype=np.float32).reshape(-1,4)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        
        # Estimate Ground
        self.PatchworkPLUSPLUS.estimateGround(points)

        # Get Ground and Nonground
        ground      = self.PatchworkPLUSPLUS.getGround()
        nonground   = self.PatchworkPLUSPLUS.getNonground()
        time_taken  = self.PatchworkPLUSPLUS.getTimeTaken()
        ground_idcs = self.PatchworkPLUSPLUS.get_ground_idcs()
        nonground_idcs = self.PatchworkPLUSPLUS.get_nonground_idcs()

        # Get centers and normals for patches
        centers     = self.PatchworkPLUSPLUS.getCenters()
        normals     = self.PatchworkPLUSPLUS.getNormals()

        labels = np.zeros(points.shape[0])
        labels[ground_idcs] = 1 # road

        not_road_points = np.delete(points, ground_idcs, axis=0)
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