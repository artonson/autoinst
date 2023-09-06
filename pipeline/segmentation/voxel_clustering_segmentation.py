import numpy as np
import open3d as o3d


from abstract_segmentation import AbstractSegmentation
import sys 
import os

lib_path = os.path.expanduser('~') + '/Thesis-Code/numpy_lidar_segementation_wrapper/build/'  ##need to add the build dependencies in a seperate folder 
sys.path.insert(0, lib_path)
import pycluster
import pypatchworkpp

class VoxelClusterSegmentation(AbstractSegmentation):
    def __init__(self):
        #self.dataset = dataset
        params = pypatchworkpp.Parameters()
        params.verbose = True

        self.PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params) #ground segmentation class
        params = [2,0.4,1.5]
        self.cvc = pycluster.CVC_cluster(params) #clustering class 

    def segment_instances(self, index):
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


        ###clustering 
        capr = self.cvc.calculateAPR(nonground)
        hash_table = self.cvc.build_hash_table(capr)
        cluster_indices = self.cvc.cluster(hash_table,capr)
        cluster_id = self.cvc.most_frequent_value(cluster_indices)

        labels = np.zeros(points.shape[0])
        labels[ground_idcs] = 1 
        
        #colors = [[0,0,0] for i in range(points.shape[0])]
        
        #for idx in nonground_idcs : 
        #    colors[idx]  = [1,0,0]
            
        for i in range(len(cluster_id)): 
                for j in range(len(cluster_indices)):
                        if cluster_indices[j] == cluster_id[i]:
                                ##append point to cloud with certain colour 
                                labels[nonground_idcs[j]] = cluster_id[i] 
                                
        
        #pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors)) 
        #o3d.visualization.draw_geometries([pcd])

        return labels
    

if __name__ == "__main__":
    cluster = VoxelClusterSegmentation()
    cluster.segment_instances(0)
    
    