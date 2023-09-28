import numpy as np
import open3d as o3d

from sklearn.cluster import DBSCAN
import gc

from segmentation.abstract_segmentation import AbstractSegmentation
from segmentation.utils.aggregate import AggregationClustering
import matplotlib.pyplot as plt
from memory_profiler import profile

import random

def generate_random_colors(N):
    colors = []
    for _ in range(N):
        colors.append([random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)])

    return colors

random_colors = generate_random_colors(2000)

class AggregatedSegmentation(AbstractSegmentation):
    def __init__(self, dataset,dataset_name='kitti',clusterer='dbscan',sequence='00',window=10):
        self.dataset = dataset
        self.window = window
        self.dataset_name = dataset_name
        
        if dataset_name == 'kitti':
            self.base_dir = self.dataset.dataset_path + 'sequences/' + self.dataset.sequence  + '/'
        
            self.last_idx = len(self.dataset.poses) - 1 
            
        elif dataset_name == 'nuScenes':
            self.last_idx = len(self.dataset.tokens['LIDAR_TOP']) - 1
                
        self.clustering = AggregationClustering(dataset_name=dataset_name,
                                        clusterer=clusterer,sequence=sequence,dataset=dataset)
        
        
    def segment_instances(self, index):
        #points = self.dataset.get_point_cloud(index)
        points_set, ground_label, parse_idx = None, None, None
        segments = None 
        
        start = index - self.window//2
        end = index + self.window//2 + 1
        if start < 0 :
            diff = abs(start) 
            end = end + diff
            start = 0 
            aggregated_file_nums = list(range(start,end))
        elif end > self.last_idx  :
            diff = end - self.last_idx
            start = start - diff
            end = self.last_idx + 1
            
            aggregated_file_nums = list(range(start,end))
        
        else : 
            aggregated_file_nums = list(range(start,end))
            
        if self.dataset_name == 'kitti':
            fns = [self.base_dir + 'velodyne/' + str(i).zfill(6) + '.bin' for i in aggregated_file_nums]
            points_set, ground_label, parse_idx = self.clustering.aggregate_pcds(fns,self.dataset.dataset_path)
        
        elif self.dataset_name == 'nuScenes':
            points_set, ground_label, parse_idx = self.clustering.aggregate_pcds_nuscenes(aggregated_file_nums)
        
        segments = self.clustering.clusterize_pcd(points_set, ground_label)
        segments[parse_idx] = -np.inf
        segments = segments.astype(np.float16)
        
        file_idcs = list(range(0,self.window)) 
        
        pcd_parse_idx = np.unique(np.argwhere(segments == -np.inf)[:,0])
        ##test extraction with one point cloud (the first one)
        test_idx = aggregated_file_nums.index(index)
        
        if self.dataset_name == 'kitti':
            pts = np.fromfile(fns[test_idx], dtype=np.float32) ##full points file 
            pts = pts.reshape((-1, 4))
        elif self.dataset_name == 'nuScenes':
            pts = self.dataset.get_point_cloud(index)
        
        seg = segments[pcd_parse_idx[test_idx]+1:pcd_parse_idx[test_idx+1]]  #non ground points 

        ps1 = np.concatenate((pts, seg), axis=-1)
        pts = ps1[:,:-1]  #remove the 5th dimension from point cloud tensor  
        s1 = ps1[:,-1][:,np.newaxis] ##labels 
        #import pdb; pdb.set_trace()
        gc.collect()
        return s1.reshape(-1)
        
        
    def visualize_pcd_clusters(self,points, labels):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        
        colors = np.zeros((len(labels), 4))
        flat_indices = np.unique(labels[:,-1])
        max_instance = len(flat_indices)
        colors_instance = plt.get_cmap("prism")(np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))

        for idx in range(len(flat_indices)):
            colors[labels[:,-1] == flat_indices[int(idx)]] = colors_instance[int(idx)]

        colors[labels[:,-1] == -1] = [0.,0.,0.,0.]

        pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])

        o3d.visualization.draw_geometries([pcd])

        