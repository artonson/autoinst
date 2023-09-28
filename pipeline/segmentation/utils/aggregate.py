import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import sys 
from memory_profiler import profile
import gc 
import random
random.seed(40)


lib_path = os.path.expanduser('~') + '/unsup_3d_instances/pipeline/segmentation/utils/voxel_clustering_dependencies/build/'  
sys.path.insert(0, lib_path+ "patchworkpp")
sys.path.insert(0, lib_path+ "clustering")

import pypatchworkpp
import pycluster

from sklearn.cluster import DBSCAN


def generate_random_colors(N):
    colors = []
    for _ in range(N):
        colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    return colors

class AggregationClustering(): 
    def __init__(self,dataset_name='kitti',clusterer='dbscan',sequence='00',dataset=None): 
        params = pypatchworkpp.Parameters()
        params.verbose = False

        self.dataset = dataset
        self.PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)
        self.eps = 0.7 
        self.min_samples = 20 
        self.sequence = sequence
        
        if dataset_name == 'nuScenes':
            self.eps = 0.5
            self.min_samples = 30 
            
        self.clusterer = clusterer
        self.ground_idcs = []

    def overlap_clusters(self,cluster_i, cluster_j, min_cluster_point=10):
        # get unique labels from pcd_i and pcd_j from segments bigger than min_clsuter_point
        unique_i, count_i = np.unique(cluster_i, return_counts=True)
        unique_i = unique_i[count_i > min_cluster_point]

        unique_j, count_j = np.unique(cluster_j, return_counts=True)
        unique_j = unique_j[count_j > min_cluster_point]

        # get labels present on both pcd (intersection)
        unique_ij = np.intersect1d(unique_i, unique_j)[1:]
            
        # labels not intersecting both pcd are assigned as -1 (unlabeled)
        cluster_i[np.in1d(cluster_i, unique_ij, invert=True)] = -1
        cluster_j[np.in1d(cluster_j, unique_ij, invert=True)] = -1

        return cluster_i, cluster_j

    def parse_calibration(self,filename):
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def load_poses_kitti(self,calib_fname, poses_fname):
        calibration = self.parse_calibration(calib_fname)
        poses_file = open(poses_fname)

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        poses = []

        for line in poses_file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def apply_transform(self,points, pose):
        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        return np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)[:,:3]

    def undo_transform(self,points, pose):
        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        return np.sum(np.expand_dims(hpoints, 2) * np.linalg.inv(pose).T, axis=1)[:,:3]
    
    def aggregate_pcds(self,data_batch, data_dir, use_ground_label=True):
        # load empty pcd point cloud to aggregate
        
        points_set = np.empty((0,4))
        self.ground_idcs = []

        # define a delimiter to divide the aggregated pcd
        p_delimiter = np.asarray([[-np.inf, -np.inf, -np.inf, -np.inf]])
        ground_label = np.empty((0,1))
        g_delimiter = np.asarray([[-np.inf]])

        # define "namespace"
        seq_num = data_batch[0].split('/')[-3]
        fname = data_batch[0].split('/')[-1].split('.')[0]

        # load poses
        datapath = data_batch[0].split('velodyne')[0]
        poses = self.load_poses_kitti(os.path.join(datapath, 'calib.txt'), 
                        os.path.join(datapath.split('sequences')[0] + 'poses' ,self.sequence + '.txt'))
        
        for t in range(len(data_batch)):
            fname = data_batch[t].split('/')[-1].split('.')[0]
            # load the next t scan, apply pose and aggregate
            p_set_orig = np.fromfile(data_batch[t], dtype=np.float32)
            p_set_orig = p_set_orig.reshape((-1, 4))
            pose_idx = int(fname)
            p_set = p_set_orig.copy()
            p_set[:,:3] = self.apply_transform(p_set_orig[:,:3], poses[pose_idx])
            # aggregate a delimiter and the next scan
            points_set = np.vstack([points_set, p_delimiter, p_set])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(p_set[:,:3])
            labels = np.ones((p_set.shape[0],1))
            
            self.PatchworkPLUSPLUS.estimateGround(p_set_orig)
            ground      = self.PatchworkPLUSPLUS.getGround()
            nonground   = self.PatchworkPLUSPLUS.getNonground()
            time_taken  = self.PatchworkPLUSPLUS.getTimeTaken()
            ground_idcs = self.PatchworkPLUSPLUS.get_ground_idcs()
            self.ground_idcs.append(ground_idcs)
            nonground_idcs = self.PatchworkPLUSPLUS.get_nonground_idcs()

            # Get centers and normals for patches
            centers     = self.PatchworkPLUSPLUS.getCenters()
            normals     = self.PatchworkPLUSPLUS.getNormals()
            
            inliers = ground_idcs 
            #_, inliers = pcd.segment_plane(distance_threshold=0.25, ransac_n=3, num_iterations=200)
            g_set = np.ones((len(p_set),1)) * 251
            g_set[inliers] = 9
            
            labels[ground_idcs] = 0 

            
            #self.visualize_pcd_clusters(p_set[:,:3],labels)

            # aggregate a delimiter and the next scan
            ground_label = np.vstack([ground_label, g_delimiter, g_set])

        ground_label = np.vstack([ground_label, g_delimiter])
        points_set = np.vstack([points_set, p_delimiter])

        # get start position of each aggregated pcd
        pcd_parse_idx = np.unique(np.argwhere(ground_label == g_delimiter)[:,0])

        pose_idx = int(fname)
        points_set[:,:3] = self.undo_transform(points_set[:,:3], poses[pose_idx])
        points_set[pcd_parse_idx] = p_delimiter

        return points_set, ground_label, pcd_parse_idx

    
    def aggregate_pcds_nuscenes(self,data_batch):
        # load empty pcd point cloud to aggregate
        
        points_set = np.empty((0,4))
        self.ground_idcs = []

        # define a delimiter to divide the aggregated pcd
        p_delimiter = np.asarray([[-np.inf, -np.inf, -np.inf, -np.inf]])
        ground_label = np.empty((0,1))
        g_delimiter = np.asarray([[-np.inf]])



        for t in range(len(data_batch)):
            p_set = self.dataset.get_point_cloud(data_batch[t],pose_correction=True)
            # aggregate a delimiter and the next scan
            points_set = np.vstack([points_set, p_delimiter, p_set])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(p_set[:,:3])
            labels = np.ones((p_set.shape[0],1))
            
            self.PatchworkPLUSPLUS.estimateGround(p_set)
            ground      = self.PatchworkPLUSPLUS.getGround()
            nonground   = self.PatchworkPLUSPLUS.getNonground()
            time_taken  = self.PatchworkPLUSPLUS.getTimeTaken()
            ground_idcs = self.PatchworkPLUSPLUS.get_ground_idcs()
            self.ground_idcs.append(ground_idcs)
            nonground_idcs = self.PatchworkPLUSPLUS.get_nonground_idcs()

            # Get centers and normals for patches
            centers     = self.PatchworkPLUSPLUS.getCenters()
            normals     = self.PatchworkPLUSPLUS.getNormals()
            
            inliers = ground_idcs 
            #_, inliers = pcd.segment_plane(distance_threshold=0.25, ransac_n=3, num_iterations=200)
            g_set = np.ones((len(p_set),1)) * 251
            g_set[inliers] = 9
            
            labels[ground_idcs] = 0 

            
            #self.visualize_pcd_clusters(p_set[:,:3],labels)

            # aggregate a delimiter and the next scan
            ground_label = np.vstack([ground_label, g_delimiter, g_set])

        ground_label = np.vstack([ground_label, g_delimiter])
        points_set = np.vstack([points_set, p_delimiter])

        # get start position of each aggregated pcd
        pcd_parse_idx = np.unique(np.argwhere(ground_label == g_delimiter)[:,0])

        points_set[pcd_parse_idx] = p_delimiter
        
        

        return points_set, ground_label, pcd_parse_idx

    #@profile
    def clusters_hdbscan(self,points_set, n_clusters=50):
        
        capr = None
        hash_table = None
        cluster_indices = None
        cluster_id = None
                
        
        
        inf_rows = np.all(points_set == -np.inf, axis=1)
        non_inf_rows = ~np.all(points_set == -np.inf, axis=1)
        labels = np.ones((points_set.shape[0])) * -np.inf
        
        non_inf_points = points_set[~inf_rows] 
        
        if self.clusterer == 'dbscan': 
                clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
                clusterer.fit(non_inf_points[:,:3])
                labels_clustered = clusterer.labels_ 
                labels[non_inf_rows] = labels_clustered
        else : ##use voxel clustering
                params = [2,0.4,1.5]
                clusterer = pycluster.CVC_cluster(params) #clustering class
                capr = self.cvc.calculateAPR(non_inf_points)
                hash_table = self.cvc.build_hash_table(capr)
                cluster_indices = self.cvc.cluster(hash_table,capr)
                cluster_id = self.cvc.most_frequent_value(cluster_indices)
                labels = cluster_indices
		
        lbls, counts = np.unique(labels, return_counts=True)
        cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
        cluster_info = cluster_info[cluster_info[:,1].argsort()]

        clusters_labels = cluster_info[::-1][:n_clusters, 0]
        labels[np.in1d(labels, clusters_labels, invert=True)] = -1
        
        #self.visualize_pcd_clusters(non_inf_points[:,:3],labels)
        
        del clusterer
        del labels_clustered
        del lbls,counts, non_inf_points
        
        gc.collect()

        return labels

    def clusterize_pcd(self,points, ground):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # instead of ransac use patchwork
        inliers = list(np.where(ground == 9)[0])
        
        #print('inliers',len(inliers)) 
        pcd_ = pcd.select_by_index(inliers, invert=True)
        
        labels_ = np.expand_dims(self.clusters_hdbscan(np.asarray(pcd_.points)), axis=-1)

        labels = np.ones((points.shape[0], )) * -1
        mask = np.ones(labels.shape[0], dtype=bool)
        mask[inliers] = False

        labels[mask] = labels_.reshape(-1,)
        
        #pcd = o3d.geometry.PointCloud()
        
        #inf_rows = np.all(points == -np.inf, axis=1)
        #points = points[~inf_rows]
        #labels = labels[~inf_rows]
        
        #self.visualize_pcd_clusters(points,labels)
        
        
        return labels.reshape(-1,1)


    def visualize_pcd_clusters(self,points, labels):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        
        colors = []
        flat_indices = np.unique(labels)
        max_instance = len(flat_indices)
        colors_instance = plt.get_cmap("prism")(np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))
        unique_clusters = list(np.unique(labels))
        random_colors = generate_random_colors(len(unique_clusters))
        
        
        for i in range(labels.shape[0]):
            if labels[i] == -1:
                colors.append([0,0,0])
            else : 
                colors.append(random_colors[unique_clusters.index(labels[i])])

        
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors) /255)

        o3d.visualization.draw_geometries([pcd])
        
    