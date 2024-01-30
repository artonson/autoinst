import hdbscan
import numpy as np
import open3d as o3d

import sys 

custom_directory = '/home/cedric/3DUIS/utils/voxel_clustering_dependencies/build/patchworkpp/'
sys.path.insert(0, custom_directory)

import pypatchworkpp

params = pypatchworkpp.Parameters()
params.verbose = False

PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)
params = [2,0.4,1.5]


def clusters_hdbscan(points_set):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=20, min_samples=None
                            )

    clusterer.fit(points_set)

    labels = clusterer.labels_.copy()

    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return labels

def clusters_from_pcd(pcd):
    # clusterize pcd points
    labels = np.array(pcd.cluster_dbscan(eps=0.25, min_points=10))
    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return labels

def clusterize_pcd(points, scan_path):
    scan_info = scan_path.split('/')
    scan_file = scan_info[-1].split('.')[0]
    seq_num = scan_info[-3]
    
    # Estimate Ground
    PatchworkPLUSPLUS.estimateGround(points)

    # Get Ground and Nonground
    ground      = PatchworkPLUSPLUS.getGround()
    nonground   = PatchworkPLUSPLUS.getNonground()
    time_taken  = PatchworkPLUSPLUS.getTimeTaken()
    
    ground_idcs = PatchworkPLUSPLUS.get_ground_idcs()
    nonground_idcs = PatchworkPLUSPLUS.get_nonground_idcs()

    # Get centers and normals for patches
    centers     = PatchworkPLUSPLUS.getCenters()
    normals     = PatchworkPLUSPLUS.getNormals()
    
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # instead of ransac use patchwork
    #ground_file = f'./Datasets/SemanticKITTI/assets/patchwork/{seq_num}/{scan_file}.label'
    #ground_labels = np.fromfile(ground_file, dtype=np.uint32)
    #ground_labels.reshape((-1))
    
    
    
    #inliers = list(np.where(ground_labels == 9)[0])
    inliers = ground_idcs

    pcd_ = pcd.select_by_index(inliers, invert=True)
    labels_ = np.expand_dims(clusters_hdbscan(np.asarray(pcd_.points)), axis=-1)

    labels = np.ones((points.shape[0], 1)) * -1
    mask = np.ones(labels.shape[0], dtype=bool)
    mask[inliers] = False

    labels[mask] = labels_
    
    
    #o3d.visualization.draw_geometries([pcd_])

    return np.concatenate((points, labels), axis=-1)

