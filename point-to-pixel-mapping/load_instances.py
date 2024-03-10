import os 
import open3d as o3d 
from visualization_utils import * 


pcd_merge = o3d.io.read_point_cloud('/home/cedric/unsup_3d_instances/point-to-pixel-mapping/pcd_preprocessed/semantics/7/ncuts_tarl_dino_spatial7_0.pcd')

with np.load('/home/cedric/unsup_3d_instances/point-to-pixel-mapping/pcd_preprocessed/semantics/7/kitti_semantic7_0.npz') as data : 
        labels = data['labels_kitti']


pcd_re = color_pcd_by_labels(pcd_merge,labels)
o3d.visualization.draw_geometries([pcd_re])
