import numpy as np 
import open3d as o3d
from visualization_utils import * 
import os

pth = '/home/cedric/unsup_3d_instances/point-to-pixel-mapping/pcd_preprocessed/1/'

fs = os.listdir(pth)

for f in fs : 
	
	with np.load(pth + f) as data:
	            xyz = data['pts'].astype(np.float)
	            labels = data['ncut_labels'].astype(np.int32)  
	            kitti_labels = data['kitti_labels']
	            #intensity = data['intensities']
	
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(xyz)
	pcd = color_pcd_by_labels(pcd,labels)
	o3d.visualization.draw_geometries([pcd])