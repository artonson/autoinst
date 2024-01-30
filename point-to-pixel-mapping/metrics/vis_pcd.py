import os 
import numpy as np
import open3d as o3d

fn = '/home/cedric/Downloads/non_ground7.pcd'

pcd = o3d.io.read_point_cloud(fn)
print(pcd)
o3d.visualization.draw_geometries([pcd])


