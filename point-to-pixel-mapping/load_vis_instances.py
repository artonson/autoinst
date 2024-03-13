from visualization_utils import *
import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("pcd_preprocessed/merge_part_kitti_instance_fixed7.pcd")
# pcd = o3d.io.read_point_cloud("pcd_preprocessed/spatial7.pcd")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
unique_colors, label_vec = np.unique(colors, axis=0, return_inverse=True)

# o3d.visualization.draw_geometries([pcd])
labels = np.unique(label_vec)
points_of_instances = []
for label in labels:
    if label != 0:
        idcs = np.where(label_vec == label)
        idcs = idcs[0]
        instance_points = points[idcs]
        points_of_instances.append(instance_points)
        cur_pcd = o3d.geometry.PointCloud()
        cur_pcd.points = o3d.utility.Vector3dVector(instance_points)
        o3d.visualization.draw_geometries([cur_pcd])
