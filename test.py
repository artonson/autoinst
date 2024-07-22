import open3d as o3d 

pcd = o3d.io.read_point_cloud('ncuts_instances_tarl1.pcd')
o3d.visualization.draw_geometries([pcd])
