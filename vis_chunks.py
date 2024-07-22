import open3d as o3d 
import os 
direc = '/home/cedric/unsup_3d_instances/point-to-pixel-mapping/out_refined/'
fs = os.listdir(direc)
sorted_file_list = sorted(fs, key=lambda x: int(x.split('.')[0]))


for f in sorted_file_list : 
    if f.endswith('.pcd'): 
        pcd = o3d.io.read_point_cloud(direc + f)
        o3d.visualization.draw_geometries([pcd])
