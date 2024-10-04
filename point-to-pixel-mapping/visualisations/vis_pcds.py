import open3d as o3d 
import numpy as np 
import os 
import random 

folder_base = 'pcd_preprocessed/'

name_refined = folder_base + 'merge_refined6.pcd'
name_base = folder_base + 'sam_only66.pcd'

pcd_refined = o3d.io.read_point_cloud(name_refined)
pcd_base = o3d.io.read_point_cloud(name_base)


#o3d.visualization.draw_geometries([pcd_base,pcd_refined.translate([0,50,0])])

def generate_random_colors(N):
    colors = []
    for _ in range(N):
        colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    return colors

cols = generate_random_colors(900)

def demo_manual_registration(pcd):
    # pick points from two point clouds and builds correspondences
    vis_start(pcd)

def vis_start(pcd):
    #print("")
    #print(
    #    "1) Please pick at least one point [shift + left click]"
    #)
    #print("   Press [shift + right click] to undo point picking")
    #print("2) After picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()


if __name__ == '__main__': 
    demo_manual_registration(pcd_base)
    


