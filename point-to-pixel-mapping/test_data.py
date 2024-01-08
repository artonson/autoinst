import os 
import numpy as np 
import open3d as o3d 
import copy
import random 

def generate_random_colors(N, seed=0):
    colors = set()  # Use a set to store unique colors
    while len(colors) < N:  # Keep generating colors until we have N unique ones
        # Generate a random color and add it to the set
        colors.add((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    return list(colors)  # Convert the set to a list before returning


def color_pcd_by_labels(pcd, labels,colors=None):

    if colors == None :
        colors = generate_random_colors(2000)
    pcd_colored = copy.deepcopy(pcd)
    pcd_colors = np.zeros(np.asarray(pcd.points).shape)
    unique_labels = list(np.unique(labels))

    #background_color = np.array([0,0,0])


    #for i in range(len(pcd_colored.points)):
    for i in unique_labels:
        if i == -1 :
            continue
        idcs = np.where(labels == i)
        idcs = idcs[0]
        if i == 0 :
            pass
        else :
            pcd_colors[idcs] = np.array(colors[unique_labels.index(i)])

        #if labels[i] != (-1):
        #    pcd_colored.colors[i] = np.array(colors[labels[i]]) / 255
    pcd_colored.colors = o3d.utility.Vector3dVector(pcd_colors/ 255)
    return pcd_colored


pth = '/home/cedric/unsup_3d_instances/point-to-pixel-mapping/pcd_preprocessed/0/'
for f in os.listdir(pth):
    fn = pth + f
    with np.load(fn) as data:
                        xyz = data['pts']
                        ncut_labels = data['ncut_labels']
                        kitti_labels = data['kitti_labels']
    cur_pcd = o3d.geometry.PointCloud()
    cur_pcd.points = o3d.utility.Vector3dVector(xyz)
    cur_pcd = color_pcd_by_labels(cur_pcd,ncut_labels)
    o3d.visualization.draw_geometries([cur_pcd])

