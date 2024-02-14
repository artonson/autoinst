import os 
import numpy as np
import open3d as o3d
import copy

data = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
    252: [245, 150, 100],
    256: [255, 0, 0],
    253: [200, 40, 255],
    254: [30, 30, 255],
    255: [90, 30, 150],
    257: [250, 80, 100],
    258: [180, 30, 80],
    259: [255, 0, 0]
}

print(data)



def color_pcd_by_labels(pcd, labels):

    pcd_colored = copy.deepcopy(pcd)
    pcd_colors = np.zeros(np.asarray(pcd.points).shape)
    unique_labels = list(np.unique(labels))

    #for i in range(len(pcd_colored.points)):
    for i in unique_labels:
        idcs = np.where(labels == i)
        idcs = idcs[0]
        pcd_colors[idcs] = np.array(data[i])

    pcd_colored.colors = o3d.utility.Vector3dVector(pcd_colors/255)
    return pcd_colored

#base_fn = '/home/cedric/unsup_3d_instances/point-to-pixel-mapping/pcd_preprocessed/output_chunks/'
base_fn = '/home/cedric/unsup_3d_instances/point-to-pixel-mapping/pcd_preprocessed/out_maps/5/'
#cur = 'tarl_spatial_data/'
fn = os.listdir(base_fn)
fn = [f for f in fn if f.endswith('pcd')]
fn = sorted(fn)
#pcd = o3d.io.read_point_cloud(fp)

for f in fn :
    fp = base_fn +  f 
    '''
    p_label = fp.split('.')[0] + '.npz' 
    with np.load(p_label) as f :
        labels = f['labels']
    '''
    pcd = o3d.io.read_point_cloud(fp)
    #pcd = color_pcd_by_labels(pcd,labels)

    o3d.visualization.draw_geometries([pcd])

