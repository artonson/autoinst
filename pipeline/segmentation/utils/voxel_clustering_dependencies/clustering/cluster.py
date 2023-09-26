import sys 
import os 
import numpy as np 
import open3d as o3d
import random
import seaborn as sns

random.seed(30)

import random
import matplotlib.pyplot as plt

# Function to generate a random RGB color
def random_color():
    return (random.random(), random.random(), random.random())



# Number of random colors to generate
num_colors = 200

# Generate the random colors
random_colors = [random_color() for _ in range(num_colors)]


pts_list = []
custom_directory = 'build/'
sys.path.insert(0, custom_directory)

import pycluster
import pypatchworkpp
print('imported pycluster')



pcd_file_path = 'test.bin'
#cloud = pcl.load(pcd_file_path)

pts_array = np.fromfile(pcd_file_path, dtype=np.float32)
pointcloud = pts_array.reshape(-1,4)
#pts_array = pts_array[:,0:3]

'''
threshold_x = 10 
threshold_x_neg = -5
threshold_y = 10
threshold_y_neg = -5

# Create a boolean mask for rows where the value at index 0 is above the threshold
maskx = pts_array[:, 0] <= threshold_x  # Assumes you want to compare the first column (index 0)
# Use the mask to filter out rows where the value at index 0 is above the threshold
pts_array = pts_array[maskx]


maskx = pts_array[:, 0] >= threshold_x_neg  # Assumes you want to compare the first column (index 0)
# Use the mask to filter out rows where the value at index 0 is above the threshold
pts_array = pts_array[maskx]


masky = pts_array[:, 1] <= threshold_y  # Assumes you want to compare the first column (index 0)
# Use the mask to filter out rows where the value at index 0 is above the threshold
pts_array = pts_array[masky]

masky = pts_array[:, 1] >= threshold_y_neg  # Assumes you want to compare the first column (index 0)
# Use the mask to filter out rows where the value at index 0 is above the threshold
pts_array = pts_array[masky]
'''

params = pypatchworkpp.Parameters()
params.verbose = True

PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)
params = [2,0.4,1.5]
cvc = pycluster.CVC_cluster(params)

# Estimate Ground
PatchworkPLUSPLUS.estimateGround(pointcloud)

# Get Ground and Nonground
ground      = PatchworkPLUSPLUS.getGround()
nonground   = PatchworkPLUSPLUS.getNonground()
time_taken  = PatchworkPLUSPLUS.getTimeTaken()

# Get centers and normals for patches
centers     = PatchworkPLUSPLUS.getCenters()
normals     = PatchworkPLUSPLUS.getNormals()


###clustering 
capr = cvc.calculateAPR(nonground)
hash_table = cvc.build_hash_table(capr)
cluster_indices = cvc.cluster(hash_table,capr)
cluster_id = cvc.most_frequent_value(cluster_indices)

pt_colors = []
pt_coords = []
sparse_res = 0.05

for i in range(len(cluster_id)): 
        color = random_colors[cluster_id.index(cluster_id[i])]
        for j in range(len(cluster_indices)):
                if cluster_indices[j] == cluster_id[i]:
                        ##append point to cloud with certain colour 
                        pt_colors.append(color)
                        pt_coords.append(sparse_res* nonground[j])
                        
print("num clusters", len(cluster_id))
        
pt_colors = np.array(pt_colors)
pt_colors = pt_colors[:, ::-1]
pt_coords = np.vstack(pt_coords)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pt_coords)
pcd.colors = o3d.utility.Vector3dVector(pt_colors)

pcd_ground = o3d.geometry.PointCloud()
pcd_ground.points = o3d.utility.Vector3dVector(ground * sparse_res)
pcd_ground.colors = o3d.utility.Vector3dVector(np.vstack([np.array([0,1,0]) for i in range(len(ground))]))

o3d.visualization.draw_geometries([pcd,pcd_ground])

'''
print("Origianl Points  #: ", pointcloud.shape[0])
print("Ground Points    #: ", ground.shape[0])
print("Nonground Points #: ", nonground.shape[0])
print("Time Taken : ", time_taken / 1000000, "(sec)")
print("Press ... \n")
print("\t H  : help")
print("\t N  : visualize the surface normals")
print("\tESC : close the Open3D window")

# Visualize
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width = 600, height = 400)

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

ground_o3d = o3d.geometry.PointCloud()
ground_o3d.points = o3d.utility.Vector3dVector(ground)
ground_o3d.colors = o3d.utility.Vector3dVector(
np.array([[0.0, 1.0, 0.0] for _ in range(ground.shape[0])], dtype=float) # RGB
)

nonground_o3d = o3d.geometry.PointCloud()
nonground_o3d.points = o3d.utility.Vector3dVector(nonground)
nonground_o3d.colors = o3d.utility.Vector3dVector(
np.array([[1.0, 0.0, 0.0] for _ in range(nonground.shape[0])], dtype=float) #RGB
)

centers_o3d = o3d.geometry.PointCloud()
centers_o3d.points = o3d.utility.Vector3dVector(centers)
centers_o3d.normals = o3d.utility.Vector3dVector(normals)
centers_o3d.colors = o3d.utility.Vector3dVector(
np.array([[1.0, 1.0, 0.0] for _ in range(centers.shape[0])], dtype=float) #RGB
)

vis.add_geometry(mesh)
vis.add_geometry(ground_o3d)
vis.add_geometry(nonground_o3d)
vis.add_geometry(centers_o3d)

vis.run()
vis.destroy_window()
'''

