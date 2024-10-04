import numpy as np
import copy
import open3d as o3d
import zlib
import torch
import torch.nn.functional as F
import random 

file_folder = 'data/'

folder = '/media/cedric/Datasets2/semantic_kitti/sequences/05/'
fn = '002664'

tarl_file =  '/media/cedric/Datasets2/semantic_kitti/tarl_features/05/' + fn + '.bin'
points_file = folder + 'velodyne/' +   fn + '.bin'
label_file = folder + 'labels/' +   fn + '.label'

POINT_MODE = False ###if set to False : average the features over some cluster (in this example we take the clusters from the kitti, labels but could be done with clustering as well)

def generate_random_colors(N):
    colors = []
    for _ in range(N):
        colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    return colors


def color_points_by_labels(points, labels,unique_labels=None,shift=False):
    random_colors = generate_random_colors(200)
    pcd = o3d.geometry.PointCloud()
    if shift : 
        points[:,0] += 150
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = []
    if unique_labels is None:
        unique_labels = list(np.unique(labels)) 
    for i in range(labels.shape[0]):
        colors.append(random_colors[unique_labels.index(int(labels[i])) + 1])

    pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors) / 255)

    return pcd


def vis_pcd_similarity(points, feats,shift=False,indices=None):
    pcd = o3d.geometry.PointCloud()

    feats = (feats - feats.min()) / (feats.max() - feats.min()) * feats
    Q1 = np.percentile(feats, 1)
    Q3 = np.percentile(feats, 99)
    
    IQR = Q3 - Q1

    # Perform Robust Scaling
    robust_scaled_data = (feats - Q1) / IQR * 1.7
    feats = robust_scaled_data*255

    colors = [ [0, int(label), int(label)] for label in feats]
    if indices is not None :
        idcs = indices.reshape(-1).tolist()
        for idx in idcs : 
            colors[idx] = [0, 255, 255]

    colors = np.asarray(colors) / 255.
    colors = colors[:, ::-1]

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if shift:
        shift_size = (pcd.get_max_bound()[1] - pcd.get_min_bound()[1]) * 1.5
        points[:, 1] = points[:, 1] + shift_size
        pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def load_data():
        ##load tarl features 
        with open(tarl_file, 'rb') as f_in:
                tarl_dim = 96
                compressed_data = f_in.read()
                decompressed_data = zlib.decompress(compressed_data)
                loaded_array = np.frombuffer(decompressed_data, dtype=np.float32)
                features = loaded_array.reshape(-1,tarl_dim)

        ##load kitti point cloud file 
        points_set = np.fromfile(points_file, dtype=np.float32)
        points_set = points_set.reshape((-1, 4))
        points = points_set[:,:3]

        labels_orig = np.fromfile(label_file, dtype=np.uint32)
        labels_orig = labels_orig.reshape((-1))
        labels = labels_orig & 0xFFFF
        instance_labels = labels_orig & 0xFFFF0000
        unique_instances = list(np.unique(instance_labels)) 

        pcd_labeled = color_points_by_labels(points,instance_labels)
        return points, features, instance_labels, pcd_labeled


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least one point [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def demo_manual_registration(points, features,instance_labels,pcd):
    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(pcd)
    mean_features = np.zeros_like(features)
    unique_instances = list(np.unique(instance_labels))
    if POINT_MODE == False :
                ##assing mean features for each instance
                for random_id in unique_instances:
                        (cluster_idcs,) = np.where(instance_labels == random_id)
                        cluster_feature_mean = features[cluster_idcs].mean(0)
                        mean_features[cluster_idcs] = cluster_feature_mean
                        
    for point_id in picked_id_source:
        if POINT_MODE : 
                point_feature = features[point_id]
                coords_feature = points[point_id]
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=20)
                sphere.translate([coords_feature[0],coords_feature[1],coords_feature[2]])
                sphere.paint_uniform_color([1, 0, 0])

                cos_sim = F.cosine_similarity(torch.from_numpy(point_feature), torch.from_numpy(features), dim=1).unsqueeze(1).detach() 

                pcd_vis = vis_pcd_similarity(points,cos_sim.numpy())
                o3d.visualization.draw_geometries([pcd_vis,sphere])
        else :  
                point_feature_mean = mean_features[point_id]
                (cluster_idcs,) = np.where(instance_labels == instance_labels[point_id])
                cos_sim = F.cosine_similarity(torch.from_numpy(point_feature_mean), torch.from_numpy(mean_features), dim=1).unsqueeze(1).detach() 
                pcd_vis = vis_pcd_similarity(points,cos_sim.numpy(),indices=cluster_idcs)
                o3d.visualization.draw_geometries([pcd_vis])
                o3d.io.write_point_cloud('/home/cedric/Lidar_Segmentation_Clustering/outputs/similarity_' + fn + '.pcd' ,pcd_vis)

if __name__ == "__main__":
    points, features, labels, pcd = load_data()
    demo_manual_registration(points,features,labels,pcd)
