import open3d as o3d
import numpy as np
from open3d.pipelines import registration
from point_cloud_utils import get_pcd, get_subpcd, get_statistical_inlier_indices
import sys
#lib_path = '/Users/cedric/Lidar_Segmentation_Clustering/voxel_clustering_dependencies/build/patchworkpp/'
lib_path = '/home/cedric/unsup_3d_instances/pipeline/segmentation/utils/voxel_clustering_dependencies/build/patchworkpp/'
sys.path.insert(0,lib_path)
import pypatchworkpp


def aggregate_pointcloud(dataset, ind_start, ind_end, icp=False, icp_threshold=0.9, ground_segmentation=None):
    '''
    Returns aggregated point cloud from dataset in world coordinate system
    Args:
        dataset:                dataset object
        ind_start:              start index of point clouds to aggregate
        ind_end:                end index of point clouds to aggregate
        icp:                    whether to use icp for registration
        icp_threshold:          threshold for icp
        ground_segmentation:    None, 'patchwork' or 'open3d'
    Returns:    
        map_pcd_ground:         Merged point cloud of ground points
        map_pcd_nonground:      Merged point cloud of non-ground points
        poses:                  List of poses of point clouds
        world_pose:             Pose of world coordinate system, i.e. of aggregated point cloud

    '''
    poses = []
    world_pose = np.eye(4)
    
    #pantopic_labels = []
    panoptic_labels_nonground = []
    panoptic_labels_ground = []
    seg_labels_nonground = []
    seg_labels_ground = []
    instance_labels_nonground = []
    instance_labels_ground = []

    if ground_segmentation is None:

        map_pcd = o3d.geometry.PointCloud()

        for i in range(ind_start, ind_end):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dataset[i].point_cloud[:, :3])
            
            pose = dataset.get_pose(i)
            poses.append(pose)

            transform = pose

            if icp and i != ind_start:

                map_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=200))
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=200))

                reg_p2l = registration.registration_icp(pcd, map_pcd, icp_threshold, transform, registration.TransformationEstimationPointToPlane(), registration.ICPConvergenceCriteria(max_iteration=1000))

                transform = reg_p2l.transformation

            map_pcd += pcd.transform(transform)

        map_pcd.normals = o3d.utility.Vector3dVector([])

        return map_pcd, poses

    else:

        map_pcd_ground = o3d.geometry.PointCloud()
        map_pcd_nonground = o3d.geometry.PointCloud()

        if ground_segmentation == 'patchwork':
            params = pypatchworkpp.Parameters()
            params.verbose = False
            PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

        for i in range(ind_start, ind_end):
            
            panoptic_labels =  dataset[i].panoptic_labels
            semantic_labels =  dataset[i].semantic_labels
            instance_labels =  dataset[i].instance_labels
            

            
            pcd = get_pcd(dataset[i].point_cloud)
            pose = dataset.get_pose(i)
            poses.append(pose)
            transform = pose
            
            if ground_segmentation == 'patchwork':
                intensity = dataset[i].intensity
                PatchworkPLUSPLUS.estimateGround(np.hstack((np.asarray(pcd.points),intensity.reshape(-1,1))))
                ground_idcs = PatchworkPLUSPLUS.get_ground_idcs()
                nonground_idcs = PatchworkPLUSPLUS.get_nonground_idcs()
                

            elif ground_segmentation == 'open3d':
                _, ground_idcs = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=2000)
                nonground_idcs = np.setdiff1d(np.arange(0, np.asarray(pcd.points).shape[0]), ground_idcs)
            else:
                raise ValueError('ground_segmentation must be either "None", "patchwork" or "open3d"')

            panoptic_labels_nonground.append(panoptic_labels[nonground_idcs])
            panoptic_labels_ground.append(panoptic_labels[ground_idcs])
            
            seg_labels_nonground.append(semantic_labels[nonground_idcs])
            seg_labels_ground.append(semantic_labels[ground_idcs])
            
            instance_labels_ground.append(instance_labels[ground_idcs])
            instance_labels_nonground.append(instance_labels[nonground_idcs])
            
            pcd_ground = get_subpcd(pcd, ground_idcs) 
            pcd_nonground = get_subpcd(pcd, nonground_idcs)

            if icp:
                if i != ind_start:
                    merge = map_pcd_ground + map_pcd_nonground
                    merge = merge.voxel_down_sample(voxel_size=0.3)
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=200))

                    reg_p2l = registration.registration_icp(pcd, merge, icp_threshold, transform, registration.TransformationEstimationPointToPlane(), registration.ICPConvergenceCriteria(max_iteration=1000))
                    transform = reg_p2l.transformation

                pcd_ground.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=200))
                pcd_nonground.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=200))

            map_pcd_ground += pcd_ground.transform(transform)
            map_pcd_nonground += pcd_nonground.transform(transform)

        map_pcd_ground.normals = o3d.utility.Vector3dVector([])
        map_pcd_nonground.normals = o3d.utility.Vector3dVector([])

        labels = {'seg_ground':seg_labels_ground,'seg_nonground':seg_labels_nonground,
                  'instance_ground':instance_labels_ground,'instance_nonground':instance_labels_nonground,
                  'panoptic_ground':panoptic_labels_ground,'panoptic_nonground':panoptic_labels_nonground}
        
        return map_pcd_ground, map_pcd_nonground, poses, world_pose, labels