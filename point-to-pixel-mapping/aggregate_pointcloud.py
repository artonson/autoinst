import open3d as o3d
import numpy as np
from open3d.pipelines import registration
from point_cloud_utils import get_pcd, get_subpcd, get_statistical_inlier_indices
import sys
lib_path = '/Users/laurenzheidrich/Documents/Studium/Hiwi_TUM.nosync/Programming/voxel_clustering_dependencies/build/patchworkpp'
sys.path.insert(0,lib_path)
import pypatchworkpp

def aggregate_pointcloud(dataset, ind_start, ind_end, icp=False, icp_threshold=0.9, ground_segmentation=None):
    '''
    Args:
        dataset:    dataset object
        ind_start:  start index of point clouds to aggregate
        ind_end:    end index of point clouds to aggregate
    Returns:    
        map_pcd:    aggregated point cloud
        poses:      list of poses of point clouds
    '''
    poses = []
    first_pose = dataset.get_pose(ind_start)


    if ground_segmentation is None:

        map_pcd = o3d.geometry.PointCloud()

        for i in range(ind_start, ind_end):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dataset[i].point_cloud[:, :3])
            pose = dataset.get_pose(i)
            poses.append(pose)

            transform = np.linalg.inv(first_pose) @ pose

            if icp and i != ind_start:

                map_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=30))
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=30))

                reg_p2l = registration.registration_icp(pcd, map_pcd, icp_threshold, transform, registration.TransformationEstimationPointToPlane(), registration.ICPConvergenceCriteria(max_iteration=1000))

                transform = reg_p2l.transformation

            map_pcd += pcd.transform(transform)

        if icp:
            map_without_normals = o3d.geometry.PointCloud()
            map_without_normals.points = map_pcd.points
            return map_without_normals, poses

        return map_pcd, poses

    else:

        map_pcd_ground = o3d.geometry.PointCloud()
        map_pcd_nonground = o3d.geometry.PointCloud()

        if ground_segmentation == 'patchwork':
            params = pypatchworkpp.Parameters()
            params.verbose = False
            PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

        for i in range(ind_start, ind_end):

            pcd = get_pcd(dataset[i].point_cloud)
            pose = dataset.get_pose(i)
            poses.append(pose)
            transform = np.linalg.inv(first_pose) @ pose
            
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

            pcd_ground = get_subpcd(pcd, ground_idcs) 
            pcd_nonground = get_subpcd(pcd, nonground_idcs)

            if icp and i != ind_start:
                merge = map_pcd_ground + map_pcd_nonground
                merge.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=30))
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=30))

                reg_p2l = registration.registration_icp(pcd, merge, icp_threshold, transform, registration.TransformationEstimationPointToPlane(), registration.ICPConvergenceCriteria(max_iteration=1000))
                transform = reg_p2l.transformation

            map_pcd_ground += pcd_ground.transform(transform)
            map_pcd_nonground += pcd_nonground.transform(transform)

        return map_pcd_ground, map_pcd_nonground, poses