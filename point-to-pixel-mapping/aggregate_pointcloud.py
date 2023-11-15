import open3d as o3d
import numpy as np
from open3d.pipelines import registration

def aggregate_pointcloud(dataset, ind_start, ind_end, icp=False, icp_threshold=0.9):
    '''
    Args:
        dataset:    dataset object
        ind_start:  start index of point clouds to aggregate
        ind_end:    end index of point clouds to aggregate
    Returns:    
        map_pcd:    aggregated point cloud
        poses:      list of poses of point clouds
    '''

    map_pcd = o3d.geometry.PointCloud()
    poses = []
    first_pose = dataset.get_pose(ind_start)

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