import open3d as o3d
import numpy as np

def aggregate_pointcloud(dataset, ind_start, ind_end):
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
        map_pcd += pcd.transform(transform)

    return map_pcd, poses