### Inspired / Copied from https://github.com/darylclimb/cvml_project/blob/master/projections/lidar_camera_projection/utils.py ###

import open3d as o3d
import numpy as np


def get_pcd(points: np.array):
    """
    Convert numpy array to open3d point cloud
    Args:
        points: 3D points in camera coordinate [npoints, 3]
    Returns:
        pcd: open3d point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd