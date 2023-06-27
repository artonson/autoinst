# Different functions that accept point clouds as numpy arrays and return mask of visible points
import numpy as np
import open3d as o3d

def hidden_point_removal_o3d(points_np: np.array, camera: list):
    """
    Removes points that are not visible from the camera
    Args:
        points_np: 3D points in camera coordinate [npoints, 3]
        camera:    Camera position [3]
    Returns:
        pt_map:    Mask of visible points
    """
    assert(len(camera)==3)

    # Convert to open3d point cloud
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(points_np)
    

    diameter = np.linalg.norm(np.asarray(points_o3d.get_max_bound()) - np.asarray(points_o3d.get_min_bound()))
    radius = diameter * 100
    _, pt_map = points_o3d.hidden_point_removal(camera, radius)
    
    return pt_map