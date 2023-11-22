import numpy as np
import open3d as o3d

def hidden_points_removal(path):
    pcd = o3d.io.read_point_cloud(str(path))
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    )

    # Define parameters used for hidden_point_removal
    camera = [0, 0, 0]
    radius = diameter * 100

    # Get all points that are visible from given view point
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    pcd = pcd.select_by_index(pt_map)

    return pcd, pt_map