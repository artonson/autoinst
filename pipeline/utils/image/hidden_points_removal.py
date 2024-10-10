# Different functions that accept point clouds as numpy arrays and return mask of visible points
import numpy as np
from utils.point_cloud.point_cloud_utils import get_pcd
from scipy.spatial import cKDTree

def hidden_point_removal_o3d(points_np: np.array, camera: list, radius_factor=100):
    """
    Removes points that are not visible from the camera
    Args:
        points_np:      3D points in camera coordinate [npoints, 3]
        camera:         Camera position [3]
        radius_factor:  Factor to determine the radius of the sphere
    Returns:
        pt_map:         Mask of visible points
    """
    assert(len(camera) == 3)

    points_o3d = get_pcd(points_np)

    diameter = np.linalg.norm(np.asarray(points_o3d.get_max_bound()) - 
                              np.asarray(points_o3d.get_min_bound()))
    radius = diameter * radius_factor
    _, pt_map = points_o3d.hidden_point_removal(camera, radius)
    
    return pt_map

def hidden_point_removal_biasutti(points: np.array, n_neighbours=64, n_jobs=4, vis_type='mean'):
        """
        Removes points that are not visible from the camera
        Args:
            points:        3D points in camera coordinate [npoints, 3]
            n_neighbours:  Number of neighbours to consider for cKDTree
            n_jobs:        Number of jobs to run in parallel
            vis_type:      Type of visibility score
        Returns:
            pt_map:    Mask of visible points
        """

        # take UV coordinates of the pixel in the image space
        uv, z = points[:, :2], points[:, 2]

        # index for fast neighbour queries
        kdt = cKDTree(uv)
        _, indexes = kdt.query(
            uv,
            k=n_neighbours + 1,
            workers=n_jobs)

        # take groups of points around each query point
        z_nbr = z[indexes[:, 1:]]

        # min/max depth of points in each group
        z_min = np.min(z_nbr, axis=1)
        z_max = np.max(z_nbr, axis=1)

        # depth of the query point
        z_point = z[indexes[:, 0]]

        # compute continuous visibility scores;
        # score -> 0 (very large d_p) means background/invisible,
        # score -> 1 (small d_p) means foreground/visible
        visible_score = np.exp(-(z_point - z_min) ** 2 / (z_max - z_min) ** 2)

        if vis_type == 'mean':
            thr = np.mean(visible_score)
        else:
            assert isinstance(vis_type, float)
            thr = vis_type

        pt_map = (visible_score > thr).astype(bool)

        return pt_map