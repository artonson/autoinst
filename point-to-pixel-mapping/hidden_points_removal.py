# Different functions that accept point clouds as numpy arrays and return mask of visible points
import numpy as np
from point_cloud_utils import get_pcd
from scipy.spatial import cKDTree

def hidden_point_removal_o3d(points_np: np.array, camera: list, radius_factor=100):
    """
    Removes points that are not visible from the camera
    Args:
        points_np: 3D points in camera coordinate [npoints, 3]
        camera:    Camera position [3]
        radius_factor:  Factor to determine the radius of the sphere
    Returns:
        pt_map:    Mask of visible points
    """
    assert(len(camera) == 3)

    points_o3d = get_pcd(points_np)

    diameter = np.linalg.norm(np.asarray(points_o3d.get_max_bound()) - 
                              np.asarray(points_o3d.get_min_bound()))
    radius = diameter * radius_factor
    _, pt_map = points_o3d.hidden_point_removal(camera, radius)
    
    return pt_map


class BiasuttiVisibility:
    # An implementation of
    # Biasutti, Pierre, et al. "Visibility estimation in point clouds with
    # variable density." International Conference on Computer Vision Theory and
    # Applications (VISAPP). 2019.

    #it assumes self.view.depth is a [u v z] array of 
    #point coordinates (obtained after performing perspective projection

    def __init__(self, depth, n_neighbours=64, n_jobs=4, vis_type='mean'):  
        self._n_neighbours = n_neighbours
        self._n_jobs = n_jobs
        self._type = vis_type
        self._depth = depth

    def at_image(self):
        depth = self._depth

        # take UV coordinates of the pixel in the image space
        uv, z = depth[:, :2], depth[:, 2]

        # index for fast neighbour queries
        kdt = cKDTree(uv)
        _, indexes = kdt.query(
            uv,
            k=self._n_neighbours + 1,
            workers=self._n_jobs)

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

        if 'mean' == self._type:
            thr = np.mean(visible_score)
        else:
            assert isinstance(self._type, float)
            thr = self._type

        visibility_mask = (visible_score > thr).astype(bool)

        return visibility_mask
