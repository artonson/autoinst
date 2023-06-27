# Function that take point clouds and calibration dict as input and return a dict that maps point indices to pixel coordinates
import numpy as np
from point_cloud_utils import filter_points_fov

def point_to_pixel(points_camframe: np.array, points_velo: np.array, proj_velo2cam: np.array, img_height: int, img_width: int):    
    """
    Function that takes point clouds and calibration dict as input and return a dict that maps point indices to pixel coordinates
    Args:
        points_camframe:    3D points in camera coordinate [npoints, 3]
        points_velo:        3D points in velo coordinate [npoints, 3]
        proj_velo2cam:      Projection matrix from velo to cam [3, 4]
        img_width:          Image width
        img_height:         Image height
    Returns:
        point_to_pixel_dict: dict that maps point indices to pixel coordinates
    """

    filtered_points, inds = filter_points_fov(points_camframe.transpose(), points_velo.transpose(), img_width, img_height)

    point_to_pixel_dict = {}
    for i in range(filtered_points.shape[1]):
        point_to_pixel_dict[inds[i]] = [int(np.round(filtered_points[0,i])), int(np.round(filtered_points[1,i]))]

    return point_to_pixel_dict