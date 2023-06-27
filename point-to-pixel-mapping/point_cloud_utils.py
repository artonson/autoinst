# Transformation on point clouds
import numpy as np


def project_pcd(points, projection_matrix):
    """
    Apply the perspective projection
    Args:
        points:              3D points in coordinate system 1 [npoints, 3]
        projection_matrix:   Projection matrix from coordinate system 1 to 2 [3, 4]
    Returns:
        points:              3D points in coordinate system 2 [npoints, 3]
    """
    num_pts = points.shape[0]

    # Change to homogenous coordinate
    points = np.vstack((points.transpose(), np.ones((1, num_pts))))
    points = projection_matrix @ points
    points[:2, :] /= points[2, :]

    return points.transpose()


def filter_points_fov(points_camframe, points_velo, img_width, img_height):
    """
    Filter points to be within the image FOV
    Args:
        points_camframe:    3D points in camera coordinate [3, npoints]
        points_velo:        3D points in velo coordinate [3, npoints]
        img_width:          Image width
        img_height:         Image height
    Returns:
        points_camframe_fov:    3D points in camera coordinate within image FOV [3, npoints]
    """

    # Filter lidar points to be within image FOV
    inds = np.where((points_camframe[0, :] < img_width) & (points_camframe[0, :] >= 0) &
                    (points_camframe[1, :] < img_height) & (points_camframe[1, :] >= 0) &
                    (points_velo[0,:] > 0)
                    )[0]

    # Filter out pixels points
    points_camframe_fov = points_camframe[:, inds]
    return points_camframe_fov, inds

def filter_points_from_dict(points, filter_dict):
    """
    Filter points based on a dict
    Args:
        points:      3D points in camera coordinate [npoints, 3]
        filter_dict: dict that maps point indices to pixel coordinates
    Returns:
        points:    3D points in camera coordinate within image FOV [npoints, 3]
    """

    inds = np.array(list(filter_dict.keys()))
    return points[inds]