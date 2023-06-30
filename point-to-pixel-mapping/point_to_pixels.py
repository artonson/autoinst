# Function that take point clouds and calibration dict as input and
# return a dict that maps point indices to pixel coordinates

import numpy as np

def point_to_pixel(points_camframe: np.array, cam_intrinsics: np.array,
                   img_height: int, img_width: int):
    """
    Function that takes point clouds and calibration dict as input and return a 
    dict that maps point indices to pixel coordinates
    Args:
        points_camframe:    3D points in camera coordinate [npoints, 3]
        cam_intrinsics:     Camera intrinsics [3, 3]
        img_width:          Image width
        img_height:         Image height
    Returns:
        point_to_pixel_dict: dict that maps point indices to pixel coordinates
    """

    points_imgframe = cam_intrinsics @ points_camframe.transpose()
    points_imgframe[:2, :] /= points_imgframe[2, :]

    inds = np.where((points_imgframe[0, :] < img_width) & (points_imgframe[0, :] >= 0) &
                    (points_imgframe[1, :] < img_height) & (points_imgframe[1, :] >= 0) &
                    (points_imgframe[2, :] > 0)
                    )[0]

    point_ind_to_pixel_dict = {}
    for ind in inds:
        point_ind_to_pixel_dict[ind] = {}
        point_ind_to_pixel_dict[ind]['pixels'] = np.round(points_imgframe[:2, ind]).astype(int)
        
        distance = points_imgframe[:2, ind] - np.round(points_imgframe[:2, ind])
        distance_norm = np.linalg.norm(distance)
        point_ind_to_pixel_dict[ind]['distance'] = distance_norm

    return point_ind_to_pixel_dict
