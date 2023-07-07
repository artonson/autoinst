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

    inds = np.where((points_imgframe[0, :] < img_width - 1) & (points_imgframe[0, :] >= 0) &
                    (points_imgframe[1, :] < img_height - 1) & (points_imgframe[1, :] >= 0) &
                    (points_imgframe[2, :] > 0)
                    )[0]

    point_ind_to_pixel_dict = {}
    for ind in inds:
        point_ind_to_pixel_dict[ind] = {}
        point_ind_to_pixel_dict[ind]['pixels'] = np.round(points_imgframe[:2, ind]).astype(int)
        
        point_ind_to_pixel_dict[ind]['depth'] = points_imgframe[2, ind]

    return point_ind_to_pixel_dict


def pixel_to_point_from_point_to_pixel(point_to_pixel: dict):
    '''
    Args:
        point_to_pixel: dict that maps point indices to pixel coordinates
    Returns:
        pixel_to_point: dict that maps pixel coordinates to point indices
    '''

    pixel_to_point = {}

    for index, point_data in point_to_pixel.items():
        pixel = point_data['pixels']
        pixel_tpl = (pixel[0], pixel[1])
        depth = point_data['depth']

        if pixel_tpl not in pixel_to_point.keys():
            # Add the pixel and point index to the dictionary
            pixel_to_point[pixel_tpl] = {'index': index, 'depth': depth}
        elif depth < pixel_to_point[pixel_tpl]['depth']:
            # Update the dictionary with the new point index and depth
            pixel_to_point[pixel_tpl] = {'index': index, 'depth': depth}
        
    return pixel_to_point