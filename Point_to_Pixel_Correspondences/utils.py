### Inspired / Copied from https://github.com/darylclimb/cvml_project/blob/master/projections/lidar_camera_projection/utils.py ###

import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_calib_file(filepath, data_dict):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """

    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data_dict[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data_dict

def project_velo_to_cam2(calib):

    '''
    odometry development kit:
    calib.txt: Calibration data for the cameras: P0/P1 are the 3x4 projection
    matrices after rectification. Here P0/P2 (grey / color) denotes the left and P1/P3 (grey / color) denotes the
    right camera. Tr transforms a point from velodyne coordinates into the
    left rectified camera coordinate system. In order to map a point X from the
    velodyne scanner to a point x in the i'th image plane, you thus have to
    transform it like:

    x = P_i * Tr * X
    '''
    
    P_velo2cam_ref = np.vstack((calib['Tr'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ P_velo2cam_ref
    return proj_mat, P_velo2cam_ref

def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        points:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]

def filter_points_fov(pc_2d, pc_xyz, img_width, img_height):
    # Filter lidar points to be within image FOV
    inds = np.where((pc_2d[0, :] < img_width) & (pc_2d[0, :] >= 0) &
                    (pc_2d[1, :] < img_height) & (pc_2d[1, :] >= 0) &
                    (pc_xyz[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pc_2d[:, inds]   #xy
    return imgfov_pc_pixel, inds


def unite_pc_and_img(imgfov_pc_pixel, img, pc_velo, inds, proj_velo2cam2):
    """
    Apply the perspective projection
    Args:
        imgfov_pc_pixel:   2D Point Cloud projected on image plane
        img:               RGB Image
        pc_velo:           Point Cloud with xyz coordinates
        inds:              Indices of points of point cloud that lie within the image plane
        proj_velo2cam2:    Projection matrix from velodyne coordinate systeme to cam2 coordinate system
    """

    ### Needed for coloring according to depth
    imgfov_pc_velo = pc_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    img_with_pc = img.copy()
    
    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        id = min(int(255), int(640.0 / depth))
        color = cmap[id, :]
        cv2.circle(img_with_pc, (int(np.round(imgfov_pc_pixel[0, i])),
                    int(np.round(imgfov_pc_pixel[1, i]))),
                    2, color=tuple(color), thickness=-1)
    return img_with_pc










'''
def project_cam2_to_velo(calib):
    R_ref2rect = np.eye(4)
    R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    R_ref2rect_inv = np.linalg.inv(R_ref2rect)  # rect2ref_cam

    # inverse rigid transformation
    velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    P_cam_ref2velo = np.linalg.inv(velo2cam_ref)

    proj_mat = P_cam_ref2velo @ R_ref2rect_inv
    return proj_mat





def project_camera_to_lidar(points, proj_mat):
    """
    Args:
        points:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]

    Returns:
        points in lidar coordinate:     [3, npoints]
    """
    num_pts = points.shape[1]
    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    return points[:3, :]

'''