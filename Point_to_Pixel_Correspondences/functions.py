import open3d as o3d
import numpy as np
from utils import read_calib_file, unite_pc_and_img, project_velo_to_cam2, project_to_image, filter_points_fov

def hidden_point_removal(points_np, camera: list):
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(points_np)
    print("Number of points in Point Cloud before Hidden Points removal: ", len(points_o3d.points))
    
    diameter = np.linalg.norm(np.asarray(points_o3d.get_max_bound()) - np.asarray(points_o3d.get_min_bound()))
    assert(len(camera)==3) # Is this correct? Online they just always do [0, 0, diameter], but that produces poor results
    radius = diameter * 100
    _, pt_map = points_o3d.hidden_point_removal(camera, radius)
    points_o3d_hpr = points_o3d.select_by_index(pt_map)
    print("Number of points in Point Cloud after Hidden Points removal: ", len(points_o3d_hpr.points))

    # Convert back to numpy array
    points_np_hpr = np.asarray(points_o3d_hpr.points)
    return points_np_hpr


def project_point_cloud_to_image(path_calib, points, img):    
    calib_dic = {}
    calib = read_calib_file(path_calib, calib_dic)

    img_height, img_width, _ = img.shape


    ### Get projection matrices from Velodyne to Cam2 and from Velodyne to reference Camera
    proj_velo2cam2, P_velo2cam_ref = project_velo_to_cam2(calib)

    ### Project Point Cloud to Image Frame using projection matrix
    pc_2d = project_to_image(points.transpose(), proj_velo2cam2)
    print("Number of points in 2D projected Point Cloud before FOV filter: ", pc_2d.shape[1])
    imgfov_pc_pixel, inds = filter_points_fov(pc_2d, points, img_width, img_height)
    print("Number of points in 2D projected Point Cloud after FOV filter: ", imgfov_pc_pixel.shape[1])
    # imgfov_pc_pixel has dimension [2, #points_in_image_frame], where [0,:] are x coordinates and [1,:] are y coordinates

    ### Unites the 2D projected Point Cloud with the rgb image and colors points according to depth
    img_overlay = unite_pc_and_img(imgfov_pc_pixel, img, points, inds, proj_velo2cam2)
    
    return img_overlay, inds