from point_cloud_utils import get_pcd, transform_pcd
from merge_pointclouds import merge_pointclouds
import numpy as np

def aggregate_pointcloud(dataset, ind_start, ind_end, clip_to_imageframe=False, remove_backward_points=False, return_poses=False, cam="cam2"):
    '''
    Args:
        dataset:                dataset object
        ind_start:              index of first point cloud
        ind_end:                index of last point cloud
        sequence_length:        number of point clouds to merge
        clip_to_imageframe:     if True, only points that are in the image frame of the respective time step are kept.
                                This implicitly includes removing points that are behind the camera.
        remove_backward_points: if True, only points that are in front of the camera are kept.
    Returns:
        pcd_merge:          merged point cloud
        T_pcd:              The transformation matrix of the merged point cloud
    '''

    first_pose = dataset.get_pose(ind_start)
    poses = [first_pose]
    pcd_merge = None

    for points_index in range(ind_start, ind_end):
        pcd_o3d = get_pcd(dataset[points_index].point_cloud)

        if clip_to_imageframe:
            T_lidar2leftcam, K_leftcam = dataset.get_calibration_matrices(cam)
            (img_width, img_height) = dataset.get_image(cam, points_index).size

            pcd = np.array(pcd_o3d.points)
            points_camframe = transform_pcd(pcd, T_lidar2leftcam)
            points_imgframe = K_leftcam @ points_camframe.transpose()
            points_imgframe[:2, :] /= points_imgframe[2, :]
            inds = np.where((points_imgframe[0, :] < img_width) & (points_imgframe[0, :] >= 0) &
                            (points_imgframe[1, :] < img_height) & (points_imgframe[1, :] >= 0) &
                            (points_imgframe[2, :] > 0))[0]
            pcd = pcd[inds]
            pcd_o3d = get_pcd(pcd)

        elif remove_backward_points:
            T_lidar2leftcam, K_leftcam = dataset.get_calibration_matrices(cam)
            pcd = np.array(pcd_o3d.points)
            points_camframe = transform_pcd(pcd, T_lidar2leftcam)
            points_imgframe = K_leftcam @ points_camframe.transpose()
            inds = np.where(points_imgframe[2, :] > 0)[0]
            pcd = pcd[inds]
            pcd_o3d = get_pcd(pcd)

        if pcd_merge is None:
            pcd_merge = pcd_o3d
        else:
            pose = dataset.get_pose(points_index)
            poses.append(pose)
            pcd_merge = merge_pointclouds(pcd_merge, pcd_o3d, first_pose, pose)

    if return_poses:
        return pcd_merge, first_pose, poses
    else:
        return pcd_merge, first_pose