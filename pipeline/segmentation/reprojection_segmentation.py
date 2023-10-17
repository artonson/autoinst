import numpy as np
import open3d as o3d

from segmentation.abstract_segmentation import AbstractSegmentation

class ReprojectionSegmentation(AbstractSegmentation):
    def __init__(self, dataset):
        self.dataset = dataset

    def segment_instances(self, index):
        cam_name = 'cam2'
        masks = self.dataset.get_image_instances(cam_name, index)
        image_labels = self.__masks_to_image(masks)
        points = self.dataset.get_point_cloud(index)
        labels = np.zeros(points.shape[0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        T, K = self.dataset.get_calibration_matrices(cam_name)
        pcd.transform(T)

        p2pix = self.__points_to_pixels(np.asarray(pcd.points), K, (image_labels.shape[1], image_labels.shape[0]))
        for ind, value in p2pix.items():
            labels[ind] = image_labels[value[1], value[0]]

        return labels
        
    def __masks_to_image(self, masks):
        image_labels = np.zeros(masks[0]['segmentation'].shape)
        for i, mask in enumerate(masks):
            image_labels[mask['segmentation']] = i + 1

        return image_labels

    def __points_to_pixels(self, points, cam_intrinsics, img_shape):
        img_width, img_height = img_shape
        points_proj = cam_intrinsics @ points.T
        points_proj[:2, :] /= points_proj[2, :]
        points_coord = points_proj.T
        
        inds = np.where(
            (points_coord[:, 0] < img_width) & (points_coord[:, 0] >= 0) &
            (points_coord[:, 1] < img_height) & (points_coord[:, 1] >= 0) &
            (points_coord[:, 2] > 0)
        )[0]

        points_ind_to_pixels = {}
        for ind in inds:
            points_ind_to_pixels[ind] = points_coord[ind][:2].astype(int)

        return points_ind_to_pixels
