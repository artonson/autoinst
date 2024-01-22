import os
import yaml
import shutil
from easydict import EasyDict as edict
from os.path import join
import torch
from models.mask_model import MaskPS
import numpy as np
import open3d as o3d
from visualization_utils import color_pcd_by_labels
from point_cloud_utils import kDTree_1NN_feature_reprojection


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


class RefinerModel():

    def __init__(self) -> None:

        model_cfg = edict(
            yaml.safe_load(
                open(
                    join(
                        getDir(__file__),
                        "models/config/model.yaml"))))

        backbone_cfg = edict(
            yaml.safe_load(
                open(
                    join(
                        getDir(__file__),
                        "models/config/backbone.yaml"))))

        decoder_cfg = edict(
            yaml.safe_load(
                open(
                    join(
                        getDir(__file__),
                        "models/config/decoder.yaml"))))
        cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
        cfg.EVALUATE = True

        self.model = MaskPS(cfg)
        w = "/home/cedric/unsup_segmentation/MaskPLS/mask_pls/experiments/mask_pls_oversegmented/lightning_logs/version_110/checkpoints/mask_pls_oversegmented_epoch=05.ckpt"
        w = torch.load(w, map_location="cpu")
        self.model.load_state_dict(w["state_dict"])
        self.model.cuda()
        self.model.backbone.train()

    def uniform_down_sample_with_indices(self, points, every_k_points):
        # Create a new point cloud for the downsampled output

        # List to hold the indices of the points that are kept
        indices = []

        # Iterate over the points and keep every k-th point
        for i in range(0, points.shape[0], every_k_points):
            indices.append(i)

        return indices

    def downsample_chunk(self, points):
        num_points_to_sample = 60000
        every_k_points = int(
            points.shape[0] /
            num_points_to_sample)
        indeces = self.uniform_down_sample_with_indices(
            points, every_k_points)
 

        return points[indeces]

    def forward_point_cloud(self, pcd_full):
        pcd_minor = o3d.geometry.PointCloud()
        minor_points = self.downsample_chunk(np.asarray(pcd_full.points))
        pcd_minor.points = o3d.utility.Vector3dVector(minor_points)
        xyz = minor_points.copy()
        mean_x = xyz[:, 0].mean()
        mean_y = xyz[:, 1].mean()
        xyz[:, 0] -= mean_x
        xyz[:, 1] -= mean_y
        intensity = np.ones((xyz.shape[0]))

        feats = np.concatenate(
            (xyz, np.expand_dims(
                intensity, axis=1)), axis=1)
        x = {}
        x['feats'] = [feats]
        x['pt_coord'] = [xyz]

        sem_pred, ins_pred, max_confs = self.model(x)
        return ins_pred, pcd_minor
        
    def forward_and_project(self,pcd_full):
        ins_pred, pcd_minor = self.forward_point_cloud(pcd_full)
        pcd_minor = color_pcd_by_labels(pcd_minor,ins_pred[0])
        pcd_colors = np.zeros_like(np.asarray(pcd_full.points))
        #o3d.visualization.draw_geometries([pcd_minor])
        colors_new = kDTree_1NN_feature_reprojection(pcd_colors,pcd_full,np.asarray(pcd_minor.colors),pcd_minor)
        
        pcd_full.colors = o3d.utility.Vector3dVector(colors_new)
        return pcd_full
        
        
        
        


if __name__ == "__main__":
    model = RefinerModel()
    pcd = o3d.io.read_point_cloud(
        "/home/cedric/unsup_3d_instances/point-to-pixel-mapping/out_kitti_instance/000275.pcd")
    # o3d.visualization.draw_geometries([pcd]) 
    pcd_full = model.forward_and_project(pcd)
    o3d.visualization.draw_geometries([pcd_full])

