import os
import yaml
import shutil
from easydict import EasyDict as edict
from os.path import join
import torch
from utils.maskpls.mask_model import MaskPS
import numpy as np
import open3d as o3d
from utils.visualization_utils import  generate_random_colors
from utils.point_cloud.point_cloud_utils import kDTree_1NN_feature_reprojection
import copy
import json
from config import *


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


class RefinerModel:

    def __init__(self, dataset="kitti") -> None:

        model_cfg = edict(
            yaml.safe_load(
                open(join(getDir(__file__), "config/model.yaml"))
            )
        )

        backbone_cfg = edict(
            yaml.safe_load(
                open(join(getDir(__file__), "config/backbone.yaml"))
            )
        )

        decoder_cfg = edict(
            yaml.safe_load(
                open(join(getDir(__file__), "config/decoder.yaml"))
            )
        )
        cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
        cfg.EVALUATE = True

        self.model = MaskPS(cfg)
        w = MASKPLS_weights
        print("using weights", w)
        w = torch.load(w, map_location="cpu")
        self.model.load_state_dict(w["state_dict"])
        self.model.cuda()
        self.model.backbone.train()
        self.confs_dict = {}

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
        every_k_points = int(points.shape[0] / num_points_to_sample)
        if every_k_points == 0:
            every_k_points = 1
        indeces = self.uniform_down_sample_with_indices(points, every_k_points)

        return points[indeces]

    def forward_point_cloud(self, pcd_full):
        pcd_minor = o3d.geometry.PointCloud()
        minor_points = self.downsample_chunk(np.asarray(pcd_full.points))
        pcd_minor.points = o3d.utility.Vector3dVector(minor_points)
        xyz = minor_points.copy()
        mean_x = xyz[:, 0].mean()
        mean_y = xyz[:, 1].mean()
        mean_z = xyz[:, 2].mean()
        xyz[:, 0] -= mean_x
        xyz[:, 1] -= mean_y
        xyz[:, 2] -= mean_z
        intensity = np.ones((xyz.shape[0]))

        feats = np.concatenate((xyz, np.expand_dims(intensity, axis=1)), axis=1)
        x = {}
        x["feats"] = [feats]
        x["pt_coord"] = [xyz]
        with torch.no_grad():
            sem_pred, ins_pred, max_confs = self.model(x)
        return ins_pred, pcd_minor, max_confs

    def color_pcd_by_labels(self, pcd, labels, confs, colors=None):

        if colors == None:
            colors = generate_random_colors(2000)
        pcd_colored = copy.deepcopy(pcd)
        pcd_colors = np.zeros(np.asarray(pcd.points).shape)
        unique_labels = list(np.unique(labels))

        largest_cluster_idx = -10
        largest = 0

        for i in unique_labels:
            idcs = np.where(labels == i)
            idcs = idcs[0]
            if idcs.shape[0] > largest:
                largest = idcs.shape[0]
                largest_cluster_idx = i

        for i in unique_labels:
            if i == -1:
                continue
            idcs = np.where(labels == i)
            idcs = idcs[0]
            cur_confs = confs[idcs].mean()
            if i == largest_cluster_idx:
                pcd_colors[idcs] = np.array([0, 0, 0])
                self.confs_dict[str(0) + "|" + str(0) + "|" + str(0)] = cur_confs.item()
            else:
                col_val = np.array(colors[unique_labels.index(i)])
                pcd_colors[idcs] = col_val
                self.confs_dict[
                    str(col_val[0]) + "|" + str(col_val[1]) + "|" + str(col_val[2])
                ] = cur_confs.item()

        pcd_colored.colors = o3d.utility.Vector3dVector(pcd_colors / 255)
        return pcd_colored

    def forward_and_project(self, pcd_full):
        ins_pred, pcd_minor, max_confs = self.forward_point_cloud(pcd_full)
        pcd_minor = self.color_pcd_by_labels(pcd_minor, ins_pred[0], max_confs[0])
        pcd_colors = np.zeros_like(np.asarray(pcd_full.points))
        colors_new = kDTree_1NN_feature_reprojection(
            pcd_colors, pcd_full, np.asarray(pcd_minor.colors), pcd_minor
        )

        pcd_full.colors = o3d.utility.Vector3dVector(colors_new)
        return pcd_full
    
    def label_to_conf(self,merge_ncuts,instance_preds):
        label_to_confidence = {}
        pcd_cols = np.asarray(merge_ncuts.colors)
        for label in list(np.unique(instance_preds)):
                idcs = np.where(instance_preds == label)[0]
                cur_color = pcd_cols[idcs[0]]
                key = (
                    str(int(cur_color[0] * 255))
                    + "|"
                    + str(int(cur_color[1] * 255))
                    + "|"
                    + str(int(cur_color[2] * 255))
                )
                label_to_confidence[label] = self.confs_dict[key]
        return label_to_confidence
    
    def store_conf_dict(self,data_store_folder,name,seq,cur_idx):
        with open(
                data_store_folder
                + name
                + "_confs"
                + str(seq)
                + "_"
                + str(cur_idx)
                + ".json",
                "w",
            ) as fp:
                json.dump(self.confs_dict, fp)

if __name__ == "__main__":
    model = RefinerModel()
    for i in range(20):
        print(i)
        pcd = o3d.io.read_point_cloud(
            "/home/cedric/unsup_3d_instances/point-to-pixel-mapping/out_kitti_instance2/000275.pcd"
        )
        pcd_full = model.forward_and_project(pcd)
        o3d.visualization.draw_geometries([pcd_full])
