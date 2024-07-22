import gc
import os
import zlib

import MinkowskiEngine as ME
import numpy as np
import torch
from tqdm import tqdm

from minkunet import MinkUNet
from utils import latent_features, numpy_to_sparse_tensor, set_deterministic


class TarlExtractor:
    def __init__(self, args, vis=False):
        self.vis = vis
        self.out_dir = args.output_path
        self.input_dir = args.input_path
        self.dataset = args.dataset
        self.label_clusters = False
        if os.path.exists(self.out_dir) is False:
            os.makedirs(self.out_dir)

        self.sparse_resolution = 0.05  # according to standard TARL settings
        self.use_intensity = True

        self.minkunet = MinkUNet(
            in_channels=4 if self.use_intensity else 3,
            out_channels=latent_features["MinkUNet"],
        )

        self.minkunet.cuda()
        self.minkunet.eval()
        set_deterministic()

        model_filename = "tarl.pt"
        checkpoint = torch.load(model_filename)
        self.minkunet.load_state_dict(checkpoint["model"])
        self.max_len = 1091
        self.device = torch.device("cuda")
        torch.cuda.set_device(0)

    def extract_instance(self, pts, name):
        output_name = name.split(".")[0] + ".bin"
        fp = self.out_dir + output_name
        # if os.path.exists(fp) == True :
        #    print("output file already exists")
        #    return

        points, point_feats, mapping_idcs = self.point_set_to_coord_feats(
            pts, deterministic=True
        )
        points = np.asarray(points)[None, :]
        point_feats = np.asarray(point_feats)[None, :]
        x = numpy_to_sparse_tensor(points, point_feats)
        h = self.minkunet(x)
        minkunet_feats = h.features.cpu()  # 96 dim features of minkunet

        # create original size of point cloud features
        point_feats_minkunet = torch.zeros((pts.shape[0], 96))
        extracted_features = torch.zeros(
            (pts.shape[0], 1)
        )  # to save indices accordingly
        extracted_features[mapping_idcs] = 1
        non_mapped_idcs = (
            torch.nonzero(extracted_features == 0).squeeze().reshape(-1, 2)[:, 0]
        )

        """
        as minkunet takes in quantized points, we assign the features of
        the closest point to the points that are not mapped
        """
        point_feats_minkunet[mapping_idcs] = minkunet_feats.detach()
        distances = torch.cdist(
            torch.from_numpy(pts[non_mapped_idcs, :3]).cuda(),
            torch.from_numpy(pts[mapping_idcs, :3]).cuda(),
        )  # speedup by moving to GPU
        closest_indices = torch.argmin(distances, dim=1).cpu()
        point_feats_minkunet[non_mapped_idcs] = point_feats_minkunet[mapping_idcs][
            closest_indices
        ]

        # loop over clusters and assign the mean features to each cluster
        data_store = point_feats_minkunet.detach().numpy()
        # with open(self.out_dir + output_name, 'wb') as f_out:
        # np.savez(self.out_dir + output_name,feats=data_store)
        output_name = name.split(".")[0] + ".bin"
        with open(self.out_dir + output_name, "wb") as f_out:
            f_out.write(zlib.compress(data_store.tobytes()))

        del distances, x, h, points, non_mapped_idcs
        torch.cuda.empty_cache()
        del closest_indices, extracted_features, pts
        del data_store
        del point_feats_minkunet
        del point_feats, mapping_idcs
        gc.collect()

    def point_set_to_coord_feats(self, point_set, deterministic=False):
        p_feats = point_set.copy()
        p_coord = np.round(point_set[:, :3] / self.sparse_resolution)
        p_coord -= p_coord.min(0, keepdims=1)

        _, mapping = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True)

        return p_coord[mapping], p_feats[mapping], mapping

    def run_on_folder(self):
        all_lidars = os.listdir(self.input_dir)
        if self.dataset == "kitti":
            all_lidars = sorted(all_lidars, key=lambda x: int(x.split(".")[0]))

        # replace_file = str(1513)
        for lidar_name in tqdm(all_lidars):
                if self.dataset == "nuscenes":
                    points = np.fromfile(
                        os.path.join(self.input_dir, lidar_name), dtype=np.float32
                    ).reshape(-1, 5)[:, :4]
                elif self.dataset == "kitti":
                    points = np.fromfile(
                        os.path.join(self.input_dir, lidar_name), dtype=np.float32
                    ).reshape(-1, 4)
                else:
                    raise "Dataset" + self.dataset + "does not exist"
                self.extract_instance(points, lidar_name)
