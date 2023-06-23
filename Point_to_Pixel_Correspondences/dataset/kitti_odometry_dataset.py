### Copied from https://github.com/artonson/dense_lidar_recon ###

import os
from dataclasses import dataclass
from typing import Union

import numpy as np
import pykitti
from nptyping import Float, NDArray, Shape
from PIL import Image

from dataset.dataset import Dataset, cache_points
from dataset.dataset_config import DatasetConfig
from dataset.dataset_entry import DatasetEntry
from dataset.types import DatasetPathLike


@dataclass
class KittiOdometryDatasetConfig(DatasetConfig):
    # to correct the calibration of KITTI's HDL-64 scan
    correct_scan_calibration: bool


class KittiOdometryDataset(Dataset):
    def __init__(self, config: KittiOdometryDatasetConfig, seq_num: int) -> None:
        """
        Dataset class for Kitti Odometry dataset

        Requires KITTI `velodyne laser data`, `color`, `ground truth poses`,
        `calibration files` and `semantic labels`. Please extract all data into
        the same folder. Dataset path should be the folder containing the
        `sequences` folder.

        Optionally, you can use SuMa poses as the ground truth

        Download link for KITTI: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
        Download link for KITTI labels: http://semantic-kitti.org/dataset.html
        Download link for SuMa Poses: http://jbehley.github.io/projects/surfel_mapping/

        Args:
            config (KittiDatasetConfig): dataset config
            seq_num (int): dataset sequence number
        """
        # parse inputs
        self.config = config
        self.seq_str: str = str(seq_num).zfill(2)
        self.ds_path: DatasetPathLike = self.config.dataset_path
        self.sequence_path: os.PathLike = os.path.join(
            self.ds_path, "sequences", self.seq_str, ""
        )

        # class members
        self.camera_names = ("cam0", "cam1", "cam2", "cam3")
        self.dataset: pykitti.odometry = pykitti.odometry(self.ds_path, self.seq_str)
        self._poses = self.__parse_poses()

    def __parse_poses(self) -> NDArray[Shape["*, 4, 4"], Float]:
        t_cam_velo = self.dataset.calib.T_cam0_velo
        t_velo_cam = np.linalg.inv(t_cam_velo)
        poses = t_velo_cam @ self.dataset.poses @ t_cam_velo

        return poses

    def __len__(self) -> int:
        return len(self._poses)

    def get_pose(self, index: int) -> NDArray[Shape["4, 4"], Float]:
        return self._poses[index]

    @cache_points
    def get_point_cloud(self, index: int) -> NDArray[Shape["*, 3"], Float]:
        """
        Retrieves the lidar scan of the specified index

        Args:
            index (int): scan index

        Returns:
            NDArray[Shape["*, 3"], Float]: (N, 3) homogeneous points
        """

        points = self.dataset.get_velo(index)

        if self.config.correct_scan_calibration:
            points = self._correct_scan_calibration(points)

        return points

    def get_image(self, camera_name: str, index: int) -> Union[Image.Image, None]:
        """
        Retrieves the frame of the specified index and camera or None if not
        exists.

        Args:
            camera_name (str): name of the camera (cam0, cam1, cam2, cam3)
            index (int): frame index, from 0 to size of the sequence

        Returns:
            Union[Image.Image, None]: corresponding image or None
        """
        camera_func = {
            "cam0": (self.dataset.cam0_files, self.dataset.get_cam0),
            "cam1": (self.dataset.cam1_files, self.dataset.get_cam1),
            "cam2": (self.dataset.cam2_files, self.dataset.get_cam2),
            "cam3": (self.dataset.cam3_files, self.dataset.get_cam3),
        }

        files, get = camera_func[camera_name]
        return get(index) if len(files) > index else None

    @staticmethod
    def _correct_scan_calibration(scan: np.ndarray):
        """Corrects the calibration of KITTI's HDL-64 scan.

        Taken from PyLidar SLAM
        """
        xyz = scan[:, :3]
        n = scan.shape[0]
        z = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (n, 1))
        axes = np.cross(xyz, z)
        # Normalize the axes
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        theta = 0.205 * np.pi / 180.0

        # Build the rotation matrix for each point
        c = np.cos(theta)
        s = np.sin(theta)

        u_outer = axes.reshape(n, 3, 1) * axes.reshape(n, 1, 3)
        u_cross = np.zeros((n, 3, 3), dtype=np.float32)
        u_cross[:, 0, 1] = -axes[:, 2]
        u_cross[:, 1, 0] = axes[:, 2]
        u_cross[:, 0, 2] = axes[:, 1]
        u_cross[:, 2, 0] = -axes[:, 1]
        u_cross[:, 1, 2] = -axes[:, 0]
        u_cross[:, 2, 1] = axes[:, 0]

        eye = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        rotations = c * eye + s * u_cross + (1 - c) * u_outer
        corrected_scan = np.einsum("nij,nj->ni", rotations, xyz)
        return corrected_scan

    def __getitem__(self, index: int) -> DatasetEntry:
        entry = DatasetEntry(
            index,
            self.get_pose(index),
            self.get_point_cloud(index),
            {
                cam_name: self.get_image(cam_name, index)
                for cam_name in self.camera_names
            },
        )

        if self.config.filters:
            entry = self.config.filters(entry, self)

        return entry
