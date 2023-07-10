from dataclasses import dataclass
from typing import Union

import numpy as np
from nptyping import Float, NDArray, Shape
from pandaset import DataSet
from PIL import Image
from scipy.spatial.transform import Rotation as R

from dataset.dataset import Dataset, cache_points
from dataset.dataset_config import DatasetConfig
from dataset.dataset_entry import DatasetEntry
from dataset.types import DatasetPathLike
import transforms3d as t3d


@dataclass
class PandasetDatasetConfig(DatasetConfig):
    pass


class PandasetDataset(Dataset):
    def __init__(self, config: PandasetDatasetConfig, seq_num: int) -> None:
        """
        Dataset class for Pandaset dataset

        Requires Pandaset, you can download any batch provided in the website.
        Extract all batches into the same folder. Dataset path should be the
        folder containing the sequences (001, 002 ...).

        Download link for Pandaset: https://scale.com/resources/download/pandaset

        Args:
            config (PandasetDatasetConfig): dataset config
            seq_num (int): dataset sequence number
        """
        # parse inputs
        self.config = config
        self.seq_str: str = str(seq_num).zfill(3)
        self.ds_path: DatasetPathLike = self.config.dataset_path

        # class members
        self.camera_names = (
            "front_camera",
            "left_camera",
            "back_camera",
            "right_camera",
            "front_left_camera",
            "front_right_camera",
        )
        self.dataset: DataSet = DataSet(self.ds_path)

        self.sequence_obj = self.dataset[self.seq_str]
        self.sequence_obj.load()  # takes some time
        self._poses = self.__parse_poses()

    def __parse_poses(self) -> NDArray[Shape["4, 4"], Float]:
        poses = self.sequence_obj.lidar.poses

        _poses = []
        for p in poses:
            position, heading = p["position"], p["heading"]
            rot = R.from_quat([heading["x"], heading["y"], heading["z"], heading["w"]])
            pos = np.array([position["x"], position["y"], position["z"]])
            _p = np.identity(4)
            _p[:3, :3] = rot.as_matrix()
            _p[:3, 3] = pos
            _poses.append(_p)

        return np.array(_poses)

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

        points = self.sequence_obj.lidar[index].values[:, :3]

        return points

    def get_image(self, camera_name: str, index: int) -> Union[Image.Image, None]:
        """
        Retrieves the frame of the specified index and camera or None if not
        exists.

        Args:
            camera_name (str): name of the camera (see self.camera_names)
            index (int): frame index, from 0 to size of the sequence

        Returns:
            image: corresponding image
        """
        if not camera_name in self.camera_names:
            raise ValueError("Invalid camera name")
        if len(self) <= index:
            raise ValueError("Index out of range")

        images = self.sequence_obj.camera[camera_name]
        return images[index]

    def get_calibration_matrices(self, camera_name: str, index: int):
        """
        Retrieves the extriniscs and intrinsics matrix of the specified camera

        Args:
            camera_name (str): name of the camera (see self.camera_names)
            index (int): frame index, from 0 to size of the sequence

        Returns:
            T (np.ndarray): extrinsics matrix
            K (np.ndarray): intrinsics matrix
        """
        if not camera_name in self.camera_names:
            raise ValueError("Invalid camera name")
        
        ### Intrinsic matrix ###
        intrinsics = self.sequence_obj.camera[camera_name].intrinsics

        K = np.eye(3, dtype=np.float64)
        K[0, 0] = intrinsics.fx
        K[1, 1] = intrinsics.fy
        K[0, 2] = intrinsics.cx
        K[1, 2] = intrinsics.cy

        ### Extrinsic matrix ###
        # Lidar point cloud is stored in world coordinate frame, thus T = T_world2cam
    
        camera_pose = self.sequence_obj.camera[camera_name].poses[index]

        heading = camera_pose['heading']
        position = camera_pose['position']

        quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
        pos = np.array([position["x"], position["y"], position["z"]])

        T = t3d.affines.compose(np.array(pos),
                                t3d.quaternions.quat2mat(quat),
                                [1.0, 1.0, 1.0])

        T = np.linalg.inv(T)

        return T, K


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
