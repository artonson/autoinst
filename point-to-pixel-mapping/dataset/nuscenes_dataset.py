### Copied from https://github.com/artonson/dense_lidar_recon ###

import os
from dataclasses import dataclass
from typing import Union
import zlib

import numpy as np

from nptyping import Float, NDArray, Shape
from PIL import Image
import pykitti.utils as utils


from nuscenes import NuScenes
from nuscenes.utils.data_io import load_bin_file
from pyquaternion import Quaternion

from point_cloud_utils import transformation_matrix

from dataset.dataset import Dataset, cache_points
from dataset.dataset_config import DatasetConfig
from dataset.dataset_entry import DatasetEntry
from dataset.types import DatasetPathLike


@dataclass
class nuScenesDatasetConfig(DatasetConfig):
    # to correct the calibration of KITTI's HDL-64 scan
    cache: bool

class nuScenesOdometryDataset(Dataset):
    def __init__(self, config: nuScenesDatasetConfig, seq_num: int) -> None:

        # parse inputs
        self.nuscenes = True
        self.config = config
        self.seq_num = seq_num
        self.ds_path: DatasetPathLike = self.config.dataset_path

        self.dataset = NuScenes(version='v1.0-mini', dataroot=self.ds_path, verbose=True)
        self.scene = self.dataset.scene[self.seq_num]

        self.sample_tokens = self.__parse_tokens()

        # class members
        self.camera_names = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")

        self.sam_label_path: os.PathLike = os.path.join(
            self.ds_path, "outputs", self.config.sam_folder_name)
        self.dinov2_features_path: os.PathLike = os.path.join(
            self.ds_path, "outputs",self.config.dinov2_folder_name)
        self.dist_threshold = config.dist_threshold

        self.tarl_features_path: os.PathLike = os.path.join(
            self.ds_path, "outputs/TARL/LIDAR_TOP/")

        self._poses = self.__parse_poses()

    
    def __parse_tokens(self):
        sample_tokens = []
        next_sample_token = self.scene['first_sample_token']
        while next_sample_token != '':
            next_sample = self.dataset.get('sample', next_sample_token)
            sample_tokens.append(next_sample['token'])
            next_sample_token = next_sample['next']
        return sample_tokens
    
    def __len__(self) -> int:
        return len(self.sample_tokens)
    
    def __parse_poses(self) -> NDArray[Shape["*, 4, 4"], Float]:
       
        poses = []

        for token in self.sample_tokens:
            sample = self.dataset.get('sample', token)
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.dataset.get('sample_data', lidar_token)
            calib_data = self.dataset.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

            rotation = Quaternion(calib_data['rotation']).rotation_matrix
            translation = np.array(calib_data["translation"])
            lidar2vehicle = transformation_matrix(rotation, translation)

            egopose_data = self.dataset.get('ego_pose', lidar_data["ego_pose_token"])
            rotation = Quaternion(egopose_data['rotation']).rotation_matrix
            translation = np.array(egopose_data["translation"])
            vehicle2world = transformation_matrix(rotation, translation)

            lidar2world = vehicle2world @ lidar2vehicle

            poses.append(lidar2world)

        return poses

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

        sample_token = self.sample_tokens[index]
        sample = self.dataset.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.dataset.get('sample_data', lidar_token)
        scan = np.fromfile(os.path.join(self.ds_path, lidar_data["filename"]), dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :3]

        return points

    def get_intensity(self, index: int) -> NDArray[Shape["*, 1"], Float]:
        """
        Retrieves the lidar scan of the specified index

        Args:
            index (int): scan index

        Returns:
            NDArray[Shape["*, 3"], Float]: (N, 3) homogeneous points
        """

        sample_token = self.sample_tokens[index]
        sample = self.dataset.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.dataset.get('sample_data', lidar_token)
        scan = np.fromfile(os.path.join(self.ds_path, lidar_data["filename"]), dtype=np.float32)
        intensity = scan.reshape((-1, 5))[:, 3]

        return intensity

    def get_panoptic_labels(self, index: int):


        sample_token = self.sample_tokens[index]
        sample = self.dataset.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        pantopic_data = self.dataset.get('panoptic', lidar_token)
        label_file_path = os.path.join(self.ds_path, pantopic_data["filename"])
        panoptic = load_bin_file(label_file_path, 'panoptic')
        panoptic = panoptic[np.newaxis].T

        return panoptic

    def get_semantic_labels(self, index: int):

        sample_token = self.sample_tokens[index]
        sample = self.dataset.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidarseg_data = self.dataset.get('lidarseg', lidar_token)
        label_file_path = os.path.join(self.ds_path, lidarseg_data["filename"])
        lidarseg = load_bin_file(label_file_path, 'lidarseg')
        lidarseg = lidarseg[np.newaxis].T

        return lidarseg

    def get_instance_labels(self, index: int):
        pan_lab = self.get_panoptic_labels(index)
        ins_lab = pan_lab % 1000
        return ins_lab

    def get_image(self, camera_name: str, index: int) -> Union[Image.Image, None]:
        """
        Retrieves the frame of the specified index and camera or None if not
        exists.

        Args:
            camera_name (str): name of the camera ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
            index (int): frame index, from 0 to size of the sequence

        Returns:
            Union[Image.Image, None]: corresponding image or None
        """

        if camera_name not in self.camera_names:
            raise ValueError("Invalid camera name")

        sample_token = self.sample_tokens[index]
        sample = self.dataset.get('sample', sample_token)
        camera_token = sample['data'][camera_name]
        camera_data = self.dataset.get('sample_data', camera_token)
        image = Image.open(os.path.join(self.ds_path, camera_data["filename"]))

        return image
    
    def get_sam_label(self, camera_name: str, index: int) -> Union[Image.Image, None]:
        """
        Retrieves the SAM label of the specified index and camera

        Args:
            camera_name (str): name of the camera (cam0, cam1, cam2, cam3)
            index (int): frame index, from 0 to size of the sequence

        Returns:
            corresponding label
        """
        available_cams = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")

        if camera_name not in available_cams:
            raise ValueError("Invalid camera name. SAM labels only available for CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT")

        sample_token = self.sample_tokens[index]
        sample = self.dataset.get('sample', sample_token)
        camera_token = sample['data'][camera_name]
        camera_data = self.dataset.get('sample_data', camera_token)

        file = os.path.join(self.sam_label_path, camera_name, camera_data["filename"].split('/')[-1].split('.')[0] + '.jpg')

        return utils.load_image(file, mode='RGB')

    
    def get_sam_mask(self, camera_name: str, index: int) -> Union[Image.Image, None]:
        """
        Retrieves the SAM label of the specified index and camera

        Args:
            camera_name (str): name of the camera (cam0, cam1, cam2, cam3)
            index (int): frame index, from 0 to size of the sequence

        Returns:
            corresponding label
        """
        available_cams = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")

        if camera_name not in available_cams:
            raise ValueError("Invalid camera name. SAM labels only available for CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT")

        sample_token = self.sample_tokens[index]
        sample = self.dataset.get('sample', sample_token)
        camera_token = sample['data'][camera_name]
        camera_data = self.dataset.get('sample_data', camera_token)

        file = os.path.join(self.sam_label_path, camera_name, camera_data["filename"].split('/')[-1].split('.')[0] + '.npz')

        return np.load(file, allow_pickle=True)['masks']


    def get_dinov2_features(self, camera_name: str, index: int):
        """
        Retrieves the dinov2 features of the specified index and camera

        Args:
            camera_name (str): name of the camera
            index (int): frame index, from 0 to size of the sequence

        Returns:
            corresponding dinov2 features
        """
        available_cams = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")

        if camera_name not in available_cams:
            raise ValueError("Invalid camera name. SAM labels only available for CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT")

        sample_token = self.sample_tokens[index]
        sample = self.dataset.get('sample', sample_token)
        camera_token = sample['data'][camera_name]
        camera_data = self.dataset.get('sample_data', camera_token)

        dinov2_features = np.load(os.path.join(self.dinov2_features_path, camera_name, 
                                                camera_data["filename"].split('/')[-1].split('.')[0] + '.npz'), allow_pickle=True)["feature_map"]

        return dinov2_features

    def get_tarl_features(self, index: int):
        """
        Retrieves the tarl features for the specified point cloud

        Args:
            index (int): frame index, from 0 to size of the sequence

        Returns:
            corresponding tarl features
        """

        sample_token = self.sample_tokens[index]
        sample = self.dataset.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.dataset.get('sample_data', lidar_token)

        ##Load TARL labels 
        compressed_file = os.path.join(self.tarl_features_path + lidar_data["filename"].split('/')[-1].split('.')[0] + '.bin')
        with open(compressed_file, 'rb') as f_in:
            compressed_data = f_in.read()
        decompressed_data = zlib.decompress(compressed_data)
        loaded_array = np.frombuffer(decompressed_data, dtype=np.float32)
        tarl_dim = 96
        point_features = loaded_array.reshape(-1,tarl_dim) #these are TARL per point features

        return point_features
    
       
    def get_calibration_matrices(self, cam: str):
        """
        Retrieves the extriniscs and intrinsics matrix of the specified camera
        """
        if cam not in self.camera_names:
            raise ValueError("Invalid camera name")

        sample_token = self.sample_tokens[0]
        sample = self.dataset.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.dataset.get('sample_data', lidar_token)
        cs_record = self.dataset.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        rotation = Quaternion(cs_record['rotation']).rotation_matrix
        translation = np.array(cs_record['translation'])
        T_lidar2ego = transformation_matrix(rotation, translation)

        camera_token = sample["data"][cam]
        camera_data = self.dataset.get('sample_data', camera_token)
        cs_record = self.dataset.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
        K = np.array(cs_record['camera_intrinsic'])

        rotation = Quaternion(cs_record['rotation']).rotation_matrix.T
        translation = - rotation @ np.array(cs_record['translation'])
        T_ego2cam = transformation_matrix(rotation, translation)

        T_lidar2cam  = T_ego2cam @ T_lidar2ego

        return T_lidar2cam, K

    def __getitem__(self, index: int) -> DatasetEntry:
        entry = DatasetEntry(
            index,
            self.get_pose(index),
            self.get_point_cloud(index),
            self.get_intensity(index),
            self.get_panoptic_labels(index),
            self.get_semantic_labels(index),
            self.get_instance_labels(index),
            {
                cam_name: self.get_image(cam_name, index)
                for cam_name in self.camera_names
            },
        )

        if self.config.filters:
            entry = self.config.filters(entry, self)

        return entry
