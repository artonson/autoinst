from datasets.abstract_dataset import AbstractDataset
#from nuscenes import NuScenes
#from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np 
#from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion
from functools import reduce
from typing import Tuple, List, Dict
import os.path as osp
import pykitti
import os 

'''
Due to the use of the conversion tool "nuscenes2kitti" the dataset is currently the 
same as the kitti dataset. 
'''


class NuScenesDataset(AbstractDataset):
    def __init__(self, dataset_path, sequence, image_instances_path):
        self.dataset_path = dataset_path
        self.sequence = sequence
        self.dataset = pykitti.odometry(dataset_path, sequence)
        self.image_instances_path = image_instances_path
        self.poses = self.__parse_poses()

    def __parse_poses(self):
        t_cam_velo = self.dataset.calib.T_cam0_velo
        t_velo_cam = np.linalg.inv(t_cam_velo)
        try:    
            poses = t_velo_cam @ self.dataset.poses @ t_cam_velo
        except:
            poses = None

        return poses

    def get_pose(self, index):
        return self.poses[index]

    def get_point_cloud(self, index,intensity=False):
        points = self.dataset.get_velo(index)
        if intensity == False : 
            points = points[:, :3]
        
        return points
    
    def get_available_cameras(self):
        return ['cam2', 'cam3']

    def get_image(self, cam_name, index):
        camera_func = {
            'cam2': (self.dataset.cam2_files, self.dataset.get_cam2),
            'cam3': (self.dataset.cam3_files, self.dataset.get_cam3),
        }

        files, get = camera_func[cam_name]
        return get(index) if len(files) > index else None
    
    def get_image_instances(self, cam_name, index):
        masks_path = os.path.join(self.image_instances_path, cam_name, '{}.npz'.format(str(index).zfill(6)))
        return np.load(masks_path, allow_pickle=True)['masks']

    def get_calibration_matrices(self, cam_name):
        if cam_name == 'cam2':
            T = self.dataset.calib.T_cam2_velo
            K = self.dataset.calib.K_cam2
        elif cam_name == 'cam3':
            T = self.dataset.calib.T_cam3_velo
            K = self.dataset.calib.K_cam3
        else:
            raise ValueError('Invalid camera name')

        return T, K