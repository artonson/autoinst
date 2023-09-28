from datasets.abstract_dataset import AbstractDataset
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np 
import nuscenes.utils.geometry_utils as geoutils
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion
from functools import reduce
from typing import Tuple, List, Dict
import os.path as osp
import pykitti
import os 

class NuScenesDataset(AbstractDataset):
    def __init__(self, dataset_path,sequence, image_instances_path):
        self.dataset_path = dataset_path
        self.dataset = NuScenes(version='v1.0-mini',dataroot=dataset_path,verbose=True)
        if sequence > len(self.dataset.scene): 
            raise ValueError('Invalid sequence number')
        self.scene = self.dataset.scene[sequence]
        self.tokens = {'LIDAR_TOP':[]}
        self.get_files()
        

    def get_files(self):
        next_sample = self.scene['first_sample_token']
        while next_sample != '':
            #Current sample data
            sample = self.dataset.get('sample',next_sample)
            #Get token for the next sample
            next_sample = sample['next']

            #Get lidar, semantic and panoptic filenames
            lidar_token = sample['data']['LIDAR_TOP']
            self.tokens['LIDAR_TOP'].append(lidar_token)
            
            #velodyne bin file
        print(self.tokens)
            
    def apply_transform(self,points, pose):
        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        return np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
    
    
    def __parse_poses(self):
        poses = None

        return poses
    
    def __len__(self):
        return len(self.tokens['LIDAR_TOP'])
    
    def get_pose(self, index,sensor_name='LIDAR_TOP'):
        if sensor_name == 'LIDAR_TOP': 
            return self.get_lidar_pose(index)

    def get_lidar_pose(self, index):
        lidar_token = self.tokens['LIDAR_TOP'][index]
        lidar_data = self.dataset.get('sample_data', lidar_token)
        egopose_data = self.dataset.get('ego_pose',  lidar_data["ego_pose_token"])
            
        pose_car = geoutils.transform_matrix(egopose_data["translation"], Quaternion(egopose_data['rotation']))
        car_to_velo = self.get_lidar_calib(index)
        pose = np.dot(pose_car, car_to_velo)
        return pose

    def get_point_cloud(self, index,intensity=False,pose_correction=False):
        lidar_token = self.tokens['LIDAR_TOP'][index]
        lidar_data = self.dataset.get('sample_data', lidar_token)
        scan = np.fromfile(os.path.join(self.dataset_path, lidar_data["filename"]), dtype=np.float32)
        #Save scan
        points = scan.reshape((-1, 5))[:, :4]
        if pose_correction == True:
            pose = self.get_lidar_pose(index)
            points = self.apply_transform(points,pose)
            return points
        
        if intensity == False:
            points = points[:,:3]
        return points 
    
    def get_lidar_calib(self,index): 
        lidar_token = self.tokens['LIDAR_TOP'][index]
        lidar_data = self.dataset.get('sample_data', lidar_token)
        calib_data = self.dataset.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        car_to_velo = geoutils.transform_matrix(calib_data["translation"], Quaternion(calib_data['rotation']))
        return car_to_velo

    def get_panoptic_labels(self,index):
        lidar_token = self.tokens['LIDAR_TOP'][index]
        sem_lab_f = self.dataset.get('lidarseg',lidar_token)['filename']
        sem_lab = np.fromfile(os.path.join(self.dataset_path,sem_lab_f),dtype=np.uint8)
        pan_lab_f = self.dataset.get('panoptic',lidar_token)['filename']
        pan_lab = np.load(os.path.join(self.dataset_path,pan_lab_f))['data']
        #sem labels from panoptic labels
        sem_lab2 = (pan_lab // 1000).astype(np.uint8)
        #ins labels from panoptic labels
        ins_lab = pan_lab % 1000
        #Kitti style panoptic labels for the point cloud
        panoptic_labels = sem_lab.reshape(-1, 1) + ((ins_lab.astype(np.uint32) << 16) & 0xFFFF0000).reshape(-1,1)
        return panoptic_labels.reshape(-1,)
    
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
