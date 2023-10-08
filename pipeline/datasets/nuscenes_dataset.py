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
from PIL import Image
import os 

class NuScenesDataset(AbstractDataset):
    def __init__(self, dataset_path,sequence, image_instances_path):
        self.dataset_path = dataset_path
        self.dataset = NuScenes(version='v1.0-mini',dataroot=dataset_path,verbose=True)
        if sequence > len(self.dataset.scene): 
            raise ValueError('Invalid sequence number')
        self.scene = self.dataset.scene[sequence]
        self.tokens = {'LIDAR_TOP':[],"CAM_FRONT":[],"CAM_FRONT_LEFT":[],"CAM_FRONT_RIGHT":[]}
        self.close_pts_range = 2.0
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
            self.tokens['CAM_FRONT'].append(sample['data']['CAM_FRONT'])
            self.tokens['CAM_FRONT_LEFT'].append(sample['data']['CAM_FRONT_LEFT'])
            self.tokens['CAM_FRONT_RIGHT'].append(sample['data']['CAM_FRONT_RIGHT'])
            
            #velodyne bin file
            
    def apply_transform(self,points, pose):
        '''
        used in point cloud aggregation function 
        '''
        
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
    
    def remove_close_points(self,points, radius: float) -> None:
        """
        Stolen from nuscenes-devkit.
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """

        x_filt = np.abs(points[:,0]) < radius
        y_filt = np.abs(points[:,1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        points = points[not_close, :]
        return points, not_close

    def get_point_cloud(self, index,intensity=True,pose_correction=False):
        lidar_token = self.tokens['LIDAR_TOP'][index]
        lidar_data = self.dataset.get('sample_data', lidar_token)
        scan = np.fromfile(os.path.join(self.dataset_path, lidar_data["filename"]), dtype=np.float32)
        #Save scan
        points = scan.reshape((-1, 5))[:, :4]
        
        points, _ = self.remove_close_points(points, self.close_pts_range)
        
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
        ##requires getting point cloud as well to find the indeces to remove from range filter
        lidar_token = self.tokens['LIDAR_TOP'][index]
        lidar_data = self.dataset.get('sample_data', lidar_token)
        scan = np.fromfile(os.path.join(self.dataset_path, lidar_data["filename"]), dtype=np.float32).reshape((-1, 5))
        _,idcs = self.remove_close_points(scan[:,:3],self.close_pts_range)
        
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
        return panoptic_labels[idcs].reshape(-1,)
    
    def get_available_cameras(self):
        return ['CAM_FRONT', 'CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

    def get_image(self, index):
        cam_token_front = self.tokens['CAM_FRONT'][index]
        cam_token_front_left = self.tokens['CAM_FRONT_LEFT'][index]
        cam_token_front_right = self.tokens['CAM_FRONT_RIGHT'][index]
        
        image_data_front = self.dataset.get('sample_data', cam_token_front)
        image_data_front_left = self.dataset.get('sample_data', cam_token_front_left)
        image_data_front_right = self.dataset.get('sample_data', cam_token_front_right)
        
        image_front = Image.open(os.path.join(self.dataset_path, image_data_front["filename"])) # open jpg.
        image_front_left = Image.open(os.path.join(self.dataset_path, image_data_front_left["filename"])) # open jpg.
        image_front_right = Image.open(os.path.join(self.dataset_path, image_data_front_right["filename"])) # open jpg.
        return {'CAM_FRONT':image_front,'CAM_FRONT_LEFT':image_front_left,'CAM_FRONT_RIGHT':image_front_right}

    def get_image_instances(self, index):
        raise NotImplementedError

    def get_calibration_matrices(self, index):
        cam_token_front = self.tokens['CAM_FRONT'][index]
        cam_token_front_left = self.tokens['CAM_FRONT_LEFT'][index]
        cam_token_front_right = self.tokens['CAM_FRONT_RIGHT'][index]
        
        image_data_front = self.dataset.get('sample_data', cam_token_front)
        image_data_front_left = self.dataset.get('sample_data', cam_token_front_left)
        image_data_front_right = self.dataset.get('sample_data', cam_token_front_right)
        
        calib_data_front = self.dataset.get("calibrated_sensor", image_data_front["calibrated_sensor_token"])
        calib_data_front_left = self.dataset.get("calibrated_sensor", image_data_front_left["calibrated_sensor_token"])
        calib_data_front_right = self.dataset.get("calibrated_sensor", image_data_front_right["calibrated_sensor_token"])
        return {'CAM_FRONT':calib_data_front,'CAM_FRONT_LEFT':calib_data_front_left,'CAM_FRONT_RIGHT':calib_data_front_right}
