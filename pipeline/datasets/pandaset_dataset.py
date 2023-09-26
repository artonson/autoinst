from datasets.abstract_dataset import AbstractDataset
from pandaset import DataSet
import numpy as np 
import os 

'''
Pandaset only includes semantic and not instance level labels !
At this point it can only do geometric inference (without aggregation)
'''
class PandasetDataset(AbstractDataset):
    
    def __init__(self,data_path,seq,sam_labels_path=None) :
        self.data_path = data_path
        self.dataset = DataSet(self.data_path)
        self.sam_labels_path = sam_labels_path
        self.seq = seq
        self.points_datapath = [] 
        self.annotations_path = self.data_path + '/' + self.seq  + '/annotations/semseg/'
        self.lidar_path = self.data_path + '/' + self.seq + '/lidar/'
        self.get_files()

    def get_files(self):
        for label in os.listdir(self.annotations_path):
                if os.path.exists(os.path.join(self.lidar_path, label)) and (self.annotations_path + label).endswith('.gz') :
                        self.points_datapath.append([self.seq,label.split('.')[0],"sequence_"+ self.seq + '_frame_' + label.split('.')[0]])
    
    def get_pose(self, index):
        pass
    
    def __len__(self):
        return len(self.points_datapath)

    def get_point_cloud(self, index,intensity=False):
        seq, idx, fn = self.points_datapath[index]
        idx = int(idx)
        sequence = self.dataset[seq]
        sequence.load_lidar().load_semseg()
        points_set = sequence.lidar[idx].to_numpy()[:,:4]
        if intensity == False : 
            return points_set[:,:3]
        else : 
            return points_set
        
    def get_labels(self,index):
        seq, idx, fn = self.points_datapath[index]
        idx = int(idx)
        sequence = self.dataset[seq]
        sequence.load_lidar().load_semseg()
        labels = sequence.semseg[idx]['class'].to_numpy()
        labels = labels.reshape(-1,)
        return labels 

    def get_available_cameras(self):
        pass

    def get_image(self, cam_name, index):
        pass

    def get_image_instances(self, cam_name, index):
        pass

    def get_calibration_matrices(self, cam_name):
        pass
