from abc import ABC, abstractmethod

class AbstractDataset(ABC):
    @abstractmethod
    def get_pose(self, index):
        pass

    @abstractmethod
    def get_point_cloud(self, index):
        pass

    @abstractmethod
    def get_available_cameras(self):
        pass

    @abstractmethod
    def get_image(self, cam_name, index):
        pass

    @abstractmethod
    def get_image_instances(self, cam_name, index):
        pass

    @abstractmethod
    def get_calibration_matrices(self, cam_name):
        pass
