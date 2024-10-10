### Copied from https://github.com/artonson/dense_lidar_recon ###

from abc import ABC, abstractmethod
from functools import wraps
from typing import Union

from dataset.dataset_entry import DatasetEntry
from dataset.types import ImageEntry, PointCloudXx3, Transform4x4


class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get_pose(self, index: int) -> Transform4x4:
        pass

    @abstractmethod
    def get_point_cloud(self, index: int) -> PointCloudXx3:
        pass

    @abstractmethod
    def get_image(self, camera_name: str, index: int) -> ImageEntry:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> DatasetEntry:
        pass


def cache_points(func):
    @wraps(func)
    def wrapper(self, index: int) -> PointCloudXx3:
        if not hasattr(self, "_points_cache"):
            setattr(self, "_points_cache", {})

        if self.config.cache and self._points_cache.get(index) is not None:
            return self._points_cache[index]

        points: PointCloudXx3 = func(self, index)

        if self.config.cache:
            self._points_cache[index] = points

        return points

    return wrapper


DatasetOrNone = Union[Dataset, None]
