### Copied from https://github.com/artonson/dense_lidar_recon ###

from abc import ABC, abstractmethod

from dataset.dataset import DatasetOrNone
from dataset.dataset_entry import DatasetEntry


class Filter(ABC):
    @abstractmethod
    def __call__(
        self, data_entry: DatasetEntry, dataset: DatasetOrNone = None
    ) -> DatasetEntry:
        pass
