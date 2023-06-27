### Copied from https://github.com/artonson/dense_lidar_recon ###

import glob
import os
from typing import Union

import numpy as np

from dataset.config import SEMANTIC_KITTI_LABEL_INDICES
from dataset.dataset import DatasetOrNone
from dataset.dataset_entry import DatasetEntry
from dataset.filters.filter import Filter


class KittiGTMovingObjectFilter(Filter):
    LABEL_MASK = 2**16 - 1  # 0xffff

    def __init__(self, label_path: Union[os.PathLike, str]):
        """
        Kitti Ground Truth Moving Object Filter
        Must be the first filter, otherwise shapes might mismatch

        Args:
            label_path (Union[os.PathLike, str]): label directory
        """
        self.label_path = label_path
        self.labels = self.__parse_labels()

    def __parse_labels(self):
        if os.path.isdir(self.label_path):
            return sorted(glob.glob(os.path.join(self.label_path, "") + "*.label"))
        return None

    def __call__(
        self, data_entry: DatasetEntry, dataset: DatasetOrNone = None
    ) -> DatasetEntry:
        assert self.labels != None
        assert data_entry.index != None

        labels = (
            np.fromfile(self.labels[data_entry.index], dtype=np.uint32)
            & KittiGTMovingObjectFilter.LABEL_MASK
        )

        # only moving objects from this index
        moving_index = SEMANTIC_KITTI_LABEL_INDICES["moving"]
        filtered_points = data_entry.point_cloud[labels < moving_index]

        filtered_entry = DatasetEntry(
            data_entry.index, data_entry.pose, filtered_points, data_entry.images
        )

        return filtered_entry
