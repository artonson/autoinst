### Copied from https://github.com/artonson/dense_lidar_recon ###

import numpy as np

from dataset.dataset import DatasetOrNone
from dataset.dataset_entry import DatasetEntry
from dataset.filters.filter import Filter


class RangeFilter(Filter):
    def __init__(self, min_range: float, max_range: float, is_centered=True):
        self.min_range = min_range
        self.max_range = max_range
        self.is_centered = is_centered

    def __call__(
        self, data_entry: DatasetEntry, dataset: DatasetOrNone = None
    ) -> DatasetEntry:
        points = data_entry.point_cloud

        # move points to the origin if they are transformed
        norm = np.linalg.norm(
            points - np.tile(data_entry.pose[:3, 3].T, [len(points), 1])
            if not self.is_centered
            else points,
            axis=1,
        )
        norm = np.logical_and(norm <= self.max_range, norm >= self.min_range)
        points_filtered = points[norm]

        filtered_entry = DatasetEntry(
            data_entry.index, data_entry.pose, points_filtered, data_entry.images
        )

        return filtered_entry
