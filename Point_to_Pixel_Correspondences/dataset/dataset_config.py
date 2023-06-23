### Copied from https://github.com/artonson/dense_lidar_recon ###

from dataclasses import dataclass

from dataset.filters.filter_list import FilterListOrNone
from dataset.types import DatasetPathLike


@dataclass
class DatasetConfig:
    dataset_path: DatasetPathLike
    filters: FilterListOrNone
    cache: bool
