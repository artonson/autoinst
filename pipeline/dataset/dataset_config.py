### Copied from https://github.com/artonson/dense_lidar_recon ###

from dataclasses import dataclass
from typing import Optional
from dataset.filters.filter_list import FilterListOrNone
from dataset.types import DatasetPathLike


@dataclass
class DatasetConfig:
    dataset_path: DatasetPathLike
    sam_folder_name: str
    dinov2_folder_name: str
    filters: FilterListOrNone
    cache: bool
    dist_threshold: float
