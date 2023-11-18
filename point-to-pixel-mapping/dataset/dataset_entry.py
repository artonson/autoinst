### Copied from https://github.com/artonson/dense_lidar_recon ###

from dataclasses import dataclass
from typing import Dict, Union

from dataset.types import ImageEntry, PointCloudXx3, Transform4x4, FloatXx1


@dataclass
class DatasetEntry:
    index: int
    pose: Transform4x4
    point_cloud: PointCloudXx3
    intensity: FloatXx1
    images: Dict[str, ImageEntry]


DatasetEntryOrNone = Union[DatasetEntry, None]
