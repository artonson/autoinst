### Copied from https://github.com/artonson/dense_lidar_recon ###

import os
from typing import List, Union

from nptyping import Float, NDArray, Shape
from PIL import Image

DatasetPathLike = Union[str, os.PathLike]

FloatXx3 = NDArray[Shape["*, 3"], Float]
Float4x4 = NDArray[Shape["4, 4"], Float]

PointCloudXx3 = FloatXx3
Transform4x4 = Float4x4
ImageEntry = Union[List[Image.Image], Image.Image, None]
