import numpy as np
import open3d as o3d
import os
import torch

from make_it_dense.evaluation import run_completion_pipeline
from make_it_dense.models import CompletionNet
from make_it_dense.utils import MkdConfig
from pathlib import Path
from vdb_to_numpy import vdb_to_triangle_mesh


def read_point_cloud(filename):
    ext = os.path.splitext(filename)[-1]
    if ext == ".bin":
        scan = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3]
    else:
        scan = o3d.io.read_point_cloud(str(filename)).points
    return np.asarray(scan, dtype=np.float64)


def generate_mesh(
    pointcloud: Path,
    checkpoint: Path = "/workspace/make_it_dense/models/make_it_dense.ckpt",
    cuda: bool = False,
):
    config = MkdConfig.from_dict(
        torch.load(checkpoint, map_location=torch.device("cpu"))["hyper_parameters"]
    )
    model = CompletionNet.load_from_checkpoint(
        map_location=torch.device("cpu"), checkpoint_path=str(checkpoint), config=config
    )
    model.eval()

    # Run make-it-dense pipeline
    out_grid, in_grid = run_completion_pipeline(
        scan=read_point_cloud(pointcloud),
        voxel_size=min(model.voxel_sizes) / 100.0,
        voxel_trunc=model.voxel_trunc,
        model=model,
        cuda=cuda,
    )

    return vdb_to_triangle_mesh(out_grid)
