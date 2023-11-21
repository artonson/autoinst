# Hidden Points Removal #

This folder contains mesh based HPR method from [link](https://github.com/prime-slam/hidden-points-removal/). Dockerfile is provided to build docker-image as well.
## Building Docker Image ##
Run this in terminal:
```
docker build -t makeitdense -f Dockerfile .
```
Optionally, you can add your user to the docker group as described [here](https://docs.docker.com/engine/install/linux-postinstall/) so that running docker does not require root rights.
## Running Docker Container ##
To run the container use the following command:
```
docker run \
    -v <POINT_CLOUD_PATH>:/workspace/dataset \
    -v <OUTPUT_PATH>:/workspace/output \
    makeitdense [OPTIONAL_ARGS]
```
Here `<POINT_CLOUD_PATH>` is the path where your point cloud dataset is stored on the host machine. `<OUTPUT_PATH>` is the path where the output point cloud will be saved. Additionally, the script will default to use the `center_points.npy` saved in `<POINT_CLOUD_PATH>` to perform HPR. If you want to use a different set of center points, you can specify the path to the center points file using the `--center_point_file` argument in `[OPTIONAL_ARGS]`, or you can specify the `--center_point` argument to use a single center point.

The following `[OPTIONAL_ARGS]` can be used:
```
optional arguments:
  -h, --help            show this help message and exit
  --center_point_file STRING
                        path to center points file (default: center_points.npy). Set to False to use single center point.
  --center_point STRING
                        center point in form "[x,y,z]" (default: "[0,0,0]"). Will be overwritten if --center_point_file is specified.
  --threshold FLOAT     threshold for HPR, Unit meter (default: 0.1)
  --point_cloud_output  save output point cloud (default: False)
  --mesh_output         save output mesh (default: False)
```
## Output Structure ##

The output masks are stored with compressed `.npz` file, which can be loaded using NumPy with:
```python
import numpy as np
masks = np.load(npz_filepath, allow_pickle=True)
```
Each mask is represented as a dictionary containing the following information:
```python
{
    "center_points"          : [numpy.ndarray],         # The list of center points of the masks, each point is represented as a numpy array of shape (3,)
    "pt_masks"               : [numpy.ndarray],         # The list of masks, each mask is represented as a numpy array of shape (1,N)
}
```
The file name of the output masks are in the form of `{pcd_name}_masks_{threshold}.npz`, if the point cloud and the meshes are saved, their file names are in the form of `{pcd_name}_{num}_point_cloud_{threshold}.ply`, `{pcd_name}_mesh.ply`, respectively.