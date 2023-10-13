# TARL Docker Image
This folder contains a docker image to extract [TARL](https://github.com/PRBonn/TARL) features for KITTI and NuScenes
## Building Docker Image
1) Build the docker image using the provided `Dockerfile`:
```
docker build -t tarl -f Dockerfile ..
```

Building the docker container will take some time due to the compilation process of MinkowskiEngine. 
Optionally, you can add your user to the docker group as described [here](https://docs.docker.com/engine/install/linux-postinstall/) so that running docker does not require root rights.
## Running Docker Container
To run the container use the following command:
```
docker run --gpus all \
    -v <INPUT_PATH>:/input \
    -v <OUTPUT_PATH>:/output \
    tarl [OPTIONAL_ARGS]
```
Here `<INPUT_PATH>` is the path where the lidar point cloud files are stored. `<OUTPUT_PATH>` is the path where the tarl features will be saved.

The following `[OPTIONAL_ARGS]` can be used:
```
optional arguments:
  -h, --help            show this help message and exit
  -d, --dataset         dataset format, supports kitti and nuscenes (default: kitti)    
```
For an example of running the Docker container, refer to the provided `docker_run.sh` script.

## Output Structure
The output features are stored within compressed `.bin` file, which can be loaded using NumPy and zlib with:
```python
import zlib
import numpy as np
tarl_dim = 96
#compressed_file : path to tarl .bin file
compressed_file = '00000.bin'
with open(compressed_file, 'rb') as f_in:
    compressed_data = f_in.read()
decompressed_data = zlib.decompress(compressed_data)
loaded_array = np.frombuffer(decompressed_data, dtype=np.float32)
point_features = loaded_array.reshape(-1,tarl_dim) #features are stored per point 
```
