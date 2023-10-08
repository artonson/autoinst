# SLIC Docker Image
This folder contains image segmentation using [slic](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic) and a docker image
## Building Docker Image
1) If the NGC deep learning framework container with PyTorch Release 23.06 is not available on your system, obtain it by running:
```
docker pull nvcr.io/nvidia/pytorch:23.06-py3
``` 
2) Build the docker image using the provided `Dockerfile`:
```
docker build -t slic -f Dockerfile ..
```
Optionally, you can add your user to the docker group as described [here](https://docs.docker.com/engine/install/linux-postinstall/) so that running docker does not require root rights.
## Running Docker Container
To run the container use the following command:
```
docker run --gpus all \
    -v <IMAGES_PATH>:/input \
    -v <OUTPUT_PATH>:/output \
    slic [OPTIONAL_ARGS]
```
Here `<IMAGES_PATH>` is the path where your image dataset is stored on the host machine. `<OUTPUT_PATH>` is the path where the masks will be saved.

The following `[OPTIONAL_ARGS]` can be used:
```
optional arguments:
  -h, --help            show this help message and exit
  -f STRING, --image_format STRING
                        image format (default: png)
  -m BOOL, --mslic BOOL
                        use mask slic if true.
                        (default: True)
  -n INT, ----n_segments INT  
                        numbers of estimated segments(default: 100)
```

For an example of running the Docker container, refer to the provided `docker_run.sh` script.
## Output Structure
The output is patch-wise features and stored with compressed `.npz` file, which can be loaded using NumPy with:
```python
import numpy as np
segmentations = np.load(npz_filepath, allow_pickle=True)["masks"]
```
Each mask is represented as a dictionary containing the following information:
```python
{
    "segmentation"          : numpy.ndarray,    # Pixel-wise mask.
    "bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
    "area"                  : int,              # The area in pixels of the mask
}
```