# Instructions for running voxel clustering 
Dependencies are located inside segmentation/voxel_clustering_dependencies/. Python bindings need to be built in order to run the code. 

## Dependendencies 

- [pybind11](https://pybind11.readthedocs.io/en/stable/installing.html)
- [cmake](https://cmake.org/install/) (3.12 or higher)
- Eigen3 library : 
	```
	sudo apt update -y
	sudo apt install libeigen3-dev
	```


## Build the python bindings 

```sh 
cd segmentation/voxel_clustering_dependencies
mkdir build && cd build 
cmake ..
make -j{$nproc}
```

After successfull compilation, the python bindings are located inside ```segmentation/voxel_clustering_dependencies/build/``` 
In order to import the libraries ```lib_path``` in ```segmentation/voxel_clustering_segmentation.py``` needs to be set to this build directory. 

## Running nuScenes Comparison 
Please compile the dependencies by following the steps above first in order ot run the code. 

Currently, the evaluation only uses a scene of the smaller nuScenes 'mini' dataset. 
The preprocessed dataset is provided [here](https://drive.google.com/file/d/1n4jOrY4qFXwFza7kYoJb2OeOr4CWuXhT/view?usp=sharing) for download. 

