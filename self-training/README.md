# Mask-Based Panoptic LiDAR Segmentation for Autonomous Driving

This folder contains the self-training implementation based on the [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/marcuzzi2023ral.pdf).

The code has been tested on RTX 4090 & RTX 3080 running [Cuda 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) with the latest [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) version. 

## Installation

* Install the maskpls package by running in the root directory of this repo:

```
pip3 install -U -e .
```

* Install the packages in [requirements.txt](requirements.txt).

## Data preparation (Chunk Extraction) 

### SemanticKITTI

TODO : 
- [ ] add chunk extraction dataset
- [ ] add install instructions 
- [ ] add training instructions 


## Citation

If you use parts of this code, please cite our, as well as the original paper : 

```bibtex
@article{perauer2024autoinst,
  title={AutoInst: Automatic Instance-Based Segmentation of LiDAR 3D Scans},
  author={Perauer, Cedric and Heidrich, Laurenz Adrian and Zhang, Haifan and Nie{\ss}ner, Matthias and Kornilova, Anastasiia and Artemov, Alexey},
  journal={arXiv preprint arXiv:2403.16318},
  year={2024}
}
```

```bibtex
@article{marcuzzi2023ral,
  author = {R. Marcuzzi and L. Nunes and L. Wiesmann and J. Behley and C. Stachniss},
  title = {{Mask-Based Panoptic LiDAR Segmentation for Autonomous Driving}},
  journal = ral,
  volume = {8},
  number = {2},
  pages = {1141--1148},
  year = 2023,
  doi = {10.1109/LRA.2023.3236568},
  url = {https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/marcuzzi2023ral.pdf},
}

```
## Licence
Copyright 2023, Rodrigo Marcuzzi, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.

This project is free software made available under the MIT License. For details see the LICENSE file
