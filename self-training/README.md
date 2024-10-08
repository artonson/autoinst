# Mask-Based Panoptic LiDAR Instance Segmentation Self-Training

This folder contains the self-training implementation based on the [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/marcuzzi2023ral.pdf).

The code has been tested on RTX 4090 & RTX 3080 running [Cuda 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) with the latest [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) version. 

## Installation

* Install the maskpls package by running in the root directory of this repo:

```
cd self-training/
pip install -U -e .
pip install -r requirements.txt
```

## Data preparation (Chunk Extraction) 

To create the data yourself, you can run NCuts in data generation mode (please refer to the main readme). We also provide our training dataset on [google drive](https://drive.google.com/file/d/1f4huUeliHthBdp2sVrWa9ULmrbJhcf1r/view?usp=sharing). 

Just download the dataset and configure the ``KITTI`` ``PATH`` variable in ``self-training/mask_pls/config/model.yaml`` to point to this directory. 

## Run Training

Train the model with : 

```
python scripts/train_model.py
```

Checkpoints will be saved automatically after every epoch. By default we do not implement the validation step as we train on all the pseudo labeled sequences for refinement.
We recommend training for 7-8 epochs. We provide the weights for 7 epochs of training [here](https://drive.google.com/file/d/1tdsVv10vfaWJSU4MNpVVkh3-BNiPXUlT/view?usp=drive_link). 

## Citation

If you find this project useful, please consider citing our paper, as well as the original MaskPLS paper : 

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
