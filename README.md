# AutoInst : Automatic Instance-Based Segmentation of LiDAR 3D Scans [IROS 2024]

[![Project Page](https://badgen.net/badge/color/Project%20Page/purple?icon=atom&label)](https://artonson.github.io/publications/2024-autoinst/)
[![arXiv](https://img.shields.io/badge/arXiv-2210.07233-b31b1b.svg)](https://arxiv.org/abs/2403.16318)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Screenshot from 2024-10-07 23-03-51](https://github.com/user-attachments/assets/37aff41d-f375-4d82-885d-78dea1d3fdd6)


## Requirements 

For running our NCuts extraction, install the requirements. We ran the NCuts pipeline with both Python 3.9 (on x86 AMD CPU and without the RAM intensive map creation on M1/M2 Macbook Air). 

### Python Libraries

```bash
cd autoinst/
sh setup.sh #creates conda env named autoinst
```

For running the refined MaskPLS model, please refer to the additional instructions in [self-training readme](https://github.com/artonson/autoinst/tree/cleanup/self-training)

### Patchwork++ 

Install the python bindings for Patchwork++ : 

```bash
git clone git@github.com:url-kaist/patchwork-plusplus.git
sudo apt-get install g++ build-essential libeigen3-dev python3-pip python3-dev cmake -y
conda activate autoinst
make pyinstall
```

For more details please see their [original rep](https://github.com/url-kaist/patchwork-plusplus)

## Dataset

We use the [SemanticKITTI dataset](https://www.semantic-kitti.org/). Including our extracted features, our dataset structure looks like this : 

<img src="https://github.com/user-attachments/assets/c17ab3c0-ce4e-4585-884e-88a2fbc42dc3" alt="structure" width="50%">

## Feature Pre-processing 

We provide the scripts for extracting the relevant image and point-based features in [autoinst/2D-VFMs
/dinov2](https://github.com/artonson/autoinst/tree/cleanup/2D-VFMs/dinov2) and [autoinst/Pointcloud-Models/tarl](https://github.com/artonson/autoinst/tree/cleanup/Pointcloud-Models/tarl#readme). 

### Pre-computed sample features

We also provide the complete data (including extracted aggregated maps) for the first map for testing out our code. 
Please download the dataset related files [here](https://drive.google.com/drive/folders/1G3mEC8WLI2rGbsm3nkCuyeRFL49bUgbk?usp=sharing) and unzip the subdirectories. 
Then set ``DATASET_PATH`` in ``config.py`` to this directory.



### Pre-computed sample map

Preprocessing the maps requires more memory (our machine used 128GB), while chunk-based GraphCuts can be run on a laptop. Therefore we also provide the aggregation data for the first map [here](https://drive.google.com/drive/folders/1JpSTnZ8vBXzhLJVBAIltullTaACLQ7oX?usp=drive_link).
You should then set ``OUT_FOLDER`` in ``config.py`` to this directory so the maps can be loaded correctly. 


## Running the Pipeline Code 

Make sure to set the dataset path in ```autoinst/pipeline/config.py``` accordingly. You can also configure the feature combinations/use of MaskPLS in config.py accordingly (by default TARL/Spatial is used). By default, the first map is run for which we provide the data (see links above) and the metrics are computed.  

```bash
cd autoinst/pipeline/
python run_pipeline.py 
```

#### Simple 3D Graph Cuts exmaple

If you are interested in using our GraphCuts implementation for your own project, we provide a simple implementation that only uses spatial distances [here](https://github.com/Cedric-Perauer/Ncuts_Example). 


### Using MaskPLS 
To use MaskPLS inference, simply set the [config] in ```autoinst/pipeline/config.py``` to ``config_maskpls_tarl_spatial``.
You can download one of the set of weights [here](). 
 
### Generating Training Chunks 
To generate training chunks, simply set the [flag]() in ```autoinst/pipeline/config.py``` to True. 
Metrics computation is skipped and the output is stored in the according . 

### Self-Training 

For self-training, please refer to the [corresponding self-training readme](https://github.com/artonson/autoinst/tree/cleanup/self-training).  This readme also contains the info for MaskPLS inference. 

### Expected Results for sample map 

| Method                  | AP    | P/R/F1            | S_assoc     |
|-------------------------|-------|-------------------|-------|
| NCuts Spatial           | 41.74% | 86.15%/75.67%/80.57% | 70.19% |
| NCuts TARL/Spatial      | 53.74% | 87.69%/77.02%/82.01% | 71.05% |
| NCuts TARL/Spatial/Dino | 34.33%     |   81.65%/60.13%/69.26%                | 60.00%      |
| MaskPLS Tarl/Spatial    | **65.93%**      | **91.53%**/**80.40%**/**85.61%**                  | **78.42%** |


## Acknowledgements 

Among others, our project was inspired from/uses code from [Unscene3D](https://github.com/RozDavid/UnScene3D), [MaskPLS](https://github.com/PRBonn/MaskPLS),[TARL](https://github.com/PRBonn/TARL), [Dinov2](https://github.com/facebookresearch/dinov2), [semantic kitti api](https://github.com/PRBonn/semantic-kitti-api), [Patchwork++](https://github.com/url-kaist/patchwork-plusplus) and we would like to thank the authors for their valuable work. 

If you use parts of our code or find our project useful, please consider citing our paper : 
```bibtex
@article{perauer2024autoinst,
  title={AutoInst: Automatic Instance-Based Segmentation of LiDAR 3D Scans},
  author={Perauer, Cedric and Heidrich, Laurenz Adrian and Zhang, Haifan and Nie{\ss}ner, Matthias and Kornilova, Anastasiia and Artemov, Alexey},
  journal={arXiv preprint arXiv:2403.16318},
  year={2024}
}
