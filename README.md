# AutoInst : Automatic Instance-Based Segmentation of LiDAR 3D Scans [IROS 2024]



![Screenshot from 2024-10-07 23-03-51](https://github.com/user-attachments/assets/37aff41d-f375-4d82-885d-78dea1d3fdd6)

## Requirements 
- ToDo

## Dataset

We use the [SemanticKITTI dataset](https://www.semantic-kitti.org/). Including our extracted features, our dataset structure looks like this : 

<img src="https://github.com/user-attachments/assets/c17ab3c0-ce4e-4585-884e-88a2fbc42dc3" alt="structure" width="50%">

## Feature Pre-processing 

We provide the scripts for extracting the relevant image and point-based features in [autoinst/2D-VFMs
/dinov2](https://github.com/artonson/autoinst/tree/cleanup/2D-VFMs/dinov2) and [autoinst/Pointcloud-Models/tarl](https://github.com/artonson/autoinst/tree/cleanup/Pointcloud-Models/tarl#readme). 

### Pre-computed sample features

We also provide the complete data (including extracted aggregated maps) for the first map for testing out our code. 
Please download the dataset related files [here](https://drive.google.com/drive/folders/1G3mEC8WLI2rGbsm3nkCuyeRFL49bUgbk?usp=sharing) and unzip the subdirectories. 

### Pre-computed sample map

Preprocessing the maps requires more memory (our machine used 128GB), while chunk-based GraphCuts can be run on a laptop. Therefore we also provide the aggregation data for the first map [here](https://drive.google.com/drive/folders/1JpSTnZ8vBXzhLJVBAIltullTaACLQ7oX?usp=drive_link). 

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

### Expected Results for provided sample map 

to do : add metrics table

### Self-Training 

For self-training, please refer to the [corresponding self-training readme](https://github.com/artonson/autoinst/tree/cleanup/self-training). 

### Expected Results for sample map 

| Method                  | AP    | P/R/F1            | S_assoc     |
|-------------------------|-------|-------------------|-------|
| NCuts Spatial           | 41.74 | 86.15/75.67/80.57 | 70.19 |
| NCuts TARL/Spatial      | 53.74 | 87.69/77.02/82.01 | 71.05 |
| NCuts TARL/Spatial/Dino | tbd     |   tbd                | tbd      |
| MaskPLS Tarl/Spatial    |       |                   |       |


## Acknowledgements 

Among others, our project was inspired from/uses code from [Unscene3D](https://github.com/RozDavid/UnScene3D), [MaskPLS](https://github.com/PRBonn/MaskPLS),[TARL](https://github.com/PRBonn/TARL), [Dinov2](https://github.com/facebookresearch/dinov2), [semantic kitti api](https://github.com/PRBonn/semantic-kitti-api) and we would like to thank the authors for their valuable work.  
