import os
from os.path import join

import click
import torch
import yaml
import numpy as np 
from easydict import EasyDict as edict
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.mask_model import MaskPS
from pytorch_lightning import Trainer
from mask_pls.datasets.pseudo_dataset import PseudoSemanticDatasetModule
from tqdm import tqdm 

def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


@click.command()
@click.option("--save_testset", is_flag=True)
@click.option("--nuscenes", is_flag=True)
def main(save_testset, nuscenes):
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml")))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml")))
    )
    decoder_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})

    cfg.EVALUATE = True
    if save_testset:
        results_dir = create_dirs(nuscenes)
        print(f"Saving test set predictions in directory {results_dir}")
        cfg.RESULTS_DIR = results_dir

    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"

    dataset = PseudoSemanticDatasetModule(cfg)
    dataset.setup()
    train_loader = dataset.train_dataloader()
    max_num_labels = 0 
    
    for batch in tqdm(train_loader):
     num_labels = batch['masks'][0].shape[0]
     max_num_labels = max(num_labels,max_num_labels)
     print("Max labels is",max_num_labels)
     
    
    
main()