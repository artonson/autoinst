import os
from os.path import join

import click
import torch
import yaml
from easydict import EasyDict as edict
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.datasets.pseudo_dataset import PseudoSemanticDatasetModule
from mask_pls.models.mask_model import MaskPS
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


@click.command()
@click.option("--w", type=str, default=None, required=False)
@click.option("--ckpt", type=str, default=None, required=False)
@click.option("--nuscenes", is_flag=True)
def main(w, ckpt, nuscenes):
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
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"

    data = PseudoSemanticDatasetModule(cfg)
    model = MaskPS(cfg)
    if w:
        w = torch.load(w, map_location="cpu")
        model.load_state_dict(w["state_dict"])

    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + cfg.EXPERIMENT.ID, default_hp_metric=False
    )

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    iou_ckpt = ModelCheckpoint(
        monitor="metrics/iou",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_iou{metrics/iou:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )
    pq_ckpt = ModelCheckpoint(
        monitor="metrics/pq",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_pq{metrics/pq:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )
    
    epoch_callback = ModelCheckpoint(
    save_top_k=-1,  # set to save all checkpoints
    every_n_epochs=50,
    filename=cfg.EXPERIMENT.ID + "_{epoch:02d}",
    )
    
    trainer = Trainer(
        accelerator="gpu",
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=[lr_monitor, pq_ckpt, iou_ckpt,epoch_callback],
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC
    )

    trainer.fit(model, data,ckpt_path=ckpt)


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


if __name__ == "__main__":
    main()


