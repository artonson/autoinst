import mask_pls.utils.testing as testing
import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
from mask_pls.models.decoder import MaskedTransformerDecoder
from mask_pls.models.loss import MaskLoss, SemLoss
from mask_pls.models.mink import MinkEncoderDecoder
from mask_pls.utils.evaluate_panoptic import PanopticEvaluator
from pytorch_lightning.core.module import LightningModule
import wandb
import os 

class MaskPS(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(dict(hparams))
        self.cfg = hparams

        backbone = MinkEncoderDecoder(hparams.BACKBONE, hparams[hparams.MODEL.DATASET])
        self.backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(backbone)

        self.decoder = MaskedTransformerDecoder(
            hparams.DECODER,
            hparams.BACKBONE,
            hparams[hparams.MODEL.DATASET],
        )
        wandb.init(entity="cedric-perauer", project="maskpls_chunks")
        

        self.mask_loss = MaskLoss(hparams.LOSS, hparams[hparams.MODEL.DATASET])
        self.sem_loss = SemLoss(hparams.LOSS.SEM.WEIGHTS)

        self.evaluator = PanopticEvaluator(
            hparams[hparams.MODEL.DATASET], hparams.MODEL.DATASET
        )
        
        self.evaluator_train = PanopticEvaluator(
            hparams[hparams.MODEL.DATASET], hparams.MODEL.DATASET
        )

    def forward(self, x):
        feats, coors, pad_masks, bb_logits = self.backbone(x)
        #bb_logits = torch.zeros_like(bb_logits)
        #bb_logits[:,:,1] = 1000
        outputs, padding = self.decoder(feats, coors, pad_masks)
        
        return outputs, padding, bb_logits

    def getLoss(self, x, outputs, padding, bb_logits):
        ## create a mask here based on relevant indices 
        # mask out pt
        #breakpoint()
    
        
        targets = {"classes": x["masks_cls"], "masks": x["masks"]}

        loss_mask = self.mask_loss(outputs, targets, x["masks_ids"], torch.from_numpy(x["pt_coord"][0]).cuda())
        #print(loss_mask)
        #sem_labels = [

        return loss_mask

    def training_step(self, x: dict, idx):
        outputs, padding, bb_logits = self.forward(x)
        loss_dict = self.getLoss(x, outputs, padding, bb_logits)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        total_loss = sum(loss_dict.values())
        #print('loss',loss_dict)
        self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        wandb.log({"epoch": self.trainer.current_epoch, "main/train_loss": total_loss})
        for key, value in loss_dict.items(): 
            wandb.log({"epoch": self.trainer.current_epoch, 
                        'detailed_losses_train/' + key: value})
        
        #sem_pred, ins_pred, max_confs = self.panoptic_inference(outputs, padding)
        #self.evaluator_train.update(sem_pred, ins_pred, x)
        torch.cuda.empty_cache()
        '''
        with torch.no_grad(): 
            self.cfg.RESULTS_DIR = 'outputs/test/'
            if "RESULTS_DIR" in self.cfg:
                results_dir = self.cfg.RESULTS_DIR
                class_inv_lut = self.evaluator.get_class_inv_lut()
                dt = self.cfg.MODEL.DATASET
                testing.save_results(
                    sem_pred, ins_pred,max_confs, results_dir, x, class_inv_lut, x["token"], dt
                )
        '''
            
            
        return total_loss
    
    '''
    def on_train_epoch_end(self):
        bs = self.cfg.TRAIN.BATCH_SIZE
        wandb.log({"epoch": self.trainer.current_epoch, "train_metrics/pq": self.evaluator_train.get_mean_pq()})
        wandb.log({"epoch": self.trainer.current_epoch, "train_metrics/iou": self.evaluator_train.get_mean_iou()})
        wandb.log({"epoch": self.trainer.current_epoch, "train_metrics/rq": self.evaluator_train.get_mean_rq()})

        if not "EVALUATE" in self.cfg:
            self.evaluator_train.reset()
    '''

    def validation_step(self, x: dict, idx):
        '''
        self.backbone.train()
        
        if "EVALUATE" in self.cfg : 
            self.evaluation_step(x, idx)
            return
        try : 
            outputs, padding, bb_logits = self.forward(x)
            loss_dict = self.getLoss(x, outputs, padding, bb_logits)
            for k, v in loss_dict.items():
                self.log(f"val/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            total_loss = sum(loss_dict.values())
            self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
            for key, value in loss_dict.items(): 
                    wandb.log({"epoch": self.trainer.current_epoch, 
                                'detailed_losses_val/' + key: value})
            wandb.log({"epoch": self.trainer.current_epoch, "main/val_loss": total_loss})
        except : 
            total_loss = 3
            pass
        sem_pred, ins_pred,_,_,max_confs = self.panoptic_inference(outputs, padding)
        self.evaluator.update(sem_pred, ins_pred, x)
        
        if self.trainer.current_epoch % 50 == 0 : 
                results_dir = 'output/train/epoch_' + str(self.trainer.current_epoch) + '/'
                if os.path.exists(results_dir) == False : 
                    os.makedirs(results_dir)
                #breakpoint()
                class_inv_lut = self.evaluator.get_class_inv_lut()
                dt = self.cfg.MODEL.DATASET
                #testing.save_results(
                #    sem_pred, ins_pred,max_confs, results_dir, x, class_inv_lut, x["token"], dt
                #)
        
        
        torch.cuda.empty_cache()
        return total_loss
        '''
        return 0 

    def on_validation_epoch_end(self):
        '''
        self.backbone.train()
        bs = self.cfg.TRAIN.BATCH_SIZE
        wandb.log({"epoch": self.trainer.current_epoch, "val_metrics/pq": self.evaluator.get_mean_pq()})
        wandb.log({"epoch": self.trainer.current_epoch, "val_metrics/iou": self.evaluator.get_mean_iou()})
        wandb.log({"epoch": self.trainer.current_epoch, "val_metrics/rq": self.evaluator.get_mean_rq()})
        
        self.log("metrics/pq", self.evaluator.get_mean_pq(), batch_size=bs)
        self.log("metrics/iou", self.evaluator.get_mean_iou(), batch_size=bs)
        self.log("metrics/rq", self.evaluator.get_mean_rq(), batch_size=bs)
        if not "EVALUATE" in self.cfg:
            self.evaluator.reset()
        '''
        pass

    def evaluation_step(self, x: dict, idx):
        outputs, padding, bb_logits = self.forward(x)
        sem_pred, ins_pred,_,_, max_confs = self.panoptic_inference(outputs, padding)
        self.evaluator.update(sem_pred, ins_pred, x)

    def test_step(self, x: dict, idx):
        #self.backbone.train()
        #self.decoder.train()
    
        outputs, padding, bb_logits = self.forward(x)
        sem_pred, ins_pred,sem_pred2,ins_pred2, max_confs = self.panoptic_inference(outputs, padding)

        if "RESULTS_DIR" in self.cfg:
            results_dir = self.cfg.RESULTS_DIR
            class_inv_lut = self.evaluator.get_class_inv_lut()
            dt = self.cfg.MODEL.DATASET
            testing.save_results(
                sem_pred, ins_pred,max_confs, results_dir, x, class_inv_lut, x["token"], dt
            )
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.LR)
        #scheduler = torch.optim.lr_scheduler.StepLR(
        #    optimizer, step_size=self.cfg.TRAIN.STEP, gamma=self.cfg.TRAIN.DECAY
        #)
        return [optimizer]

    def semantic_inference(self, outputs, padding):
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        semseg = []
        for mask_cls, mask_pred, pad in zip(mask_cls, mask_pred, padding):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred[~pad].sigmoid()  # throw padding points
            pred = torch.einsum("qc,pq->pc", mask_cls, mask_pred)
            semseg.append(torch.argmax(pred, dim=1))
        return semseg

    def panoptic_inference(self, outputs, padding):
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        things_ids = self.trainer.datamodule.things_ids
        num_classes = self.cfg[self.cfg.MODEL.DATASET].NUM_CLASSES
        sem_pred = []
        ins_pred = []
        ins_pred2 = [] ##second most confident mask prediction 
        sem_pred2 = []
        panoptic_output = []
        panoptic_output2 = []
        info = []
        info2 = []
        all_confs = []
        for mask_cls, mask_pred, pad in zip(mask_cls, mask_pred, padding):
            scores, labels = mask_cls.max(-1)
            mask_pred = mask_pred[~pad].sigmoid()
            
            keep = labels.ne(num_classes) 
            
            
            #breakpoint()
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[:, keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            # prob to belong to each of the `keep` masks for each point
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            
            
            probabilities = F.softmax(cur_prob_masks, dim=1)
            try : 
                max_confs = probabilities.max(1)[0]
            except : 
                max_confs = torch.zeros_like(cur_prob_masks)
            all_confs.append(max_confs)
            

            panoptic_seg = torch.zeros(
                (cur_masks.shape[0]), dtype=torch.int32, device=cur_masks.device
            )
            sem = torch.zeros_like(panoptic_seg)
            ins = torch.zeros_like(panoptic_seg)
            segments_info = []
            masks = []
            segment_id = 0
            
            segment_id2 = 0 
            masks2 = []
            segments_info2 = []
            panoptic_seg2 = torch.zeros(
                (cur_masks.shape[0]), dtype=torch.int32, device=cur_masks.device
            )
            sem2 = torch.zeros_like(panoptic_seg)
            ins2 = torch.zeros_like(panoptic_seg)
            
            if cur_masks.shape[1] == 0:  # no masks detected
                panoptic_output.append(panoptic_seg)
                info.append(segments_info)
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
            else:
                # mask index for each point: between 0 and (`keep` - 1)
                cur_mask_ids = cur_prob_masks.argmax(1)
                #top2_indices = cur_prob_masks.topk(2, dim=1).indices

                # The index of the second highest number is the second element in the top2_indices
                #second_highest_indices = top2_indices[:, 1]
                
                
                stuff_memory_list = {}
                stuff_memory_list2 = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()  # current class
                    isthing = pred_class in things_ids
                    mask_area = (cur_mask_ids == k).sum().item()  # points in mask k
                    original_area = (cur_masks[:, k] >= 0.5).sum().item()  # binary mas
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    
                    
                    #mask_area2 = (second_highest_indices == k).sum().item()  # points in mask k
                    #original_area2 = (cur_masks[:, k] >= 0.5).sum().item()  # binary mask 
                    #mask2 = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.cfg.MODEL.OVERLAP_THRESHOLD:
                            continue  # binary mask occluded 80%
                        if not isthing:  # merge stuff regions
                            if int(pred_class) in stuff_memory_list.keys():
                                # in the list, asign id stored on the list for that class
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                # not in the list, class = cur_id + 1
                                stuff_memory_list[int(pred_class)] = segment_id + 1
                        segment_id += 1
                        panoptic_seg[mask] = segment_id
                        masks.append(mask)
                        # indice which class each segment id has
                        segments_info.append(
                            {
                                "id": segment_id,
                                "isthing": bool(isthing),
                                "sem_class": int(pred_class),
                            }
                        )
                    
                    '''
                    ##second highest pred 
                    if mask_area2 > 0 and original_area2 > 0 and mask2.sum().item() > 0:
                        if mask_area2 / original_area2 < self.cfg.MODEL.OVERLAP_THRESHOLD:
                            continue  # binary mask occluded 80%
                        if not isthing:  # merge stuff regions
                            if int(pred_class) in stuff_memory_list2.keys():
                                # in the list, asign id stored on the list for that class
                                panoptic_seg2[mask] = stuff_memory_list2[int(pred_class)]
                                continue
                            else:
                                # not in the list, class = cur_id + 1
                                stuff_memory_list2[int(pred_class)] = segment_id + 1
                        segment_id2 += 1
                        panoptic_seg2[mask] = segment_id
                        masks2.append(mask)
                        # indice which class each segment id has
                        segments_info2.append(
                            {
                                "id": segment_id2,
                                "isthing": bool(isthing),
                                "sem_class": int(pred_class),
                            }
                        )
                        '''
                    
                panoptic_output.append(panoptic_seg)
                info.append(segments_info)
                for mask, inf in zip(masks, segments_info):
                    sem[mask] = inf["sem_class"]
                    if inf["isthing"]:
                        ins[mask] = inf["id"]
                    else:
                        ins[mask] = 0
                
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
                
                ##for alternative method 
                '''
                panoptic_output2.append(panoptic_seg2)
                info2.append(segments_info)
                for mask, inf in zip(masks2, segments_info2):
                    sem2[mask] = inf["sem_class"]
                    if inf["isthing"]:
                        ins2[mask] = inf["id"]
                    else:
                        ins2[mask] = 0
                
                sem_pred2.append(sem2.cpu().numpy())
                ins_pred2.append(ins2.cpu().numpy())
                '''
                
        return sem_pred, ins_pred,None,None, all_confs
