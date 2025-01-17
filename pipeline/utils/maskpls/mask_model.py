import MinkowskiEngine as ME
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.maskpls.decoder import MaskedTransformerDecoder
from utils.maskpls.mink import MinkEncoderDecoder
import os


class MaskPS(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        backbone = MinkEncoderDecoder(hparams.BACKBONE, hparams[hparams.MODEL.DATASET])
        self.backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(backbone)

        self.decoder = MaskedTransformerDecoder(
            hparams.DECODER,
            hparams.BACKBONE,
            hparams[hparams.MODEL.DATASET],
        )

        self.overlap_threshold = 0.8

    def forward(self, x):
        feats, coors, pad_masks, bb_logits = self.backbone(x)
        outputs, padding = self.decoder(feats, coors, pad_masks)

        sem_pred, ins_pred, max_confs = self.panoptic_inference2(outputs, padding)
        return sem_pred, ins_pred, max_confs

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
        num_classes = 1
        sem_pred = []
        ins_pred = []
        panoptic_output = []
        info = []
        all_confs = []
        for mask_cls, mask_pred, pad in zip(mask_cls, mask_pred, padding):
            scores, labels = mask_cls.max(-1)
            mask_pred = mask_pred[~pad].sigmoid()

            keep = labels.ne(num_classes)

            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[:, keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            # prob to belong to each of the `keep` masks for each point
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks

            probabilities = F.softmax(cur_prob_masks, dim=1)
            try:
                max_confs = probabilities.max(1)[0]
            except:
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

            if cur_masks.shape[1] == 0:  # no masks detected
                panoptic_output.append(panoptic_seg)
                info.append(segments_info)
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
            else:
                # mask index for each point: between 0 and (`keep` - 1)
                cur_mask_ids = cur_prob_masks.argmax(1)
                # top2_indices = cur_prob_masks.topk(2, dim=1).indices

                # The index of the second highest number is the second element in the top2_indices
                # second_highest_indices = top2_indices[:, 1]

                stuff_memory_list = {}
                stuff_memory_list2 = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()  # current class
                    isthing = True
                    mask_area = (cur_mask_ids == k).sum().item()  # points in mask k
                    original_area = (cur_masks[:, k] >= 0.5).sum().item()  # binary mas
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
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

        return sem_pred, ins_pred, all_confs

    def panoptic_inference2(self, outputs, padding):
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        num_classes = 1
        sem_pred = []
        ins_pred = []
        panoptic_output = []
        panoptic_output2 = []
        info = []
        info2 = []
        all_confs = []
        for mask_cls, mask_pred, pad in zip(mask_cls, mask_pred, padding):
            scores, labels = mask_cls.max(-1)
            mask_pred = mask_pred[~pad].sigmoid()

            keep = labels.ne(num_classes)

            # breakpoint()
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[:, keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            # prob to belong to each of the `keep` masks for each point
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks

            probabilities = F.softmax(cur_prob_masks, dim=1)
            try:
                max_confs = probabilities.max(1)[0]
            except:
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

            if cur_masks.shape[1] == 0:  # no masks detected
                panoptic_output.append(panoptic_seg)
                info.append(segments_info)
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
            else:
                cur_mask_ids = cur_prob_masks.argmax(1)

                stuff_memory_list = {}
                stuff_memory_list2 = {}
                cnt = 0
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()  # current class
                    isthing = True
                    mask_area = (cur_mask_ids == k).sum().item()  # points in mask k
                    original_area = (
                        (cur_masks[:, k] >= 0.001).sum().item()
                    )  # binary mas
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.001)
                    cur_ids = torch.where(cur_mask_ids == k)[0]
                    conf = probabilities[cur_ids].max(1)[0].mean()

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        # if mask_area / original_area < self.cfg.MODEL.OVERLAP_THRESHOLD:
                        #    continue  # binary mask occluded 80%
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
                                "isthing": True,
                                "sem_class": int(pred_class),
                                "conf": conf,
                                "mask_id": cnt,
                            }
                        )
                        cnt += 1

                panoptic_output.append(panoptic_seg)
                segments_info = sorted(
                    segments_info, key=lambda x: x["conf"], reverse=True
                )

                info.append(segments_info)
                for inf in segments_info:
                    mask = masks[inf["mask_id"]]
                    sem[mask] = inf["sem_class"]
                    if inf["isthing"]:
                        ins[mask] = inf["id"]
                    else:
                        ins[mask] = 0

                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())

        return sem_pred, ins_pred, all_confs
