import numpy as np
from metrics.modified_LSTQ import evaluator
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
from config import *
import json

COLORS_arr = plt.cm.viridis(np.linspace(0, 1, 30))
COLORS = list(list(col) for col in COLORS_arr)
COLORS = [list(col[:3]) for col in COLORS]

from tqdm import tqdm

class Metrics:

    def __init__(self, name="NCuts", min_points=200, thresh=0.5):
        self.thresh = thresh  # iou threshold being used
        self.tps = 0
        self.unique_gts = 0
        self.preds_total = 0
        self.name = name
        self.min_points = min_points
        self.background_label = 0
        self.mode = "normal"
        print("start")
        self.pred_indices = {}
        self.gt_indices = {}
        self.calc_ap = True
        self.eval_lstq = evaluator(min_points=min_points)
        self.eval_lstq.reset()
        self.semantic_intersection = 0.05
        self.num_processes = METRICS_THREADS
        self.cols = COLORS
        self.sequence_metrics = {'ap0.5':[],'ap0.25':[],'ap':[],'p':[],'r':[],'f1':[],'S_assoc':[]}

        # ap stuff
        self.ap = {}
        self.ar = {}
        self.overlaps = [0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.ap_overlaps = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        print("Metrics for file", name)
        self.precision_all = {}
        self.recall_all = {}
        self.all_matches = {}
        self.gt_size = {}
        self.all_pred_size = {}
        self.all_gt_size = {}
        self.all_tp = {}

        for overlap in self.overlaps:
            self.precision_all[overlap] = [1.0]
            self.recall_all[overlap] = [0.0]
            self.all_matches[overlap] = []
            self.gt_size[overlap] = []
            self.all_pred_size[overlap] = 0
            self.all_gt_size[overlap] = 0

            self.all_tp[overlap] = 0

    def get_tp_fp(
        self,
        pred_labels,
        gt_labels,
        iou_thresh=0.5,
        confs=[],
    ):
        """
        :param pred_labels: labels of points corresponding to segmented planes
        :param gt_labels: labels of points corresponding to ground truth planes
        :param tp_condition: helper function to calculate statistics
        :return: true positive received using pred_labels and gt_labels
        """
        true_positive = 0
        gt_used = set()
        false_positives = 0

        for pred_label in tqdm(np.unique(pred_labels)):
            if pred_label == 0:
                continue
            matched = False
            pred_indices = self.pred_indices[pred_label]
            cur_iou = 0.0
            for gt in np.unique(gt_labels):
                if gt == 0:
                    continue

                gt_indices = self.gt_indices[gt]

                iou = self.iou(pred_indices, gt_indices)

                if iou >= iou_thresh and (gt not in gt_used):
                    matched = True
                    true_positive += 1
                    cur_iou = iou
                    gt_used.add(gt)
                    break

            if matched:
                if confs == []:
                    self.all_matches[iou_thresh].append(
                        {"result": "tp", "iou": cur_iou}
                    )
                else:
                    self.all_matches[iou_thresh].append(
                        {"result": "tp", "iou": cur_iou, "conf": confs[pred_label]}
                    )
            else:
                false_positives += 1
                if confs == []:
                    self.all_matches[iou_thresh].append({"result": "fp"})
                else:
                    self.all_matches[iou_thresh].append(
                        {"result": "fp", "iou": cur_iou, "conf": confs[pred_label]}
                    )

        return true_positive, false_positives

    def worker_function(self, data):
        pred, ins_labels, confs, iou_thresh = data
       
        return self.average_precision(pred, ins_labels, confs, iou_thresh)

    def average(self,list):
        return sum(list)/float(len(list))

    def average_precision_parallel(self, pred, ins_labels, confs, iou_thresh=0.5):
        data_for_processes = [(pred, ins_labels, confs, iou) for iou in self.overlaps]

        # Create a pool of worker processes
        with Pool(processes=self.num_processes) as pool:
            results = pool.map(self.worker_function, data_for_processes)

        for iou, ap in zip(self.overlaps, results):
            self.ap[iou] = ap

    def update_stats(
        self,
        all_labels,
        pred_labels,
        gt_labels,
        confs=[],
        calc_all=True,
        calc_lstq=True,
    ):
        print("update stats")
        pred_labels = self.filter_labels(pred_labels)
        all_labels = self.filter_labels(all_labels)

        for pred_label in tqdm(np.unique(pred_labels)):
            if pred_label == 0:
                continue
            pred_indices = np.where(pred_labels == pred_label)[0]
            self.pred_indices[pred_label] = pred_indices

        for gt in np.unique(gt_labels):
            if gt == 0:
                continue
            gt_indices = np.where(gt_labels == gt)[0]
            self.gt_indices[gt] = gt_indices
        print("got all the idcs")
        if calc_all:
            out = self.calculate_full_stats(pred_labels, gt_labels)
        if calc_lstq:
            self.eval_lstq.add_batch(all_labels, gt_labels)
            lstq = self.eval_lstq.get_eval()
        if self.calc_ap:
            self.average_precision_parallel(pred_labels, gt_labels, confs)
            aps_list = [self.ap[o] for o in self.ap_overlaps]
            ap = sum(aps_list) / float(len(aps_list))
            self.sequence_metrics['p'].append(out['precision'])
            self.sequence_metrics['r'].append(out['recall'])
            self.sequence_metrics['f1'].append(out['fScore'])
            self.sequence_metrics['ap0.25'].append(self.ap[0.25])
            self.sequence_metrics['ap0.5'].append(self.ap[0.5])
            self.sequence_metrics['ap'].append(ap)
            self.sequence_metrics['S_assoc'].append(lstq)

        return out, {"0.25": self.ap[0.25], "0.5": self.ap[0.5], "ap": ap, "lstq": lstq}

    def average_precision(self, pred, ins_labels, confs, iou_thresh=0.5):
        self.precision = [1.0]
        self.recall = [0.0]  ##Init with values to ensure correct computation
        unique_gt_labels = list(np.unique(ins_labels))
        unique_pred_labels = list(np.unique(pred))

        # background remove
        unique_pred_labels.remove(0)
        unique_gt_labels.remove(0)

        instance_conf = {}
        for instance_id, i in enumerate(unique_pred_labels):
            if confs == []:
                # taken from unscene3d eval
                instance_conf[i] = 0.5
            else:
                instance_conf[i] = confs[i]

        if confs != []:
            instance_conf = dict(
                sorted(instance_conf.items(), key=lambda item: item[1], reverse=True)
            )
        self.tp = 0
        self.fp = 0
        self.fn = len(unique_gt_labels)

        gt_used = []

        for prediction in instance_conf.keys():
            matched = False
            pred_indices = self.pred_indices[prediction]
            for gt in unique_gt_labels:

                gt_indices = self.gt_indices[gt]

                iou = self.iou(pred_indices, gt_indices)

                if iou >= iou_thresh and (gt not in gt_used):
                    matched = True
                    gt_used.append(gt)
                    break

            if matched:
                self.tp += 1
                self.fn -= 1

            else:
                self.fp += 1

            # Calculate precision and recall
            self.precision.append(self.tp / float(self.tp + self.fp))
            self.recall.append(self.tp / float(self.tp + self.fn))

        ap = np.trapz(self.precision, self.recall)
        return ap

    def average_precision_final(self, iou_thresh=0.5):

        tp = 0
        fp = 0
        fn = self.all_gt_size[iou_thresh]

        for matched in self.all_matches[iou_thresh]:
            if "conf" in list(self.all_matches[0.25][0].keys()):
                sorted(
                    self.all_matches[iou_thresh], key=lambda x: x["conf"], reverse=True
                )

            result = matched["result"]
            if result == "tp":
                tp += 1
                fn -= 1
            else:
                fp += 1

            # Calculate precision and recall
            self.precision_all[iou_thresh].append(tp / float(tp + fp))
            self.recall_all[iou_thresh].append(tp / float(tp + fn))

    def sequence_stats(self,out_dir='results/'):
        
        sequence_results = {'p':self.average(self.sequence_metrics['p']),
                            'r':self.average(self.sequence_metrics['r']),
                            'f1':self.average(self.sequence_metrics['f1']),
                            'ap':self.average(self.sequence_metrics['ap']),
                            'ap0.25':self.average(self.sequence_metrics['ap0.25']),
                            'ap0.5':self.average(self.sequence_metrics['ap0.5']),
                            'S_assoc':self.average(self.sequence_metrics['S_assoc']),
                            }
        
        print("Precison @ " + str(self.thresh), sequence_results['p'])
        print("Recall @ " + str(self.thresh), sequence_results['r'])
        print("F Score @ " + str(self.thresh), sequence_results['f1'])
        print("S_assoc score", sequence_results['S_assoc'])
        print("AP @ 0.25", sequence_results['ap0.25'])
        print("AP @ 0.5", sequence_results['ap0.5'])
        print("AP @ [0.5:0.95]",sequence_results['ap'])
        
        if os.path.exists(out_dir) == False : 
            os.makedirs(out_dir)
        
        with open(f'{out_dir}/{self.name}', 'w') as file:
            json.dump(sequence_results, file)
        
        
        

    def mean(self):
        mean_array = []
        for instance in self.all_matches[self.thresh]:
            if instance["result"] == "tp":
                mean_array.append(instance["iou"])

        return np.array(mean_array).mean() if len(mean_array) != 0 else 0.0

    def iou(self, pred_indices, gt_indices):
        intersection = np.intersect1d(pred_indices, gt_indices)
        union = np.union1d(pred_indices, gt_indices)

        return intersection.size / union.size

    def filter_labels(self, label):
        for clid in np.unique(label):
            cur_idcs = np.where(label == clid)
            cur_idcs = cur_idcs[0]

            if cur_idcs.shape[0] < self.min_points:
                label[cur_idcs] = self.background_label
        return label

    def intersect(self, pred_indices, gt_indices):
        intersection = np.intersect1d(pred_indices, gt_indices)
        return intersection.size / pred_indices.shape[0]

    def calculate_full_stats(self, pred_labels, gt_labels):
        print("full stats")
        self.thresh = 0.5
        for overlap in tqdm([0.5]):
            tps, fps = self.get_tp_fp(pred_labels, gt_labels, iou_thresh=overlap)

            if 0 in gt_labels:
                self.all_gt_size[overlap] += np.unique(gt_labels).shape[0] - 1
            self.all_pred_size[overlap] += np.unique(pred_labels).shape[0] - 1
            self.all_tp[overlap] += tps

        prec = self.all_tp[self.thresh] / self.all_pred_size[self.thresh]
        rec = self.all_tp[self.thresh] / self.all_gt_size[self.thresh]
        try:
            f1 = 2 * (prec * rec) / (prec + rec)
        except:
            f1 = 0
        mean = self.mean()
        panoptic = mean * f1
        out = {}
        out["fScore"] = f1
        out["precision"] = prec
        out["recall"] = rec
        out["panoptic"] = panoptic

        return out
