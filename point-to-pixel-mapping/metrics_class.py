import numpy as np
import instanseg 
from instanseg.metrics import full_statistics


instanseg.metrics.constants.UNSEGMENTED_LABEL = 0
instanseg.metrics.constants.IOU_THRESHOLD_FULL = 0.5

def get_average(l):
        return sum(l)/len(l)

class Metrics:

    def __init__(self, name='NCuts'):
        self.thresh = 0.5  # iou threshold being used
        self.tps = 0
        self.unique_gts = 0
        self.preds_total = 0
        self.name = name
        self.min_points = 200
        self.background_label = 0
        self.mode = 'normal'
        self.calc_ap = True

        # ap stuff
        self.hard_false_negatives = {}
        self.y_true_cluster = {}
        self.y_true_score = {}
        self.ap = {}
        self.ar = {}
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        for overlap in self.overlaps:
            self.y_true_cluster[overlap] = np.empty(0)
            self.y_true_score[overlap] = np.empty(0)
            self.hard_false_negatives[overlap] = 0

    def update_stats(self, pred_labels, gt_labels, confs=[]):
        pred_labels = self.filter_labels(pred_labels)
        self.calculate_full_stats(pred_labels,gt_labels)
        #if self.mode == 'normal':
        #    self.tps += self.calc_tp(pred_labels, gt_labels)
        #    unique_preds = np.unique(pred_labels).shape[0]
        #else:
        #    tps, unique_preds = self.calc_modified_tp(pred_labels, gt_labels)
        #    self.tps += tps

        #unique_preds = np.unique(pred_labels).shape[0]
        #if 0 in np.unique(pred_labels):
        #    unique_preds = unique_preds - 1

        if self.calc_ap :
            for overlap in self.overlaps:
           
                self.process_data_ap(
                    gt_labels, pred_labels,
                    cur_thresh=overlap,
                    confs=confs)

        #self.preds_total += unique_preds
        #unique_gt_num = np.unique(gt_labels).shape[0]
        #if 0 in np.unique(gt_labels):
        #    unique_gt_num  = unique_gt_num- 1 
        #self.unique_gts += unique_gt_num
        #self.get_metrics()
        return pred_labels

    def process_data_ap(
            self,
            ins_labels,
            pred,
            cur_thresh=0.5,
            confs=[]):
        
        
        unique_gt_labels = list(np.unique(ins_labels))
        unique_pred_labels = list(np.unique(pred))
        pred_used = set()
        instance_conf = {}
        for instance_id, i in enumerate(unique_pred_labels):
            if confs == []:
                # taken from unscene3d eval
                instance_conf[i] = 0.99 - (instance_id * 10e-3)
            else:
                cur_idcs = np.where(pred == i)
                # breakpoint()
                cur_idcs = cur_idcs[0]
                all_confs = confs[cur_idcs]
                mean_confs = all_confs.mean()
                instance_conf[i] = mean_confs

        cur_match = np.zeros(len(unique_gt_labels), dtype=bool)
        cur_true = np.ones(len(unique_gt_labels))
        cur_score = np.ones(len(unique_gt_labels)) * (-float("inf"))

        for gti, gt_label in enumerate(unique_gt_labels):
            if gt_label == self.background_label:
                continue
            found_match = False
            gt_indices = np.where(ins_labels == gt_label)
            gt_indices = gt_indices[0]
            for pred_label in unique_pred_labels:
                # avoid matching twice
                if pred_label in pred_used or pred_label == self.background_label:
                    continue

                pred_indices = np.where(pred == pred_label)
                pred_indices = pred_indices[0]
                iou_overlap = self.iou(pred_indices, gt_indices)
                if iou_overlap >= cur_thresh:
                    conf = instance_conf[pred_label]

                    if cur_match[gti]:
                        max_score = max(cur_score[gti], conf)
                        min_score = min(cur_score[gti], conf)
                        cur_score[gti] = max_score
                        # append false positive
                        cur_true = np.append(cur_true, 0)
                        cur_score = np.append(cur_score, min_score)
                        cur_match = np.append(cur_match, True)
                    else:  # set score as it is the first iteration
                        found_match = True
                        cur_match[gti] = True
                        cur_score[gti] = conf
                        pred_used.add(pred_label)
            if not found_match:
                try:
                    self.hard_false_negatives[cur_thresh] += 1
                except BaseException:
                    breakpoint()

        # remove non-matched ground truth instances
        cur_true = cur_true[cur_match]
        cur_score = cur_score[cur_match]

        # collect non-matched predictions as false positive
        for pred_id in unique_pred_labels:
            if pred_id == self.background_label:
                continue
            found_gt = False
            pred_indices = np.where(pred == pred_id)
            pred_indices = pred_indices[0]
            for gt_id in unique_gt_labels:
                if gt_id == 0 : 
                    found_gt = True 
                    break 
                gt_indices = np.where(ins_labels == gt_id)
                gt_indices = gt_indices[0]
                overlap = self.iou(pred_indices, gt_indices)
                if overlap > cur_thresh:
                    found_gt = True
                    break
            if not found_gt:
                background_idcs = np.where(ins_labels == 0)
                background_idcs = background_idcs[0]
                try:
                    overlap_ignore = self.iou(background_idcs, pred_indices)
                except BaseException:
                    overlap_ignore = 0
                # if not ignored append false positive
                if overlap_ignore <= cur_thresh:
                    cur_true = np.append(cur_true, 0)
                    confidence = instance_conf[pred_id]
                    cur_score = np.append(cur_score, confidence)

        # append to overall results
        self.y_true_cluster[cur_thresh] = np.append(
            self.y_true_cluster[cur_thresh], cur_true)
        self.y_true_score[cur_thresh] = np.append(
            self.y_true_score[cur_thresh], cur_score)

    def iou(self, pred_indices, gt_indices):
        intersection = np.intersect1d(pred_indices, gt_indices)
        union = np.union1d(pred_indices, gt_indices)

        return intersection.size / union.size

    def compute_final_ap(self, y_true, y_score, hard_false_negatives):
        if self.calc_ap == False : 
            return 
        score_arg_sort = np.argsort(y_score)
        y_score_sorted = y_score[score_arg_sort]
        y_true_sorted = y_true[score_arg_sort]
        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

        # unique thresholds
        (thresholds, unique_indices) = np.unique(
            y_score_sorted, return_index=True)
        num_prec_recall = len(unique_indices) + 1

        # prepare precision recall
        num_examples = len(y_score_sorted)
        try:
            num_true_examples = y_true_sorted_cumsum[-1]
        except BaseException:
            num_true_examples = 0

        precision_metric = np.zeros(num_prec_recall)
        recall_metric = np.zeros(num_prec_recall)

        # deal with the first point
        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
        # deal with remaining
        for idx_res, idx_scores in enumerate(unique_indices):
            cumsum = y_true_sorted_cumsum[idx_scores - 1]
            tp = num_true_examples - cumsum
            fp = num_examples - idx_scores - tp
            fn = cumsum + hard_false_negatives
            p = float(tp) / (tp + fp)
            r = float(tp) / (tp + fn)
            precision_metric[idx_res] = p
            recall_metric[idx_res] = r

        # first point in curve is artificial
        precision_metric[-1] = 1.
        recall_metric[-1] = 0.

        # compute average of precision-recall curve
        recall_for_conv = np.copy(recall_metric)
        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
        recall_for_conv = np.append(recall_for_conv, 0.)

        stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
        # integrate is now simply a dot product
        ap_current = np.dot(precision_metric, stepWidths)
        ar_current = np.dot(recall_metric, stepWidths)

        return ap_current, ar_current

    def compute_all_aps(self):
        if self.calc_ap == False : 
            return 
        for overlap in self.overlaps:
            self.ap[overlap], self.ar[overlap] = self.compute_final_ap(
                self.y_true_cluster[overlap], self.y_true_score[overlap], self.hard_false_negatives[overlap])
        print("APs for ", self.name)
        print("mAP@0.25 ", self.ap[0.25])
        print("mAP@0.5 ", self.ap[0.5])
        ap_overlaps = np.arange(0.5, 0.95, 0.05)
        ap = get_average([self.ap[overlap] for overlap in ap_overlaps])
        print("mAP ", ap)

    def get_metrics(self):
        precision = self.get_precision()
        recall = self.get_recall()
        fscore = self.get_f1(precision, recall)
        print("P/R Metrics for " + self.name)
        print("Precision ", round(precision * 100, 2))
        print("Recall ", round(recall * 100, 2))
        print("F1Score", round(fscore * 100, 2))

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

    def calc_tp(self, pred_labels, gt_labels):
        """
        :param pred_labels: labels of points corresponding to segmented instances
        :param gt_labels: labels of points corresponding to ground truth instances
        :param tp_condition: helper function to calculate statistics
        :return: true positive received using pred_labels and gt_labels
        """
        true_positive = 0
        pred_used = set()

        unique_gt_labels = np.unique(gt_labels)
        unique_pred_labels = np.unique(pred_labels)

        for gt_label in unique_gt_labels:
            gt_indices = np.where(gt_labels == gt_label)
            gt_indices = gt_indices[0]
            for pred_label in unique_pred_labels:
                if pred_label in pred_used or pred_label == 0:
                    continue

                pred_indices = np.where(pred_labels == pred_label)
                pred_indices = pred_indices[0]
                is_overlap = self.iou(pred_indices, gt_indices)

                if is_overlap > 0.5:
                    true_positive += 1
                    pred_used.add(pred_label)
                    break

        return true_positive

    def calc_modified_tp(self, pred_labels, gt_labels, overlap_threshold=0.5):
        true_positive = 0
        pred_used = set()
        num_preds = 0
        combined_preds = 0

        unique_gt_labels = np.unique(gt_labels)
        unique_pred_labels = np.unique(pred_labels)
        single_clusters = 0

        for gt_label in unique_gt_labels:
            gt_indices = np.where(gt_labels == gt_label)[0]
            combined_pred_indices = []

            for pred_label in unique_pred_labels:
                if pred_label in pred_used or pred_label == 0:
                    continue

                pred_indices = np.where(pred_labels == pred_label)[0]
                overlap = self.calculate_overlap(pred_indices, gt_indices)

                if overlap > 0.8:
                    combined_pred_indices.extend(pred_indices)
                    pred_used.add(pred_label)

            if combined_pred_indices == []:
                for pred_label in unique_pred_labels:
                    if pred_label in pred_used or pred_label == 0:
                        continue

                    pred_indices = np.where(pred_labels == pred_label)[0]
                    overlap = self.iou(pred_indices, gt_indices)

                    if overlap > 0.5:
                        combined_pred_indices = None
                        single_clusters += 1
                        pred_used.add(pred_label)
                        true_positive += 1
                        break

            if combined_pred_indices:
                combined_preds += 1
                combined_pred_indices = np.array(combined_pred_indices)
                combined_overlap = self.iou(combined_pred_indices, gt_indices)
                if combined_overlap > overlap_threshold:
                    true_positive += 1

        num_preds = unique_pred_labels.shape[0] - \
            len(pred_used) + combined_preds + single_clusters

        return true_positive, num_preds

    def calculate_overlap(self, indices1, indices2):
        intersection = len(np.intersect1d(indices1, indices2))
        overlap = intersection / len(indices1)
        return overlap

    def get_precision(self):
        return self.tps / self.preds_total

    def get_recall(self):
        return self.tps / self.unique_gts

    def get_f1(self, precision, recall):
        return 2 * precision * recall / (precision + recall)
    
    def calculate_full_stats(self,pred_indices,gt_indices):
        out = full_statistics(pred_indices,gt_indices,instanseg.metrics.iou,'iou')
        print(out)
        
        
        
        
        

if __name__ == "__main__":
    metrics_class = Metrics('test')