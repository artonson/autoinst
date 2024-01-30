import numpy as np

class evaluator:
    def __init__(self, offset = 2 ** 32, min_points = 200):

        self.reset()
        self.offset = offset  # largest number of instances in a given scan
        self.min_points = min_points  # smallest number of points to consider instances in gt
        self.eps = 1e-15

    def reset(self):
        self.preds = []
        self.gts = []
        self.intersects = []
        

    def _update_dict(self, dict, key, value):
        if key in dict:
            dict[key] += value
        else:
            dict[key] = value
    
    def add_batch(self, pred_labels, gt_labels):
        pred_mask = np.logical_and(pred_labels != 0, pred_labels != -1)
        gt_mask = gt_labels != 0
        valid_pred = pred_labels[pred_mask]
        valid_gt = gt_labels[gt_mask]

        valid_pred_label, valid_pred_area = np.unique(valid_pred, return_counts=True)
        valid_gt_label, valid_gt_area = np.unique(valid_gt, return_counts=True)
        valid_gt_label = valid_gt_label[valid_gt_area > self.min_points]
        valid_gt_area = valid_gt_area[valid_gt_area > self.min_points]

        pred = {}
        gt = {}
        intersect = {}

        for i, label in enumerate(valid_pred_label):
            area = valid_pred_area[i]
            self._update_dict(pred, label, area)
        
        for i, label in enumerate(valid_gt_label):
            area = valid_gt_area[i]
            self._update_dict(gt, label, area)

        valid_intersect = np.logical_and(pred_labels > 0, gt_labels > 0)
        intersect_key = pred_labels[valid_intersect] + gt_labels[valid_intersect] * self.offset
        valid_intersect_label, valid_intersect_area = np.unique(intersect_key, return_counts=True)
        for i, label in enumerate(valid_intersect_label):
            area = valid_intersect_area[i]
            self._update_dict(intersect, label, area)
        
        self.gts.append(gt)
        self.preds.append(pred)
        self.intersects.append(intersect)
    
    def get_eval(self):
        preds = self.preds
        gts = self.gts
        intersects = self.intersects
        num_batches = len(self.gts)
        self.S_assoc_list = []
        for idx in range(num_batches):
            preds = self.preds[idx]
            gts = self.gts[idx]
            outer_sum_iou = 0.0
            intersects = self.intersects[idx]
            for gt_id, gt_area in gts.items():
                inner_sum_iou = 0.0
                for pred_id, pred_area in preds.items():
                    TPA_id = pred_id + gt_id * self.offset
                    if TPA_id in intersects:
                        TPA_area = intersects[TPA_id]
                        inner_sum_iou += TPA_area * (TPA_area / (gt_area + pred_area - TPA_area))
                outer_sum_iou += float(inner_sum_iou) / float(gt_area)
            S_assoc = outer_sum_iou / len(list(gts.items()))
            self.S_assoc_list.append(S_assoc)
        
        return np.average(self.S_assoc_list)