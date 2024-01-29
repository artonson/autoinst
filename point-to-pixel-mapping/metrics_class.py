import numpy as np
import instanseg 
from instanseg.metrics import full_statistics
from sklearn.metrics import auc
from modified_LSTQ import evaluator

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
        self.eval_lstq = evaluator()
        

        # ap stuff
        self.hard_false_negatives = {}
        self.y_true_cluster = {}
        self.y_true_score = {}
        self.ap = {}
        self.ar = {}
        self.overlaps = [0.25,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
        self.ap_overlaps = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
        print("Metrics for file",name)


    def update_stats(self, pred_labels, gt_labels, confs=[],calc_all=True,calc_lstq=True):
        self.eval_lstq.reset()
        pred_labels = self.filter_labels(pred_labels)
        if calc_all == True : 
            self.calculate_full_stats(pred_labels,gt_labels)
        if calc_lstq == True: 
            self.eval_lstq.add_batch(pred_labels,gt_labels)
            lstq = self.eval_lstq.get_eval()
            print('lstq value : ',lstq)
        
        
        for overlap in list(self.overlaps) : 
            print('calc AP for overlap @' + str(overlap))
            self.ap[overlap] = self.average_precision(pred_labels,gt_labels,confs,iou_thresh=overlap)
        
        print("AP @ 0.25",self.ap[0.25])
        print("AP @ 0.5",self.ap[0.5])
        aps_list = [self.ap[o] for o in self.ap_overlaps]
        print("AP @ [0.5:0.95]", sum(aps_list)/float(len(aps_list)))

        return pred_labels
        
    def average_precision(self,pred,ins_labels,confs,iou_thresh=0.5): 
        self.precision = []
        self.recall = []
        unique_gt_labels = list(np.unique(ins_labels))
        unique_pred_labels = list(np.unique(pred))
        
        # background remove
        unique_pred_labels.remove(0)
        unique_gt_labels.remove(0)

        pred_used = set()
        instance_conf = {}
        for instance_id, i in enumerate(unique_pred_labels):
            if confs == []:
                # taken from unscene3d eval
                instance_conf[i] = 0.5
            else:
                instance_conf[i] = confs[i].cpu().item()
        
        if confs != []: 
            instance_conf = dict(sorted(instance_conf.items(), key=lambda item: item[1],reverse=True))
        self.tp = 0
        self.fp = 0
        self.fn = len(unique_gt_labels)

        gt_used = []
        
        for prediction in instance_conf.keys():
            matched = False
            max_iou = 0 
            for gt in unique_gt_labels:
                pred_indices = np.where(pred == prediction)[0]
                gt_indices = np.where(ins_labels == gt)[0]
                
                iou = self.iou(pred_indices,gt_indices)
                max_iou = max(iou,max_iou)
                
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
    
        # Calculate AP
        ap = np.trapz(self.precision,self.recall) 
        print("Average Precision @ " + str(iou_thresh) ,ap )
        
        return ap

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

    def calculate_overlap(self, indices1, indices2):
        intersection = len(np.intersect1d(indices1, indices2))
        overlap = intersection / len(indices1)
        return overlap

    def calculate_full_stats(self,pred_indices,gt_indices):
        out = full_statistics(pred_indices,gt_indices,instanseg.metrics.iou,'iou')
        print(out)
        
        
        

if __name__ == "__main__":
    metrics_class = Metrics('test')