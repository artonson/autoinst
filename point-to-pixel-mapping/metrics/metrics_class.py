import numpy as np
import instanseg
from instanseg.metrics import full_statistics
from metrics.modified_LSTQ import evaluator
from multiprocessing import Pool
import open3d as o3d
import random 
import copy 
import matplotlib.pyplot as plt


COLORS_arr = plt.cm.viridis(np.linspace(0, 1, 30)) 
COLORS = list(list(col) for col in COLORS_arr) 
COLORS = [list(col[:3]) for col in COLORS]

instanseg.metrics.constants.UNSEGMENTED_LABEL = 0
instanseg.metrics.constants.IOU_THRESHOLD_FULL = 0.5


def iou(
    pred_indices,
    gt_indices
    ) : 
    intersection = np.intersect1d(pred_indices, gt_indices)
    union = np.union1d(pred_indices, gt_indices)

    return intersection.size / union.size





def find_index_2d_list(my_list, target_sublist):
    for i, sublist in enumerate(my_list):
        print(sublist)
        print(target_sublist)
        print(sublist == list(target_sublist))
        if sublist == list(target_sublist):
            
            return i
    return None  # Sublist not found in the list


def generate_random_colors(N, seed=0):
    colors = set()  # Use a set to store unique colors
    while len(colors) < N:  # Keep generating colors until we have N unique ones
        # Generate a random color and add it to the set
        colors.add((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    return list(colors)  # Convert the set to a list before returning

def generate_random_colors_map(N, seed=0):
    random.seed(42)
    colors = set()  # Use a set to store unique colors
    while len(colors) < N:  # Keep generating colors until we have N unique ones
        # Generate a random color and add it to the set
        colors.add((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    return list(colors)  # Convert the


def color_pcd_by_labels(pcd, labels,colors=None,largest=True):
    
    if colors == None : 
        colors = generate_random_colors(2000)
    pcd_colored = copy.deepcopy(pcd)
    pcd_colors = np.zeros(np.asarray(pcd.points).shape)
    unique_labels = list(np.unique(labels)) 
    
    #background_color = np.array([0,0,0])


    #for i in range(len(pcd_colored.points)):
    largest_cluster_idx = -10 
    largest = 0 
    if largest : 
        for i in unique_labels: 
            idcs = np.where(labels == i)
            idcs = idcs[0]
            if idcs.shape[0]> largest:
                largest = idcs.shape[0]
                largest_cluster_idx = i 
    
    for i in unique_labels:
        if i == -1 : 
            continue
        idcs = np.where(labels == i)
        idcs = idcs[0]
        if i == largest_cluster_idx or i == 0 : 
            pcd_colors[idcs] = np.array([0,0,0])
        else : 
            pcd_colors[idcs] = np.array(colors[unique_labels.index(i)])
        
        #if labels[i] != (-1):
        #    pcd_colored.colors[i] = np.array(colors[labels[i]]) / 255
    pcd_colored.colors = o3d.utility.Vector3dVector(pcd_colors/ 255)
    return pcd_colored
    




def get_average(l):
    return sum(l) / len(l)


class Metrics:

    def __init__(self, name='NCuts',min_points=200,thresh=0.5):
        self.thresh = thresh  # iou threshold being used
        self.tps = 0
        self.unique_gts = 0
        self.preds_total = 0
        self.name = name
        self.min_points = min_points
        self.background_label = 0
        self.mode = 'normal'
        self.calc_ap = True
        self.eval_lstq = evaluator(min_points=min_points)
        self.eval_lstq.reset()
        self.semantic_intersection = 0.05
        self.num_processes = 6
        
        
        self.labels_dict = {
                        0: "unlabeled",
                        1: "outlier",
                        10: "car",
                        11: "bicycle",
                        13: "bus",
                        15: "motorcycle",
                        16: "on-rails",
                        18: "truck",
                        20: "other-vehicle",
                        30: "person",
                        31: "bicyclist",
                        32: "motorcyclist",
                        40: "road",
                        44: "parking",
                        48: "sidewalk",
                        49: "other-ground",
                        50: "building",
                        51: "fence",
                        52: "other-structure",
                        60: "lane-marking",
                        70: "vegetation",
                        71: "trunk",
                        72: "terrain",
                        80: "pole",
                        81: "traffic-sign",
                        99: "other-object",
                        252: "moving-car",
                        253: "moving-bicyclist",
                        254: "moving-person",
                        255: "moving-motorcyclist",
                        256: "moving-on-rails",
                        257: "moving-bus",
                        258: "moving-truck",
                        259: "moving-other-vehicle"
                    }
        self.cols = COLORS
        class_names = ['unlabeled',
                        'car',
                        'parking',
                        'sidewalk',
                        'fence',
                        'trailer',
                        'truck',
                        'traffic_sign',
                        'other_ground',
                        'vegetation'
                        ]

        
        #self.relevant_idcs = {1:'car',3:'fence',4:'truck',6:'vegetation'}

        # ap stuff
        self.ap = {}
        self.ar = {}
        self.overlaps = [
            0.25,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95]
        self.ap_overlaps = [
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95]
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
        
        
        for pred_label in np.unique(pred_labels):
            if pred_label == 0 : 
                continue
            matched = False
            pred_indices = np.where(pred_labels == pred_label)[0]
            cur_iou = None 
            for gt in np.unique(gt_labels):
                if gt == 0 : 
                    continue

                gt_indices = np.where(gt_labels == gt)[0]

                iou = self.iou(pred_indices, gt_indices)

                if iou >= iou_thresh and (gt not in gt_used):
                    matched = True
                    true_positive += 1
                    cur_iou = iou
                    gt_used.add(gt)
                    break
                    
            if matched:
                self.all_matches[iou_thresh].append({'result':'tp','iou':cur_iou})
                
            else:
                false_positives += 1
                self.all_matches[iou_thresh].append({'result':'fp'})

        return true_positive, false_positives
        '''  
        ################################################################
        for gt_label in np.unique(gt_labels):
            if gt_label == 0 : 
                continue
        
            gt_indices = np.where(gt_labels == gt_label)[0]
            for pred_label in np.unique(pred_labels):
                matched = False 
                if pred_label in pred_used or pred_label == 0 :
                    continue
    
                pred_indices = np.where(pred_labels == pred_label)[0]
                is_overlap = iou(pred_indices,gt_indices) >= iou_thresh
    
                if is_overlap:
                    true_positive += 1
                    pred_used.add(pred_label)
                    matched = True 
                    self.all_matches[iou_thresh].append({'result':'tp'})
                    break
            
            if matched == False : 
                self.all_matches[iou_thresh].append({'result':'fp'}) 
                false_positives += 1 
        '''

    
        return true_positive, false_positives 

    def worker_function(self, data):
        # This function will be executed by each process
        # 'data' contains a subset of predictions and other necessary information
        pred, ins_labels, confs, iou_thresh = data
        # Perform the calculation for this chunk
        # Return the partial results (e.g., precision and recall for this
        # chunk)
        return self.average_precision(pred, ins_labels, confs, iou_thresh)

    def average_precision_parallel(
            self,
            pred,
            ins_labels,
            confs,
            iou_thresh=0.5):
        # Split the predictions into chunks

        # Prepare data for each chunk
        data_for_processes = [(pred, ins_labels, confs, iou)
                              for iou in self.overlaps]

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
            calc_lstq=True):
        
        pred_labels = self.filter_labels(pred_labels)
        gt_labels = self.filter_labels(gt_labels)
        all_labels = self.filter_labels(all_labels)
    
        if calc_all:
            self.calculate_full_stats(pred_labels, gt_labels)
        if calc_lstq:
            self.eval_lstq.add_batch(all_labels, gt_labels)
            lstq = self.eval_lstq.get_eval()
            print('S_assoc score : ', lstq)
        if self.calc_ap :
            self.average_precision_parallel(pred_labels, gt_labels, confs)
            print("AP @ 0.25", round(self.ap[0.25] * 100, 3))
            print("AP @ 0.5", round(self.ap[0.5] * 100,3))
            aps_list = [self.ap[o] for o in self.ap_overlaps]
            ap = sum(aps_list) / float(len(aps_list))
            print("AP @ [0.5:0.95]", round(ap * 100, 3))
    
    def add_stats(self,
            all_labels,
            pred_labels,
            gt_labels,
            confs=[],
            calc_all=True,
            calc_lstq=True): 
        
        pred_labels = self.filter_labels(pred_labels)
        gt_labels = self.filter_labels(gt_labels)
        all_labels = self.filter_labels(all_labels)
        
        self.eval_lstq.add_batch(all_labels, gt_labels)        
        
        for overlap in self.overlaps: 
            tps, fps = self.get_tp_fp(pred_labels,gt_labels,iou_thresh=overlap)
            if 0 in gt_labels : 
                self.all_gt_size[overlap] += np.unique(gt_labels).shape[0] - 1
            self.all_pred_size[overlap] += np.unique(pred_labels).shape[0] - 1
            self.all_tp[overlap] += tps 
        

    def average_precision(self, pred, ins_labels, confs, iou_thresh=0.5):
        self.precision = [1.0]
        self.recall = [0.0] ##Init with values to ensure correct computation 
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
                instance_conf[i] = confs[i]

        if confs != []:
            instance_conf = dict(
                sorted(
                    instance_conf.items(),
                    key=lambda item: item[1],
                    reverse=True))
        self.tp = 0
        self.fp = 0
        self.fn = len(unique_gt_labels)

        gt_used = []

        for prediction in instance_conf.keys():
            matched = False
            pred_indices = np.where(pred == prediction)[0]
            for gt in unique_gt_labels:

                gt_indices = np.where(ins_labels == gt)[0]

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
        print("Average Precision @ " + str(iou_thresh), ap)
        return ap
    
    def average_precision_final(self, iou_thresh=0.5):
        
        
        tp = 0
        fp = 0
        fn = self.all_gt_size[iou_thresh]

        for matched in self.all_matches[iou_thresh]: 
            result = matched['result']
            if result == 'tp' :
                tp += 1
                fn -= 1
            else:
                fp += 1

            # Calculate precision and recall
            self.precision_all[iou_thresh].append(tp / float(tp + fp))
            self.recall_all[iou_thresh].append(tp / float(tp + fn))
    
    def compute_stats_final(self): 
        for overlap in self.overlaps: 
            self.average_precision_final(iou_thresh=overlap)
            self.ap[overlap] = np.trapz(self.precision_all[overlap],self.recall_all[overlap])
        
        
        prec = self.all_tp[self.thresh]/self.all_pred_size[self.thresh]
        rec = self.all_tp[self.thresh]/self.all_gt_size[self.thresh]
        f1 = 2 * (prec * rec)/(prec + rec)
        lstq = self.eval_lstq.get_eval()
        
        
        print("Precison @ " + str(self.thresh),prec)
        print("Recall @ " + str(self.thresh),rec)
        print("F Score @ " + str(self.thresh), f1)
        mean = self.mean()
        print('Mean @ ' + str(self.thresh),mean)
        print('Panoptic @ ' + str(self.thresh) ,mean* f1)
        print('S_assoc score',lstq)
        
        print("AP @ 0.25", round(self.ap[0.25] * 100, 3))
        print("AP @ 0.5", round(self.ap[0.5] * 100,3))
        aps_list = [self.ap[o] for o in self.ap_overlaps]
        ap = sum(aps_list) / float(len(aps_list))
        print("AP @ [0.5:0.95]", round(ap * 100, 3))
        
    def mean(self):
        mean_array = []
        for instance in self.all_matches[self.thresh] : 
                if instance['result'] == 'tp': 
                    mean_array.append(instance['iou'])
        
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

    def remove_cols(self, pcd, cols, labels, unique_labels, idx=1):

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = pcd.points
        new_cols = np.zeros((np.asarray(pcd.points).shape[0], 3))
        pcd_cols = np.asarray(pcd.colors)
        idcs = np.where(labels == idx)[0]
        new_cols[idcs] = cols[unique_labels.index(idx)]
        new_pcd.colors = o3d.utility.Vector3dVector(new_cols)
        o3d.visualization.draw_geometries([new_pcd])
        return new_pcd, idcs

    def intersect(self, pred_indices, gt_indices):
        intersection = np.intersect1d(pred_indices, gt_indices)
        return intersection.size / pred_indices.shape[0]

    def get_semantics(self, preds, gt_idcs, pcd):
        new_ncuts_labels = np.zeros_like(preds)
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = pcd.points
        num_clusters = 0
        for i in np.unique(preds):
            # new_cols = np.zeros((np.asarray(pcd.points).shape[0],3))

            pred_idcs = np.where(preds == i)[0]
            cur_intersect = self.intersect(pred_idcs, gt_idcs)

            # new_cols[pred_idcs] = np.array([1,0,0])
            # new_pcd.colors = o3d.utility.Vector3dVector(new_cols)
            # o3d.visualization.draw_geometries([new_pcd])

            if cur_intersect > self.semantic_intersection:
                # print("intersect",cur_intersect)
                new_ncuts_labels[pred_idcs] = 1
                num_clusters += 1
        #o3d.visualization.draw_geometries([color_pcd_by_labels(pcd,new_ncuts_labels,largest=True)])
        return np.where(new_ncuts_labels == 1)[0], num_clusters

    def semantic_eval(self, pred_pcd, pcd_gt,labels):
        unique_colors_kitti, labels_kitti = np.unique(
            np.asarray(pcd_gt.colors), axis=0, return_inverse=True)
        unique_colors, labels_ncuts = np.unique(
            np.asarray(pred_pcd.colors), axis=0, return_inverse=True)

        for idx in list(np.unique(labels)):
            print('file idx',idx)
            print("current class",self.labels_dict[idx])
            #print("Semantics for class", self.relevant_idcs[idx])
            gt_pcd, idcs = self.remove_cols(
                    pcd_gt, unique_colors_kitti,labels.reshape(-1,1), list(
                        np.unique(labels)), idx)
    
                #pred_labels, num_clusters = self.get_semantics(
                #    labels_ncuts, idcs, pred_pcd)
    
                #cur_iou = self.iou(pred_labels, idcs)
                #print("--- iou",cur_iou)
                #print("--- num clusters",num_clusters)
                #print("--- semantic score", cur_iou/(num_clusters + 1e-8))
                #print("-------------------")

    def calculate_full_stats(self, pred_indices, gt_indices):
        out = full_statistics(
            pred_indices,
            gt_indices,
            instanseg.metrics.iou,
            'iou')
        print(out)


if __name__ == "__main__":
    metrics_class = Metrics('test')
