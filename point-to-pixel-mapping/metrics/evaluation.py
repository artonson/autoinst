import numpy as np
import glob
from maps import learning_map
from LSTQ_eval import Panoptic4DEval
from modified_LSTQ import evaluator
from tqdm import tqdm
import argparse

def read_labels_file(labels_path):
    labels = np.fromfile(labels_path, dtype=np.uint32)
    sem_labels = labels & 0xFFFF
    inst_labels = labels >> 16
    return sem_labels, inst_labels

def load_prediction(prediction_path):
    pre_labels = np.load(prediction_path)
    return pre_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=str, default="nc", help="prediction to evaluate")
    args = parser.parse_args()

    labels_dir = ("/home/zhang/hdd/semantickitti/semantickitti/sequences/07/labels")
    if args.prediction == "nc":
        prediction_dir = ("/home/zhang/hdd/nc_labels")
    elif args.prediction == "3duis":
        prediction_dir = ("/home/zhang/3DUIS/output/3DUIS/07/raw_pred")
    predictions = glob.glob(prediction_dir + "/*.npy")
    predictions.sort()

    class_evaluator = evaluator(min_points=0)
    pbar = tqdm(predictions, total=len(predictions))
    for prediction in pbar:
        filename = prediction.split('/')[-1].split('.')[0]
        label = labels_dir + '/' + filename + '.label'
        sem_gt, inst_gt = read_labels_file(label)
        mapped_sem_gt = np.array([learning_map[label] for label in sem_gt])
        inst_pred = load_prediction(prediction)
        if args.prediction == "3duis":
            inst_pred = inst_pred[:, -1]
        class_evaluator.add_batch(inst_pred, inst_gt)
        pbar.set_description(f"Processing {label}, {prediction}")
    AQ_ovr = class_evaluator.get_eval()
    np.save("S_asso.npy", class_evaluator.S_assoc_list)
    print(f"S_asso: {AQ_ovr}")
