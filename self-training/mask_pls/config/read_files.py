import os
import numpy as np
from tqdm import tqdm

data_path = '/media/cedric/Datasets1/semantic_kitti/pseudo_labels/kitti/'

fs = os.listdir(data_path)

max_num = 0

for f in fs:
    print("folder", f)

    cur_files = os.listdir(data_path + f)
    for fn in tqdm(cur_files):
        if fn.endswith(".npz"):
            with np.load(data_path + f + '/' + fn) as data:
                xyz = data['pts'].astype(np.float)
                labels = data['ncut_labels'].astype(np.int32)
                cnt = 0
                for label in np.unique(labels):
                    idcs = np.where(labels == label)[0]
                    if idcs.shape[0] >= 100 : 
                        cnt += 1 
                    # intensity = np.ones_like(labels)
                
                max_num = max(max_num, cnt)


print(max_num)
