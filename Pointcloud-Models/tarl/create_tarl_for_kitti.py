import os
import subprocess
import time
import select

seqs = list(range(2, 3))


input_base_path = "/media/cedric/Datasets2/semantic_kitti/sequences/"
output_base_path = "/media/cedric/Datasets2/semantic_kitti/tarl_features/"

for seq in seqs:
    cur_seq = str(seq).zfill(2)
    cur_input = input_base_path + cur_seq + "/" + "velodyne/"
    cur_output = output_base_path + cur_seq + "/"
    command = "python run.py -i" + cur_input + " -o " + cur_output
    subprocess.call(command, shell=True)
