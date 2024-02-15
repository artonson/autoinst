import os 
import subprocess 
import time 



input_base_path = '/media/cedric/Datasets1/nuScenes_train/samples/LIDAR_TOP/'
output_base_path = '/media/cedric/Datasets1/nuScenes_train/outputs/TARL/'

command = 'python run.py -i' + input_base_path + ' -o ' + output_base_path + ' -d ' + 'nuscenes'
subprocess.call(command,shell=True)




