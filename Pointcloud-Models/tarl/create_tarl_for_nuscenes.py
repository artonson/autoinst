import os 
import subprocess 
import time 



input_base_path = '/media/cedric/Datasets21/nuScenes_mini/nuScenes/samples/LIDAR_TOP/'
output_base_path = '/media/cedric/Datasets21/nuScenes_mini/nuScenes/outputs/TARL/LIDAR_TOP/'

command = 'python run.py -i' + input_base_path + ' -o ' + output_base_path + ' -d ' + 'nuscenes'
subprocess.call(command,shell=True)




