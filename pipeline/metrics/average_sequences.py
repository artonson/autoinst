import os 
import json

if __name__ == "__main__":
    results_dir = 'results/'
    files = os.listdir(results_dir)
    total_results = {'ap':[],'ap0.25':[],'ap0.5':[],'p':[],'r':[],'f1':[],'S_assoc':[]}
    for f in files : 
        pth = results_dir + f
        with open(pth,'r') as json_f:
            data = json.load(json_f)
        for key in total_results.keys():
            total_results[key].append(data[key])
    
    for key in total_results.keys():
        print(f"Result for Metric {key} : {sum(total_results[key])/float(len(total_results[key]))}")
