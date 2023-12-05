import os
import sys

sys.path.insert(0,"/Users/zyxu/Documents/py/NLP/com_icl")
import json
from tools.metric import EditDistance, WER
import numpy as np
import time
cache_dir = "./data/cache"
from lm_eval.utils import time_str


output_dir = "/Users/zyxu/Documents/py/NLP/com_icl/output/euler_output"

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path, indent = 4):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent = indent)

def extract(model = "meta-llama_Llama-2-70b-hf", task = "pr_lc"):
    
    load_file_path = f"output/euler_output/{model}/{task}_write_out_info.json"
    

    ed = EditDistance()
    wer = WER()
    
    data = load_json(file_path=load_file_path)
    start_time = time.time()
    distances = []
    wers = []
    for id, res in enumerate(data):
        
        pred = res['logit_0'].split(" ")
        pred = [item for item in pred if item != ""]
        ref = res['truth'].split(" ")
        if "" in ref:
            print(f"wrong example in {id}, ground_truth contains empty string")
            continue

        try:            
            dis0 = ed.minDistance(ref, pred)
            acc0 = ed.acc(ref, pred)

            dis1 = wer.minDistance(ref, pred)
            acc1 = wer.acc(ref, pred)

            distances.append(dis0)
            wers.append(acc0)

            if dis0 - dis1 != 0 or acc0 - acc1 != 0:
                print(f"id: {id}")
                print(f"edt distance: {dis0} | acc {acc0}")
                print(f"wer distance: {dis1} | acc {acc1}")
            
            # if id % 300 == 0:
            #     print(f"{id+1} examples")
            #     time_elapsed = time.time() - start_time
            #     print(f"==== time elapsed {time_str(time_elapsed)} | {time_str(time_elapsed/(id+1)*len(data))} ====")


        except Exception as e:
            # Print the error along with the id where it occurred
            print(f"Error at id {id}: {e}")
    

    mean_dis = np.array(distances).mean()#*100
    std_mean_dis = np.array(distances).std()/ np.sqrt(len(distances))#*100
    print(f"model {model} | task {task} | dis: mean: {mean_dis:.2f} | std: {std_mean_dis:.2f}")

    mean_wer = np.array(wers).mean()#*100
    std_mean_wer = np.array(wers).std()/ np.sqrt(len(wers))#*100
    print(f"model {model} | task {task} | wer: mean: {mean_wer:.2f} | std: {std_mean_wer:.2f}")

    res = {"model": model, "task": task, "wer": mean_wer, "wer_std": std_mean_wer, "dis": mean_dis, "dis_std": std_mean_dis}



    return res


        

def main():
    save_file_path = f"output/wer_total.json"

    if os.path.exists(save_file_path):
        if input('{} exists, exit? ([y]/n): '.format(save_file_path)) != 'n':
            return
    
    
    results = []

    model_list = [
        "gpt2", "gpt2-large", "gpt2-xl", "EleutherAI_gpt-j-6b", "EleutherAI_gpt-neox-20b",
        "meta-llama_Llama-2-7b-hf", "meta-llama_Llama-2-13b-hf", 
        "meta-llama_Llama-2-70b-hf", 
        "EleutherAI_pythia-70m", 
        "EleutherAI_pythia-410m", "EleutherAI_pythia-6.9b", "EleutherAI_pythia-12b"
        ]
    task_list = ["pr", "lc", "pr_lc"]
    # task_list = ["pr_lc_no_limit"]
    start_time = time.time()
    for task in task_list:
        for model in model_list:
            res = extract(model, task)
            results.append(res)

        time_elapsed = time.time() - start_time
        print(f"==== Done task {task}, time elapsed {time_str(time_elapsed)} ====")
    
    save_json(results, save_file_path)








if __name__ == '__main__':
    main()