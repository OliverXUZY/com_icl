import sys
sys.path.append("/Users/zyxu/Documents/py/NLP/com_icl")
import argparse
import os
import numpy as np
import json

def parse_str_to_dict(d):
    dictionary = dict()
    # Removes curly braces and splits the pairs into a list
    pairs = d.strip('\n').strip('{}').split(', ')
    for i in pairs:
        pair = i.split(': ')
        if len(pair) < 2:
            continue
        # Other symbols from the key-value pair should be stripped.
        dictionary[pair[0].strip('\'\'\"\"')] = pair[1].strip('\'\'\"\"')
    return dictionary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, help="A dictionary contains conditions that the experiment results need to fulfill (e.g., tag, task_name, few_shot_type)")
    
    # These options should be kept as their default values
    parser.add_argument("--log_path", type=str, default="output/multitask/meta-llama_Llama-2-7b-hf/", help="Log path.")
    parser.add_argument("--log", type=str, default="log_summary.json", help="Log file.")
    # parser.add_argument("--key", type=str, default='', help="Validation metric name")

    args = parser.parse_args()

    condition = eval(args.condition)

    ## load result
    result_list = []
    with open(os.path.join(args.log_path, args.log)) as f:
        for line in f:
            result_list.append(parse_str_to_dict(line))

    ## set up key
    if condition['task'] == 'cola':
        key = 'mcc'
    # elif condition['task'] == 'qqp':
    #     key = 'f1'
    else:
        key = 'acc'
    
    # print("zhuoyan==========================================================================")
    print("condition is: ", condition)
    



    seed_result = {}
    seed_best = {}

    for item in result_list:
        ok = True
        for cond in condition:
            if isinstance(condition[cond], list):
                if cond not in item or (item[cond] not in condition[cond]):
                    ok = False
                    break
            else:
                # print(cond)
                if cond not in item or (item[cond] != condition[cond]):
                    ok = False
                    break
        if ok:
            # print("ok")
            seed = str(item['rnd_seed'])
            if seed not in seed_result:
                seed_result[seed] = [item]
                seed_best[seed] = item
            else:
                seed_result[seed].append(item)
                if item[key] > seed_best[seed][key]:
                    seed_best[seed] = item

    final_result_test = np.zeros((len(seed_best)))
    for i, seed in enumerate(seed_best):
        final_result_test[i] = seed_best[seed][key]
        print("{}: best test ({}: {:.5f}) | total trials: {}".format(
            seed,
            key,
            float(seed_best[seed][key]),
            len(seed_result[seed])
        ))
        s = ''
        for k in ['prompt_des', 'num_fewshot', 'batch_size']:
            s += '| {}: {} '.format(k, seed_best[seed][k])
        print('    ' + s)
    
    s = f"mean +- std: {final_result_test.mean() * 100:.2f} ({final_result_test.std() * 100:.2f}) (median {np.median(final_result_test) * 100:.2f})"
    print(s)

    # print("final_result_test: ", final_result_test)

    print("seed_best: ", seed_best)
    print(f"raw acc: {float(seed_best[seed]['acc'])*100:.2f} ({float(seed_best[seed]['acc_stderr'])*100:.2f})==")
    write_mean_txt = False
    write_all_json = False
    # print("zhuoyan==========================================================================")
    
    if write_mean_txt:
        with open(os.path.join(args.log_path,"mean.txt"), 'a') as f:
            f.write(str(condition['task']) + "," + str(condition['num_fewshot']) + "," 
                    + str(final_result_test.mean() * 100) + "," 
                    + str(final_result_test.std() * 100) 
                    + '\n')
    
    if write_all_json:
        
        json_path = os.path.join(args.log_path,"seed_result.json")
        # Read existing data if file exists
        if os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                json_result = json.load(json_file)
        else:
            json_result = []
            
        for seed in seed_best:
            tmp = {
                "seed": seed,
                "task": condition["task"],
                "num_fewshot": condition["num_fewshot"],
                "metric": key,
                "value": float(seed_best[seed][key])*100
            }
            json_result.append(tmp)
        
        # Open the file in write mode and save the dictionary as JSON
        with open(json_path, 'w') as json_file:
            json.dump(json_result, json_file, indent=4)
        
    

if __name__ == '__main__':
    main()
