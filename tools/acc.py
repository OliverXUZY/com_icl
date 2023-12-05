import os
import sys
sys.path.insert(0,"/Users/zyxu/Documents/py/NLP/com_icl")
import json
import numpy as np
import evaluate
import time
from lm_eval.utils import time_str


cache_dir = "./data/cache"

output_dir = "/Users/zyxu/Documents/py/NLP/com_icl/output/euler_output"
exact_match = evaluate.load("exact_match")

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path, indent = 4):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent = indent)

def _normalize_answer(text):
    # strip whitespace
    if len(text) > 0 and text[0] == " ":
        # print(f"text =={text}==")
        text = text.strip()

    return text
    
def process_results(ref, pred):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        continuation = _normalize_answer(pred)
        answers = ref

        # print(f"continuation:  =={continuation}==")
        # print(f"answers: =={answers}==")

        preds = continuation.split(" ")
        refs = answers.split(" ")

        # Ensure both lists are of the same length by appending empty strings or take subset
        if len(refs) > len(preds):
            preds.extend([""] * (len(refs) - len(preds)))
        elif len(preds) > len(refs):
            preds = preds[:len(refs)]  # Slicing preds to match the length of refs
        
        results = exact_match.compute(references=refs, predictions=preds)
        
        return results['exact_match']


def extract(model = "meta-llama_Llama-2-70b-hf", task = "lc"):
    
    load_file_path = f"output/euler_output/{model}/{task}_write_out_info.json"

    
    data = load_json(file_path=load_file_path)
    start_time = time.time()
    accs = []
    for id, res in enumerate(data):


        pred = res['logit_0']
        ref = res['truth']
        

        try:
            
            acc = process_results(ref, pred)
            # print(f"id: {id} | acc:{acc}")

            accs.append(acc)

            if id % 10 == 0:
                print(f"{id+1} examples")
                time_elapsed = time.time() - start_time
                print(f"==== time elapsed {time_str(time_elapsed)} | {time_str(time_elapsed/(id+1)*len(data))} ====")


        except Exception as e:
            # Print the error along with the id where it occurred
            print(f"Error at id {id}: {e}")
        
        
    
    mean_acc = np.array(accs).mean()*100
    std_mean_acc = np.array(accs).std()/ np.sqrt(len(accs))*100
    print(f"model {model} | task {task} | acc: mean: {mean_acc:.2f} | std: {std_mean_acc:.2f}")

    res = {"model": model, "task": task, "acc": mean_acc, "std": std_mean_acc}
    return res
    

def main():
    save_file_path = f"output/exact_match_total.json"
    if os.path.exists(save_file_path):
        if input('{} exists, exit? ([y]/n): '.format(save_file_path)) != 'n':
            return
    
    
    results = []
    # model = "EleutherAI_gpt-neox-20b"

    model_list = [
        # "gpt2", "gpt2-large", "gpt2-xl", "EleutherAI_gpt-j-6b", "EleutherAI_gpt-neox-20b",
        # "meta-llama_Llama-2-7b-hf", "meta-llama_Llama-2-13b-hf", 
        "meta-llama_Llama-2-70b-hf", 
        #"EleutherAI_pythia-70m", 
        # "EleutherAI_pythia-410m", "EleutherAI_pythia-6.9b", "EleutherAI_pythia-12b"
        ]
    # task_list = ["pr", "lc", "pr_lc", "pr_lc_no_limit"]
    task_list = ["pr_lc_no_limit"]
    start_time = time.time()
    for task in task_list:
        for model in model_list:
            res = extract(model, task)
            results.append(res)

        time_elapsed = time.time() - start_time
        print(f"==== Done task {task}, time elapsed {time_str(time_elapsed)} ====")
    
    # save_json(results, save_file_path)



if __name__ == '__main__':
    main()