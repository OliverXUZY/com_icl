import os
import sys
sys.path.insert(0, "/Users/zyxu/Documents/py/NLP/com_icl")
import json
from datasets import load_dataset
from lm_eval.tasks.cofe import cofe


cache_dir = "./cofe/cache"
data_files = 'cofe/only_primitive_coverage.json'
data_dir = 'cofe/only_primitive_coverage'

def main():
    # dataset = load_dataset(
    #     path = "json", 
    #     name = None, 
    #     data_files=data_files, 
    #     cache_dir=cache_dir, 
    #     split = "train"
    # )
    dataset = load_dataset("glue", "sst2",cache_dir=cache_dir)
    print(dataset['train'][0])


    task = cofe()
    print(task.dataset)
    print(task.dataset[0])


if __name__ == "__main__":
    main()