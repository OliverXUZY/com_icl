


# tasks=dn,lc,pr,psa,ps
tasks=mod_twoSum


#mkdir -p "output/equation/gpt2"
#python main.py \
#    --model gpt2 \
#    --model_args pretrained=gpt2 \
#    --tasks $tasks \
#    --device cuda \
#    --write_out \
#    --output_base_path output/gpt2 | tee "output/gpt2/log.log"

declare -a seeds=(21)

declare -a models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf")

#declare -a models=("meta-llama/Llama-2-7b-hf")
for seed in "${seeds[@]}"; do
    for model in "${models[@]}"; do
        model_filename=$(echo "$model" | tr '/' '_')
        echo "Model: $model"
    
        mkdir -p "output/debug/${model_filename}/$seed"
        python main.py \
            --model hf-causal-experimental \
            --model_args pretrained="$model",use_accelerate=True \
            --tasks $tasks \
            --device cuda \
            --write_out \
            --num_fewshot 10 \
            --limit 100 \
            --rnd_seed 42 \
            --output_base_path "output/debug/${model_filename}/$seed" | tee "output/debug/${model_filename}/$seed/log.log"
    done
done



dataset = datasets.load_dataset(
        path="glue",
        name="sst2",
        data_dir=None,
        cache_dir=None,
        download_mode=None,
    )


    print("zhuoyan--===")

    dataset = datasets.load_dataset(
        path="glue",
        name="sst2",
        data_dir=None,
        cache_dir=None,
        download_mode=None,
    )
    print("zyxuzxu444")
    assert False
    task = lm_eval.tasks.glue.SST()
    print("zhuoyan: ", task_dict)
