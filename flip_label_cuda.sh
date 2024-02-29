declare -a num_fewshots=(16)

tasks=sst,qqp,rte
#tasks=sst
dir=flip_label

declare -a seeds=(42 89 156 3407 5678)
#declare -a seeds=(42)
declare -a models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf")
#declare -a models=("meta-llama/Llama-2-7b-hf")
for seed in "${seeds[@]}"; do
    for model in "${models[@]}"; do
        model_filename=$(echo "$model" | tr '/' '_')
        echo "Model: $model"

        mkdir -p "output/${dir}/${model_filename}"
        for num_fewshot in "${num_fewshots[@]}"; do
            python main.py \
                --model hf-causal-experimental \
                --model_args pretrained="$model",use_accelerate=True \
                --tasks $tasks \
                --device cuda \
                --write_out \
                --num_fewshot $num_fewshot \
                --limit 100 \
                --rnd_seed "$seed" \
                --output_base_path "output/${dir}/${model_filename}" | tee "output/${dir}/${model_filename}/log.log"
        done
    done
done
