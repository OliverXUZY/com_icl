
declare -a num_fewshots=(0)

tasks=ind_head_random

dir=induction_head/induction_head_random


declare -a models=("meta-llama/Llama-2-70b-hf")

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
            --num_fewshot 0 \
            --limit 50 \
            --output_base_path "output/${dir}/${model_filename}" | tee "output/${dir}/${model_filename}/log.log"
    done
done


