
# tasks=passive_to_active,obj_to_subj,compose_passive_to_active_obj_to_subj,pr_lc_compose_incontext,compose_passive_to_active_obj_to_subj_incontext
#tasks=reverse,twoSum
#declare -a num_fewshots=(5 10 15)

declare -a num_fewshots=(0)

tasks=ind_head_random

dir=induction_head/induction_head_random
declare -a models=("gpt2" "gpt2-xl")

for model in "${models[@]}"; do
    echo "Model: $model"
    mkdir -p "output/${dir}/${model}"
    python main.py \
        --model gpt2 \
        --model_args pretrained=${model} \
        --tasks $tasks \
        --device cuda \
        --write_out \
        --num_fewshot 0 \
        --limit 50 \
        --output_base_path "output/${dir}/${model}" | tee "output/${dir}/${model}/log.log"
done


declare -a models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf")

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


