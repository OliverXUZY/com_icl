
tasks=passive_to_active,obj_to_subj,compose_passive_to_active_obj_to_subj,pr_lc_compose_incontext,compose_passive_to_active_obj_to_subj_incontext



declare -a models=("EleutherAI/gpt-j-6b" "EleutherAI/gpt-neox-20b")

for model in "${models[@]}"; do
    model_filename=$(echo "$model" | tr '/' '_')
    echo "Model: $model"
    
    mkdir -p "output/${model_filename}"
    python main.py \
        --model hf-causal-experimental \
        --model_args pretrained="$model",use_accelerate=True \
        --tasks $tasks \
        --device cuda \
        --write_out \
        --output_base_path "output/${model_filename}" | tee "output/${model_filename}/log.log"

done

