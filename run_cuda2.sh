tasks=passive_to_active,obj_to_subj,compose_passive_to_active_obj_to_subj



declare -a models=("meta-llama/Llama-2-70b-hf")

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

