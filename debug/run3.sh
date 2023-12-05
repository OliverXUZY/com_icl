declare -a models=("meta-llama/Llama-2-7b-hf")
output_path=debug
for model in "${models[@]}"; do
    model_filename=$(echo "$model" | tr '/' '_')
    echo "Model: $model"
    mkdir -p "${output_path}/${model_filename}"
    python main.py \
        --model hf-causal-experimental \
        --model_args pretrained="$model",use_accelerate=True \
        --tasks pr \
        --device cuda \
        --write_out \
        --limit 2 \
        --output_base_path "output/debug/${model_filename}" | tee "${output_path}/debug/${model_filename}/log.log"
done
