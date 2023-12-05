


# tasks=dn,lc,pr,psa,ps
tasks=pr_lc


mkdir -p "output/gpt2"
python main.py \
    --model gpt2 \
    --model_args pretrained=gpt2 \
    --tasks $tasks \
    --device cuda \
    --write_out \
    --output_base_path output/gpt2 | tee "output/gpt2/log.log"


declare -a models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf")

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
