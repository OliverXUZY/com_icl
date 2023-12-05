


# tasks=dn,lc,pr,psa,ps,pr_lc_no_limit
tasks=passive_to_active,obj_to_subj,compose_passive_to_active_obj_to_subj,pr_lc_compose_incontext
declare -a models=("gpt2" "gpt2-large" "gpt2-xl")

for model in "${models[@]}"; do
    echo "Model: $model"
    mkdir -p "output/${model}"
    python main.py \
        --model gpt2 \
        --model_args pretrained=${model} \
        --tasks $tasks \
        --device cuda \
        --write_out \
        --output_base_path output/${model} | tee "output/${model}/log.log"
done



declare -a models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "EleutherAI/pythia-70m" "EleutherAI/pythia-410m" "EleutherAI/pythia-6.9b" "EleutherAI/pythia-12b")

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
