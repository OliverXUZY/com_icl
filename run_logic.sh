
tasks=upper,twoSum,upper_twoSum,upper_twoSum_compose_incontext
#tasks=plusOne_upper,plusOne_upper_compose_incontext
declare -a num_fewshots=(10)
#dir=upper_plusOne
declare -a models=("EleutherAI/pythia-2.8b" "EleutherAI/pythia-6.9b" "EleutherAI/pythia-12b" "huggyllama/llama-7b" "huggyllama/llama-13b" "huggyllama/llama-30b")
#declare -a models=("EleutherAI/pythia-2.8b" "EleutherAI/pythia-6.9b" "EleutherAI/pythia-12b" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf")
declare -a seeds=(3407)

for seed in "${seeds[@]}"; do
    dir=equation/upper_twoSum/"seed${seed}"
    #dir=upper_plusOne/"seed${seed}"
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


