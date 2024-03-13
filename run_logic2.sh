
#tasks=upper,twoSum,upper_twoSum,upper_twoSum_compose_incontext
#tasks=names2_upper,plusOne,plusOne_upper,plusOne_upper_compose_incontext
#tasks=names_upper,swap,upper_swap,upper_swap_compose_incontext
tasks=a_level,b_level,ab_level,ab_level_compose_incontext
declare -a num_fewshots=(10)
#dir=upper_plusOne
#
#declare -a models=("EleutherAI/pythia-2.8b" "EleutherAI/pythia-6.9b" "EleutherAI/pythia-12b" "huggyllama/llama-7b" "huggyllama/llama-13b" "huggyllama/llama-30b" "huggyllama/llama-65b")
#declare -a models=("EleutherAI/pythia-2.8b" "EleutherAI/pythia-6.9b" "EleutherAI/pythia-12b" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-70b-hf")
#declare -a models=("huggyllama/llama-65b")
#declare -a models=("meta-llama/Llama-2-70b-hf")
#declare -a models=("openai-community/gpt2-large" "EleutherAI/gpt-neo-1.3B" "EleutherAI/gpt-neo-2.7B" "EleutherAI/gpt-j-6b" "EleutherAI/gpt-neox-20B")
# declare -a models=("allenai/OLMo-1B" "allenai/OLMo-7B-Instruct")
declare -a models=("meta-llama/Llama-2-70b-hf")
declare -a seeds=(3407)

for seed in "${seeds[@]}"; do
    #dir=equation/upper_twoSum/"seed${seed}"
    #dir=upper_plusOne/"seed${seed}"
    #dir=swap/no_instruction
    dir=hierarchy   
    for model in "${models[@]}"; do
        model_filename=$(echo "$model" | tr '/' '_')
        echo "Model: $model"

        mkdir -p "output/${dir}/${model_filename}"
        for num_fewshot in "${num_fewshots[@]}"; do
            python main.py \
                --model hf-causal-experimental \
                --model_args pretrained="$model",use_accelerate=True,trust_remote_code=True \
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


