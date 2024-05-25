
#tasks=upper,twoSum,upper_twoSum,upper_twoSum_compose_incontext
#tasks=names2_upper,plusOne,plusOne_upper,plusOne_upper_compose_incontext
# tasks=a_level,b_level,ab_level,ab_level_compose_incontext
# tasks=a_level_symbol
#tasks=a_level_symbol,ab_level_symbol,ab_level_compose_incontext_symbol
declare -a num_fewshots=(10)
tasks=reverse,twoSum,reverse_twoSum,ciphar,cipher_twoSum,reverse_twoSum_compose_incontext,cipher_twoSum_compose_incontext
#dir=upper_plusOne
#declare -a models=("EleutherAI/pythia-2.8b" "EleutherAI/pythia-6.9b" "EleutherAI/pythia-12b" "huggyllama/llama-7b" "huggyllama/llama-13b" "huggyllama/llama-30b")
#declare -a models=("EleutherAI/pythia-2.8b" "EleutherAI/pythia-6.9b" "EleutherAI/pythia-12b" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf")
#tasks=mod,twoSumPlus,mod_twoSum,mod_twoSum_compose_incontext
#tasks=names_upper,swap,upper_swap,upper_swap_compose_incontext
#,swap_upper,swap_upper_compose_incontext
# declare -a models=("openlm-research/open_llama_3b_v2")
# declare -a models=("openai-community/gpt2-large")
# declare -a models=("mistralai/Mixtral-8x7B-Instruct-v0.1")
#declare -a models=("google/gemma-2b-it" "google/gemma-7b-it") 
# declare -a models=("meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-v0.1")
# declare -a models=("mistralai/Mistral-7B-v0.1")
#declare -a models=("meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-Instruct-v0.2")
declare -a models=("meta-llama/Llama-2-7b-hf")
#declare -a models=("meta-llama/Llama-2-13b-hf"  "mistralai/Mixtral-8x7B-Instruct-v0.1" "meta-llama/Llama-2-70b-hf")
# declare -a models=("mistralai/Mistral-7B-v0.1" "mistralai/Mixtral-8x7B-v0.1")
#declare -a models=("allenai/OLMo-1B" "allenai/OLMo-7B-Instruct")

declare -a seeds=(3407)

for seed in "${seeds[@]}"; do
    #dir=equation/upper_twoSum/"seed${seed}"
    #dir=equation/mod_twoSum
    #dir=upper_plusOne/"seed${seed}"
    #dir=swap/no_instruction
    #dir=hierarchy/instruction  
    dir=twoSum_reverse_ciphar
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
                --description_dict_path "./templates/description.json" \
                --output_base_path "output/${dir}/${model_filename}" | tee "output/${dir}/${model_filename}/log.log"
        done
    done
done


