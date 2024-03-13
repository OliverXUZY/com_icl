
tasks=proofwriter
declare -a num_fewshots=(0)
# declare -a models=("openlm-research/open_llama_3b_v2")
# declare -a models=("openai-community/gpt2-large")
# declare -a models=("google/gemma-2b-it" "google/gemma-7b-it" "mistralai/Mistral-7B-Instruct-v0.2" "mistralai/Mixtral-8x7B-Instruct-v0.1")
#declare -a models=("allenai/OLMo-1B" "allenai/OLMo-7B-Instruct")
# declare -a models=("meta-llama/Llama-2-7b-hf")
declare -a models=("meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-70b-hf")
declare -a seeds=(3407)

for seed in "${seeds[@]}"; do
    #dir=equation/upper_twoSum/"seed${seed}"
    #dir=equation/mod_twoSum
    #dir=upper_plusOne/"seed${seed}"
    #dir=swap/no_instruction
    dir=proofwriter
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


