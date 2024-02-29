tasks=passive_to_active,obj_to_subj,compose_passive_to_active_obj_to_subj,compose_passive_to_active_obj_to_subj_incontext
# tasks=pr,lc,pr_lc,pr_lc_compose_incontext
declare -a num_fewshots=(0)
dir=cofe/pa_os/seed3407
#declare -a num_fewshots=(5 10 15)
#tasks=reverse_twoSum
#declare -a num_fewshots=(10 20 30)

#tasks=ciphar
#declare -a num_fewshots=(5 10 15)

#tasks=cipher_twoSum
#declare -a num_fewshots=(10 20 30)

#tasks=upper,twoSum,upper_twoSum,upper_twoSum_compose_incontext

#tasks=mod,twoSumPlus,mod_twoSum,mod_twoSum_compose_incontext
#tasks=upper,twoSum,upper_twoSum,upper_twoSum_compose_incontext
#tasks=names2_upper,plusOne,upper_plusOne,upper_plusOne_compose_incontext
#tasks=plusOne_upper,plusOne_upper_compose_incontext
#declare -a num_fewshots=(5 10 15)
#declare -a num_fewshots=(10 15)
#dir=upper_plusOne
#declare -a models=("EleutherAI/pythia-2.8b" "EleutherAI/pythia-6.9b" "EleutherAI/pythia-12b" "huggyllama/llama-7b" "huggyllama/llama-13b" "huggyllama/llama-30b" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-65b" "meta-llama/Llama-2-70b-hf")
declare -a models=("openai-community/gpt2-large" "EleutherAI/gpt-neo-1.3B" "EleutherAI/gpt-neo-2.7B" "EleutherAI/gpt-j-6b" "EleutherAI/gpt-neox-20B")
#declare -a models=("huggyllama/llama-65b" "meta-llama/Llama-2-70b-hf")

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
            --limit 100 \
            --description_dict_path templates/description.json \
            --num_fewshot "$num_fewshot" \
            --output_base_path "output/${dir}/${model_filename}" | tee "output/${dir}/${model_filename}/log.log"
    done
done

#--limit 100 \
