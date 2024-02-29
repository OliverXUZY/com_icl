

seed=1234
tasks=passive_to_active,obj_to_subj,compose_passive_to_active_obj_to_subj,compose_passive_to_active_obj_to_subj_incontext
dir=cofe/pa_os/"seed${seed}"
#tasks=pr,lc,pr_lc,pr_lc_compose_incontext
#tasks=pr_lc_compose_incontext
#dir=cofe/pr_lc
declare -a num_fewshots=(0)
##tasks=reverse,twoSum
#declare -a num_fewshots=(5 10 15)

#tasks=names_upper,swap,upper_swap,upper_swap_compose_incontext,swap_upper,swap_upper_compose_incontext

#tasks=upper,twoSum,upper_twoSum,upper_twoSum_compose_incontext
#tasks=names2_upper,plusOne,upper_plusOne,upper_plusOne_compose_incontext
#tasks=plusOne_upper,plusOne_upper_compose_incontext
#declare -a num_fewshots=(5 10 15)
#dir=upper_plusOne
#declare -a models=("EleutherAI/pythia-2.8b" "EleutherAI/pythia-6.9b" "EleutherAI/pythia-12b" "huggyllama/llama-7b" "huggyllama/llama-13b" "huggyllama/llama-30b" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf")
#declare -a models=("huggyllama/llama-7b" "huggyllama/llama-13b" "huggyllama/llama-30b")
#declare -a models=("EleutherAI/pythia-6.9b" "EleutherAI/pythia-12b")
declare -a models=("openlm-research/open_llama_3b_v2")
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
            --rnd_seed "$seed" \
            --description_dict_path templates/description.json \
            --num_fewshot "$num_fewshot" \
            --output_base_path "output/${dir}/${model_filename}" | tee "output/${dir}/${model_filename}/log.log"
    done
done


#--limit 100 \
