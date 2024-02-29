


# tasks=dn,lc,pr,psa,ps,pr_lc_no_limit
# tasks=passive_to_active,obj_to_subj,compose_passive_to_active_obj_to_subj,pr_lc_compose_incontext,compose_passive_to_active_obj_to_subj_incontext

#tasks=reverse,twoSum
#declare -a num_fewshots=(5 10 15)

#tasks=reverse_twoSum
#declare -a num_fewshots=(10 20 30)

#tasks=ciphar
#declare -a num_fewshots=(5 10 15)

#tasks=cipher_twoSum
#declare -a num_fewshots=(10 20 30)

tasks=reverse_twoSum_compose_incontext,cipher_twoSum_compose_incontext
declare -a num_fewshots=(5 10 15)

declare -a models=("gpt2" "gpt2-large" "gpt2-xl")

for model in "${models[@]}"; do
    mkdir -p "output/equation/${model}"
    for num_fewshot in "${num_fewshots[@]}"; do
        echo "Model: $model"
        python main.py \
            --model gpt2 \
            --model_args pretrained=${model} \
            --tasks $tasks \
            --device cuda \
            --write_out \
            --limit 100 \
            --num_fewshot "$num_fewshot" \
            --output_base_path output/equation/${model} | tee "output/equation/${model}/log.log"
    done
done


declare -a models=("EleutherAI/pythia-70m" "EleutherAI/pythia-410m")

for model in "${models[@]}"; do
    model_filename=$(echo "$model" | tr '/' '_')
    echo "Model: $model"
    mkdir -p "output/equation/${model_filename}"
    for num_fewshot in "${num_fewshots[@]}"; do
        python main.py \
            --model hf-causal-experimental \
            --model_args pretrained="$model",use_accelerate=True \
            --tasks $tasks \
            --device cuda \
            --write_out \
            --limit 100 \
            --num_fewshot "$num_fewshot" \
            --output_base_path "output/equation/${model_filename}" | tee "output/equation/${model_filename}/log.log"
    done
done
