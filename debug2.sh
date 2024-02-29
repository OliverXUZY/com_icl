
tasks=reverse


declare -a models=("gpt2-large" "gpt2-xl")

for model in "${models[@]}"; do
    echo "Model: $model"
    mkdir -p "output/equation/${model}"
    python main.py \
        --model gpt2 \
        --model_args pretrained=${model} \
        --tasks $tasks \
        --device cuda \
        --write_out \
        --num_fewshot 10 \
        --limit 10 \
        --output_base_path "output/equation/${model}" | tee "output/equation/${model}/log.log"
done


