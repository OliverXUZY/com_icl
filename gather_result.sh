tasks=("dn" "lc" "pr" "psa" "ps" "cofe" "pr_lc")

# Define the num_fewshot values
num_fewshots=(0)
models=('gpt2' 'meta-llama_Llama-2-7b-hf' 'meta-llama_Llama-2-13b-hf')
# models=('meta-llama_Llama-2-70b-hf')

# Loop over each model
for model in "${models[@]}"; do
    # Loop over each task
    for task in "${tasks[@]}"; do
        # Loop over each num_fewshot value
        for num_fewshot in "${num_fewshots[@]}"; do
            # Run the Python script with the specified condition
            python tools/gather_result.py \
                --condition "{'task': '$task', 'num_fewshot': '$num_fewshot'}" \
                --log_path "./output/$model"
        done
    done
done
