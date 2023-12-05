

python main.py \
    --model gpt2 \
    --model_args pretrained=gpt2 \
    --tasks pr \
    --device cuda \
    --write_out \
    --limit 2 \
    --output_base_path output/debug/gpt2
