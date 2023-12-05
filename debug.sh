

(
    while true; do
        nvidia-smi | tee -a ./log/gpu_usage_${SLURM_JOB_ID}.log
        sleep 60  # Log every 600 seconds
    done
) &
monitor_pid=$!

python main.py \
    --model gpt2 \
    --model_args pretrained=gpt2 \
    --tasks dn,lc,pr,psa,ps \
    --device cuda \
    --write_out \
    --output_base_path debug
