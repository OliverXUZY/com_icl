#!/usr/bin/env bash
#
#SBATCH --output=./eulerlog/o_device_job_name_%j.out
#SBATCH --error=./eulerlog/o_device_job_name_%j.err
#SBATCH -J job_name  # give the job a name   
#***SBATCH --partition=batch_default ***
# 
# 1 node, 1 CPU per node (total 1 CPU), wall clock time of hours
#
#SBATCH -N 1                  ## Node count
#SBATCH --ntasks-per-node=1   ## Processors per node
#SBATCH --ntasks=1            ## Tasks
#SBATCH --gres=gpu:1          ## GPUs
#SBATCH --cpus-per-task=16     ## CPUs per task; number of threads of each task
#SBATCH -t 256:00:00          ## Walltime
#SBATCH --mem=80GB
#SBATCH -p lianglab
#SBATCH --exclude=euler[01-16],euler[20-28]
source ~/.bashrc
conda activate /srv/home/zxu444/anaconda3/envs/lmeval

echo "======== testing CUDA available ========"
echo "running on machine: " $(hostname -s)
python - << EOF
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
EOF

echo "======== run with different inputs ========"

run_command
