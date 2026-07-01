#!/bin/bash
#SBATCH --job-name=train_exp
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/train_expanded_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/train_expanded_%j.err

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"

echo "============================================"
echo "Train Expanded Combined(40D)"
echo "Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Date: $(date)"
echo "============================================"

# Use unbuffered Python output
export PYTHONUNBUFFERED=1

python3 lignos/scripts/train_expanded.py

echo "============================================"
echo "Done: $(date)"
echo "Exit: $?"
echo "============================================"
