#!/bin/bash
#SBATCH --job-name=a5_bma_pl
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_bma_pl_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_bma_pl_%j.err

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "A5.9 Sequential Pipeline: train 3 specialists independently (MSE warmup → NLL) → train router"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
export PYTHONUNBUFFERED=1
ROUTER_MODE=${ROUTER_MODE:-mlp}
echo "Router mode: ${ROUTER_MODE}"
python3 lignos/scripts/train_a5_bma_pipeline.py \
    --n-seeds-per-specialist 3 --n-seeds-router 5 --router-mode ${ROUTER_MODE}
echo "Exit: $?  Done: $(date)"
