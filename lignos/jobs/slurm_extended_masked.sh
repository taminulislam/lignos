#!/bin/bash
#SBATCH --job-name=ext_mask
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/ext_mask_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/ext_mask_%j.err

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"

echo "============================================"
echo "Extended training on FIXED LignoIL_unified cache"
echo "  Tests: baseline | extended_unmasked | extended_masked (3 arms × 10 seeds)"
echo "Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Date: $(date)"
echo "============================================"

export PYTHONUNBUFFERED=1
python3 lignos/scripts/train_extended_masked.py

echo "============================================"
echo "Done: $(date)"
echo "Exit: $?"
echo "============================================"
