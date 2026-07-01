#!/bin/bash
#SBATCH --job-name=a2_2stg
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a2_2stg_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a2_2stg_%j.err

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"

echo "============================================"
echo "A2 two-stage: A2_chemprop Stage-1 (Morgan+ChemProp gated) + Stage-2 frozen + deep lignin head"
echo "Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Date: $(date)"
echo "============================================"

export PYTHONUNBUFFERED=1
python3 lignos/scripts/train_a2_two_stage.py

echo "============================================"
echo "Done: $(date)"
echo "Exit: $?"
echo "============================================"
