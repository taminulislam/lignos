#!/bin/bash
#SBATCH --job-name=a2_plus_x
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a2_plus_x_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a2_plus_x_%j.err

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"

echo "============================================"
echo "A2 (Morgan+ChemProp gated residual) + X (G_mix>=0 hinge) on A1 cache"
echo "  4 arms x 10 seeds"
echo "Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Date: $(date)"
echo "============================================"

export PYTHONUNBUFFERED=1
python3 lignos/scripts/train_a2_plus_x.py

echo "============================================"
echo "Done: $(date)"
echo "Exit: $?"
echo "============================================"
