#!/bin/bash
#SBATCH --job-name=a4_uw
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a4_uw_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a4_uw_%j.err
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
export PYTHONUNBUFFERED=1
echo "=== A4: Kendall uncertainty-weighted multi-task loss  Job ${SLURM_JOB_ID}  GPU $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="
echo "Date: $(date)"
python3 lignos/scripts/train_a4_uncertainty.py
echo "Done: $(date)"
