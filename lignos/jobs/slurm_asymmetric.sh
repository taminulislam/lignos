#!/bin/bash
#SBATCH --job-name=asym_hd
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/asymmetric_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/asymmetric_%j.err
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
export PYTHONUNBUFFERED=1
echo "============================================"
echo "Asymmetric Heads: shallow core7 + deep lignin"
echo "Job: ${SLURM_JOB_ID}  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"
echo "============================================"
python3 lignos/scripts/train_asymmetric.py
echo "Done: $(date)"
