#!/bin/bash
#SBATCH --job-name=il_vit_bank
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/il_vit_bank_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/il_vit_bank_%j.err

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "============================================"
echo "W2: per-IL ViT feature bank"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
echo "============================================"
export PYTHONUNBUFFERED=1
python3 lignos/scripts/build_il_vit_bank.py
echo "Exit: $?  Done: $(date)"
