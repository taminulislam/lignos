#!/bin/bash
#SBATCH --job-name=a5_cosmo
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_cosmo_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_cosmo_%j.err

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "============================================"
echo "A5.3 — A2 + zero-init COSMO-SAC σ-profile residual branch (20D per IL)"
echo "Coverage: 97.9% train rows via DFT-derived σ moments"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
echo "============================================"
export PYTHONUNBUFFERED=1
python3 lignos/scripts/train_a5_cosmo.py --n-seeds 10
echo "Exit: $?  Done: $(date)"
