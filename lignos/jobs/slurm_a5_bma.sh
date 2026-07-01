#!/bin/bash
#SBATCH --job-name=a5_bma
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_bma_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_bma_%j.err

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "A5.9 — BMA with V4-anchored router (3 specialists + learned residual on inverse-variance weights)"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
export PYTHONUNBUFFERED=1
python3 lignos/scripts/train_a5_bma_router.py --n-seeds 10
echo "Exit: $?  Done: $(date)"
