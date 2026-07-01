#!/bin/bash
#SBATCH --job-name=a5_sf
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_sf_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_sf_%j.err

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"

echo "============================================"
echo "A5 = A2 + Surface(256D) + Frame(192D) zero-init gated residual branches"
echo "Cache: DFT-filled surface_fp (91%) + per-IL ViT bank (100% train coverage)"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
echo "============================================"

export PYTHONUNBUFFERED=1
python3 lignos/scripts/train_a5_surface_frames.py --n-seeds 10

echo "Exit: $?  Done: $(date)"
