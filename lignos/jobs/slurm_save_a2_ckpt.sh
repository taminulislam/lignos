#!/bin/bash
#SBATCH --job-name=a2_ckpt
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a2_ckpt_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a2_ckpt_%j.err
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "Saving A2 Stage-1 checkpoint (best of 3 seeds by val loss)..."
export PYTHONUNBUFFERED=1
python3 lignos/scripts/save_a2_stage1_ckpt.py --n-seeds 3
echo "Done: $(date)  Exit: $?"
