#!/bin/bash
#SBATCH --job-name=track1b
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/track1b_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/track1b_%j.err
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
export PYTHONUNBUFFERED=1
echo "=== Track 1b: restricted-to-covered  Job ${SLURM_JOB_ID}  GPU $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="
echo "Date: $(date)"
python3 lignos/scripts/train_track1b_restricted.py
echo "Done: $(date)"
