#!/bin/bash
#SBATCH --job-name=a5_tier1
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_tier1_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_tier1_%j.err
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "A5.9 Tier 1 ablation — C-ensemble (10 seeds) + drop-A + isotonic + gated@{50,25,10}"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
export PYTHONUNBUFFERED=1
python3 lignos/scripts/train_a5_bma_tier1.py --n-seeds-c 10 --epochs 300
echo "Exit: $?  Done: $(date)"
