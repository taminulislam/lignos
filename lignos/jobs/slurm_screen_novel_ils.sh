#!/bin/bash
#SBATCH --job-name=screen_ils
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/screen_ils_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/screen_ils_%j.err
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "Virtual IL screening — A5.9 Stage-2 inference on 3969-IL catalog"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
export PYTHONUNBUFFERED=1
python3 lignos/scripts/screen/screen_novel_ils.py --top-k 100 --confidence-quantile 0.5
echo "Exit: $?  Done: $(date)"
