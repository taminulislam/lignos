#!/bin/bash
#SBATCH --job-name=bma4_mahal
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --array=0-4
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/bma4_mahal_%A_%a.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/bma4_mahal_%A_%a.err
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "LIGNOS BMA-K4 (Pick #1) + Mahalanobis gate (Pick #3) Task 2, fold ${SLURM_ARRAY_TASK_ID} / 5"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Date: $(date)"
export PYTHONUNBUFFERED=1
python3 lignos/scripts/compare_a59_bma4_mahal_baran.py \
    --fold ${SLURM_ARRAY_TASK_ID} \
    --n-specialist-seeds 2 --n-splits 5 --epochs 200 \
    --router-epochs 120 --mahal-q 0.9
echo "Exit: $?  Done: $(date)"
