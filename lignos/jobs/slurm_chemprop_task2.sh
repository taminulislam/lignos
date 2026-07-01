#!/bin/bash
#SBATCH --job-name=chemprop_t2
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-12
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/chemprop_t2_%A_%a.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/chemprop_t2_%A_%a.err
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "Chemprop D-MPNN literature baseline, 13-fold LoIoO Task 2 fold ${SLURM_ARRAY_TASK_ID} / 13"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Date: $(date)"
source /u/kahmed2/miniconda3/etc/profile.d/conda.sh
conda activate mmseg
echo "Python: $(which python3)   Chemprop: $(which chemprop_train)"
export PYTHONUNBUFFERED=1
python3 lignos/scripts/train_chemprop_task2.py \
    --fold ${SLURM_ARRAY_TASK_ID} \
    --n-splits 13 --n-seeds 5 --epochs 30 --batch-size 50
echo "Exit: $?  Done: $(date)"
