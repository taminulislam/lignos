#!/bin/bash
#SBATCH --job-name=chemprop_t1
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/chemprop_t1_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/chemprop_t1_%j.err
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "Chemprop D-MPNN literature baseline, Task 1 (LignoIL_A1 test split, n=39)"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
source /u/kahmed2/miniconda3/etc/profile.d/conda.sh
conda activate mmseg
echo "Python: $(which python3)   Chemprop: $(which chemprop_train)"
export PYTHONUNBUFFERED=1
python3 lignos/scripts/train_chemprop_task1.py \
    --n-seeds 10 --epochs 30 --batch-size 50
echo "Exit: $?  Done: $(date)"
