#!/bin/bash
#SBATCH --job-name=t1_rowpreds
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/t1_rowpreds_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/t1_rowpreds_%j.err
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "Task-1 row-level bootstrap: re-run LIGNOS +#5+#6 10 seeds, save per-row lignin preds"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
source /u/kahmed2/miniconda3/etc/profile.d/conda.sh
conda activate mmseg
export PYTHONUNBUFFERED=1
python3 lignos/scripts/train_a5_bma_tier2.py \
    --configs tier2_mu1_aug1 \
    --n-seeds 10 --epochs 300 \
    --save-rowpreds lignos/results/task1_tier2_mu1_aug1_rowpreds.npz
echo "Exit: $?  Done: $(date)"
