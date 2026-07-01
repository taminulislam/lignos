#!/bin/bash
#SBATCH --job-name=cp_pre_t1
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/cp_pre_t1_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/cp_pre_t1_%j.err
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "Chemprop pretrained + fine-tuned, Task 1"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
source /u/kahmed2/miniconda3/etc/profile.d/conda.sh
conda activate mmseg
export PYTHONUNBUFFERED=1
python3 lignos/scripts/train_chemprop_pretrained.py \
    --task task1 --n-seeds 10 --epochs 30 --batch-size 50 \
    --init-lr 1e-5 --max-lr 1e-4 --final-lr 1e-5
echo "Exit: $?  Done: $(date)"
