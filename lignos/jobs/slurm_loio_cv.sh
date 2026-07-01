#!/bin/bash
#SBATCH --job-name=loio_cv
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/loio_cv_%A_%a.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/loio_cv_%A_%a.err
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
export PYTHONUNBUFFERED=1
N_CHUNKS=4
echo "============================================"
echo "LOIO-CV Core 7  (array ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS})"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"
echo "============================================"
python3 lignos/scripts/eval_loio_core7.py \
    --chunk-idx ${SLURM_ARRAY_TASK_ID} \
    --n-chunks ${N_CHUNKS}
echo "Done: $(date)"
