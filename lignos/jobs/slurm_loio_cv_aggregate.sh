#!/bin/bash
#SBATCH --job-name=loio_agg
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/loio_agg_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/loio_agg_%j.err
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
export PYTHONUNBUFFERED=1
echo "============================================"
echo "LOIO-CV Aggregation  Job: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "============================================"
python3 lignos/scripts/eval_loio_core7.py --aggregate
echo "Done: $(date)"
