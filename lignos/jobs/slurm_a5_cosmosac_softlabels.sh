#!/bin/bash
#SBATCH --job-name=a5_cos_sl
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_cos_sl_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_cos_sl_%j.err
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "A5.4 — COSMO-SAC γ soft labels on unlabeled ILThermo rows"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
export PYTHONUNBUFFERED=1
LAMBDA_AUX=${LAMBDA_AUX:-0.01}
python3 lignos/scripts/train_a5_cosmosac_softlabels.py --n-seeds 10 --lambda-aux ${LAMBDA_AUX}
echo "Exit: $?  Done: $(date)"
