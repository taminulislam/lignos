#!/bin/bash
#SBATCH --job-name=vit_exp
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/vit_exp_%x_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/vit_exp_%x_%j.err
# Usage: sbatch --job-name=<label> slurm_vit_experiment.sh --image-source X [--use-physchem]
# Args after the script name are forwarded to the python entry point.
set -euo pipefail

cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
export PYTHONUNBUFFERED=1

echo "============================================"
echo "ViT experiment  Job: ${SLURM_JOB_ID}  Name: ${SLURM_JOB_NAME}"
echo "Args: $*"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"
echo "============================================"

python3 lignos/scripts/train_two_stage_vit.py "$@"
echo "Done: $(date)"
