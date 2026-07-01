#!/bin/bash
#SBATCH --job-name=a5_mm
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_mm_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_mm_%j.err

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "A5.${VARIANT:-6} — Multimodal uncertainty-weighted fusion (SMILES+ViT+Surface+COSMO-SAC)"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
export PYTHONUNBUFFERED=1
EXTRA_FLAGS=${EXTRA_FLAGS:-}
echo "flags=${EXTRA_FLAGS}"
python3 lignos/scripts/train_a5_multimodal_unc.py --n-seeds 10 ${EXTRA_FLAGS}
echo "Exit: $?  Done: $(date)"
