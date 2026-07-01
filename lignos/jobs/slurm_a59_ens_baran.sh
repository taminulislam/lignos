#!/bin/bash
#SBATCH --job-name=a59_ens_t2
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a59_ens_t2_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a59_ens_t2_%j.err

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "A5.9 Ensemble Baran Task 2 CV — total (aleatoric + epistemic) uncertainty gating"
echo "3 specialists × 2 seeds × 5 folds = 30 specialist trainings"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
export PYTHONUNBUFFERED=1
python3 lignos/scripts/compare_a59_ens_vs_baran.py --n-seeds 2 --n-splits 5
echo "Exit: $?  Done: $(date)"
