#!/bin/bash
#SBATCH --job-name=a5_cosaux_sweep
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_cosaux_%j.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/a5_cosaux_%j.err

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"
echo "============================================"
echo "Plan B — A2 + σ-profile RECONSTRUCTION aux loss (λ=0.05)"
echo "Aux task: model's gated Morgan+thermo rep must reconstruct 20-D σ profile"
echo "Active on 97.9% of train rows (includes 5147 unlabeled ILThermo rows!)"
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
echo "============================================"
export PYTHONUNBUFFERED=1
LAMBDA_AUX=${LAMBDA_AUX:-0.05}
EXTRA_FLAGS=${EXTRA_FLAGS:-}
echo "Running with lambda_aux=${LAMBDA_AUX}  flags=${EXTRA_FLAGS}"
python3 lignos/scripts/train_a5_cosmo_aux.py --n-seeds 10 --lambda-aux ${LAMBDA_AUX} ${EXTRA_FLAGS}
echo "Exit: $?  Done: $(date)"
