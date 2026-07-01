#!/bin/bash
#SBATCH --job-name=vres_A
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/vres_A_%j.out
#SBATCH --error=../jobs/vres_A_%j.err

# Suggestion A: V-JEPA + thermo features -> v4 residual corrector.

source /u/kahmed2/miniconda3/bin/activate mmseg
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"

echo "Date: $(date) Node: $(hostname)"
python scripts/vjepa_residual_corrector.py \
    --seeds 10 --epochs 300 \
    --features vjepa,thermo --tag A_vjepa_thermo
echo "Done: $(date)"
