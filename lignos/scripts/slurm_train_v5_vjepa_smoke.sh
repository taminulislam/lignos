#!/bin/bash
#SBATCH --job-name=vjepa_smoke
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/train_vjepa_smoke_%j.out
#SBATCH --error=../jobs/train_vjepa_smoke_%j.err

# 1-seed validation: does the DFT V-JEPA + remapped images hit sane val R²?

source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "Date: $(date) Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

python "${V5_ROOT}/scripts/train_v5.py" --config configs/v5_vjepa.yaml --seeds 0-0
echo "Done: $(date)"
