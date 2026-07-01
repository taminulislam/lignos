#!/bin/bash
#SBATCH --job-name=dapt_v5
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/dapt_v5_%j.out
#SBATCH --error=../jobs/dapt_v5_%j.err

# ============================================================
# DAPT: Domain-Adaptive Pre-training
# Phase 1: ALL iThermo (broad domain) -> Phase 2: Original 28 ILs (task)
# Cleaner than STILT: no masking, no oversampling, all data used
# ============================================================

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "DAPT: Domain-Adaptive Pre-training"
echo "Phase 1: ALL iThermo -> Phase 2: Original 28 ILs"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

python "${V5_ROOT}/scripts/train_dapt.py" --seeds 0-9

echo ""
echo "Done: $(date)"
