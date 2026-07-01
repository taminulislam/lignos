#!/bin/bash
#SBATCH --job-name=v4_merged
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/v4_merged_%j.out
#SBATCH --error=../jobs/v4_merged_%j.err

# ============================================================
# v4 + Merged iThermo Data (baseline for data expansion effect)
# No images -- measures improvement from data merging alone
# ============================================================

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "v4 + Merged iThermo Data"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

python "${V5_ROOT}/scripts/train_v4_merged.py" \
    --seeds 0-9

echo ""
echo "Done: $(date)"
