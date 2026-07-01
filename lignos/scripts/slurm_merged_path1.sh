#!/bin/bash
#SBATCH --job-name=path1
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/path1_%j.out
#SBATCH --error=../jobs/path1_%j.err

# ============================================================
# Path 1: All 143 iThermo ILs with real Chemprop features
# ============================================================

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "Path 1: All iThermo ILs (real Chemprop features)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================"

python "${V5_ROOT}/scripts/train_merged_v2.py" \
    --mode path1 \
    --seeds 0-9 \
    --precomputed "${V5_ROOT}/data/precomputed_chemprop_features.npz"

echo ""
echo "Done: $(date)"
