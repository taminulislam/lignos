#!/bin/bash
#SBATCH --job-name=v4_img
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/v4_img_%j.out
#SBATCH --error=../jobs/v4_img_%j.err

# ============================================================
# v4 + Image: exact v4 model + images as 3rd frozen path
# THE definitive test: do images help on top of v4?
# ============================================================

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "v4 + Image Path"
echo "Exact v4 protocol + images as 3rd pathway"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

python "${V5_ROOT}/scripts/train_v4_plus_image.py" --seeds 0-9

echo ""
echo "Done: $(date)"
