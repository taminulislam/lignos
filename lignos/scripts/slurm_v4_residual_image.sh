#!/bin/bash
#SBATCH --job-name=v4_resimg
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/v4_resimg_%j.out
#SBATCH --error=../jobs/v4_resimg_%j.err

# ============================================================
# v4 + Residual Image: images predict v4's errors
# Solutions 1 (residual) + 4 (PCA) + 9 (T-conditioning)
# ============================================================

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "v4 + Residual Image (Solutions 1+4+9)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

python "${V5_ROOT}/scripts/train_v4_residual_image.py" --seeds 0-9 --pca_dim 20

echo ""
echo "Done: $(date)"
