#!/bin/bash
#SBATCH --job-name=simclr_v5
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/simclr_v5_%j.out
#SBATCH --error=../jobs/simclr_v5_%j.err

# ============================================================
# SimCLR Self-Supervised Pre-training for COSMOBridge v5
# Pre-trains ViT-Tiny on 301 molecules x 36 views = 10,836 images
# ============================================================

# Load modules
module load python/3.10 2>/dev/null || true

# Activate environment
# Activate conda environment
source /u/kahmed2/miniconda3/bin/activate mmseg

# Project paths
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "SimCLR Pre-training"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

python "${V5_ROOT}/scripts/pretrain_simclr.py" \
    --config configs/simclr.yaml \
    --image_dir "${V5_ROOT}/data/cosmo_images" \
    --extra_image_dirs "${PROJECT_ROOT}/data/pipeline/cosmo_images" \
    --output_dir "${V5_ROOT}/checkpoints/simclr" \
    --epochs 200 \
    --batch_size 256 \
    --lr 0.3

echo ""
echo "Done: $(date)"
