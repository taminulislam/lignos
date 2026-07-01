#!/bin/bash
#SBATCH --job-name=vjepa_v5
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/vjepa_v5_%j.out
#SBATCH --error=../jobs/vjepa_v5_%j.err

# ============================================================
# V-JEPA Self-Supervised Pre-training for COSMO Surfaces
# Alternative to SimCLR: predicts masked rotation views in latent space
# ============================================================

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "V-JEPA Pre-training"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

python "${V5_ROOT}/scripts/pretrain_vjepa.py" \
    --epochs 200 \
    --batch_size 32 \
    --lr 1.5e-4 \
    --n_views 36 \
    --mask_ratio_min 0.6 \
    --mask_ratio_max 0.8 \
    --embed_dim 192 \
    --predictor_dim 96 \
    --ema_decay 0.996 \
    --output_dir "${V5_ROOT}/checkpoints/vjepa"

echo ""
echo "Done: $(date)"
