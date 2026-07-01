#!/bin/bash
#SBATCH --job-name=full_pb
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/full_pb_%j.out
#SBATCH --error=../jobs/full_pb_%j.err

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"
cd "${PROJECT_ROOT}"

echo "FULL PHASE B: V-JEPA on 70 ILs + DAPT + v4 Pipeline + Image | $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# Step 4: V-JEPA pre-train on ALL 70 ILs (original + new)
echo ""
echo "=== STEP 4: V-JEPA Pre-training on 70 ILs ==="
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
    --output_dir "${V5_ROOT}/checkpoints/vjepa_70il"

# Step 5-7: DAPT + v4 routing + image residual with new V-JEPA features
echo ""
echo "=== STEPS 5-7: DAPT + Routing + Image Residual ==="
python "${V5_ROOT}/scripts/train_phase_b_complete.py" \
    --vjepa_checkpoint "${V5_ROOT}/checkpoints/vjepa_70il/vit_pretrained_vjepa.pt" \
    --seeds 0-9

echo ""
echo "Done: $(date)"
