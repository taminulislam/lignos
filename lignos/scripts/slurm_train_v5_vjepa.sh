#!/bin/bash
#SBATCH --job-name=train_vjepa
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/train_vjepa_%j.out
#SBATCH --error=../jobs/train_vjepa_%j.err

# ============================================================
# COSMOBridge v5: Supervised Training with V-JEPA Pre-trained ViT
# Head-to-head comparison with SimCLR pre-training
# ============================================================

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "COSMOBridge v5: V-JEPA Supervised Training"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Verify V-JEPA checkpoint exists
if [ ! -f "${V5_ROOT}/checkpoints/vjepa/vit_pretrained_vjepa.pt" ]; then
    echo "ERROR: V-JEPA checkpoint not found!"
    echo "Expected: ${V5_ROOT}/checkpoints/vjepa/vit_pretrained_vjepa.pt"
    echo "Run pretrain_vjepa.py first."
    exit 1
fi

echo "V-JEPA checkpoint found: $(ls -lh ${V5_ROOT}/checkpoints/vjepa/vit_pretrained_vjepa.pt)"

# Train all 10 seeds with V-JEPA pre-trained ViT
python "${V5_ROOT}/scripts/train_v5.py" \
    --config configs/v5_vjepa.yaml \
    --seeds 0-9

# Run ensemble evaluation
echo ""
echo "Running ensemble evaluation (V-JEPA)..."
python "${V5_ROOT}/scripts/ensemble_eval.py" \
    --predictions_dir "${V5_ROOT}/results/vjepa/seed_predictions" \
    --output "${V5_ROOT}/results/vjepa/ensemble_metrics.json"

echo ""
echo "Done: $(date)"
