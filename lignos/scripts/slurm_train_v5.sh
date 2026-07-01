#!/bin/bash
#SBATCH --job-name=train_v5
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/train_v5_%j.out
#SBATCH --error=../jobs/train_v5_%j.err

# ============================================================
# COSMOBridge v5: 3-Stage Supervised Training (10 Seeds)
# ============================================================

module load python/3.10 2>/dev/null || true

# Activate conda environment
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "COSMOBridge v5: Supervised Training"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Train all 10 seeds
python "${V5_ROOT}/scripts/train_v5.py" \
    --config configs/v5_full.yaml \
    --seeds 0-9

# Run ensemble evaluation
echo ""
echo "Running ensemble evaluation..."
python "${V5_ROOT}/scripts/ensemble_eval.py" \
    --predictions_dir "${V5_ROOT}/results/seed_predictions" \
    --output "${V5_ROOT}/results/ensemble_metrics.json"

echo ""
echo "Done: $(date)"
