#!/bin/bash
#SBATCH --job-name=v5_lite
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/v5_lite_%j.out
#SBATCH --error=../jobs/v5_lite_%j.err

# ============================================================
# COSMOBridge v5-Lite: Merged Data + Frozen Encoders + Distillation
# Solutions 1+3+6 to combat overfitting
# ============================================================

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "COSMOBridge v5-Lite Training"
echo "Solutions: Merged data + Lite model + Distillation"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

python "${V5_ROOT}/scripts/train_v5_lite.py" \
    --seeds 0-9 \
    --distill_alpha 0.7

echo ""
echo "Running ensemble evaluation..."
python "${V5_ROOT}/scripts/ensemble_eval.py" \
    --predictions_dir "${V5_ROOT}/results/lite/seed_predictions" \
    --output "${V5_ROOT}/results/lite/ensemble_metrics.json"

echo ""
echo "Done: $(date)"
