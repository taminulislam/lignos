#!/bin/bash
#SBATCH --job-name=ablation_v5
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/ablation_v5_%j.out
#SBATCH --error=../jobs/ablation_v5_%j.err

# ============================================================
# COSMOBridge v5: Ablation Study (6 experiments x 10 seeds)
# ============================================================

module load python/3.10 2>/dev/null || true

# Activate conda environment
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "COSMOBridge v5: Ablation Study"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

python "${V5_ROOT}/scripts/ablation_study.py" \
    --config configs/ablation.yaml \
    --base_config configs/v5_full.yaml \
    --seeds 0-9

echo ""
echo "Done: $(date)"
