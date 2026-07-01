#!/bin/bash
#SBATCH --job-name=v5_1
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/v5_1_%j.out
#SBATCH --error=../jobs/v5_1_%j.err

# ============================================================
# COSMOBridge v5.1: Better Fusion + Descriptors + DAPT
# Tests: original (152 samples) and DAPT (143 ILs → fine-tune)
# ============================================================

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "COSMOBridge v5.1"
echo "Better Fusion + Descriptor Pathway + DAPT"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

python "${V5_ROOT}/scripts/train_v5_1.py" --mode both --seeds 0-9

echo ""
echo "Done: $(date)"
