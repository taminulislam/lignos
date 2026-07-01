#!/bin/bash
#SBATCH --job-name=merged_v2
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/merged_v2_%j.out
#SBATCH --error=../jobs/merged_v2_%j.err

# ============================================================
# Merged Data Experiments (v2: proper feature alignment)
# Runs: baseline (223 samples) + path2 (same 28 ILs, ~983 samples)
# ============================================================

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "Merged Data Experiments (v2)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Run baseline + path2 (path1 skipped until real Chemprop features available)
python "${V5_ROOT}/scripts/train_merged_v2.py" \
    --mode baseline \
    --seeds 0-9

python "${V5_ROOT}/scripts/train_merged_v2.py" \
    --mode path2 \
    --seeds 0-9

echo ""
echo "Done: $(date)"
