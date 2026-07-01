#!/bin/bash
#SBATCH --job-name=v51_stg
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/v51_stg_%j.out
#SBATCH --error=../jobs/v51_stg_%j.err

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"
cd "${PROJECT_ROOT}"

echo "============================================"
echo "v5.1 STAGED: Pretrain paths -> Freeze -> Router"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

python "${V5_ROOT}/scripts/train_v5_1_staged.py" --seeds 0-9

echo "Done: $(date)"
