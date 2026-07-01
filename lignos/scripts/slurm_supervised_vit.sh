#!/bin/bash
#SBATCH --job-name=sup_vit
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/sup_vit_%j.out
#SBATCH --error=../jobs/sup_vit_%j.err

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry

echo "MULTI-TASK SUPERVISED ViT | $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

python lignos/scripts/train_supervised_vit.py \
    --epochs 100 --seeds 0-9 \
    --alpha 1.0 --beta 0.5 --gamma_w 0.1

echo "Done: $(date)"
