#!/bin/bash
#SBATCH --job-name=full_pipe
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/full_pipe_%j.out
#SBATCH --error=../jobs/full_pipe_%j.err

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry

echo "FULL PIPELINE: v4 Router + Physics + Image Residual | $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

python lignos/scripts/train_full_pipeline.py --phase A --seeds 0-9

echo "Done: $(date)"
