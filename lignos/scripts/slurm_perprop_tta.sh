#!/bin/bash
#SBATCH --job-name=pp_tta
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/pp_tta_%j.out
#SBATCH --error=../jobs/pp_tta_%j.err
# Idea ζ: Test-time augmentation — mean-pooled training (hybrid recipe),
# but inference averages predictions across 36 rotation frames.

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"
python scripts/perprop_advanced.py --mode tta --seeds 10 --epochs 300 --tag tta
echo "Done: $(date)"
