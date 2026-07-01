#!/bin/bash
#SBATCH --job-name=pp_swap
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/pp_swap_%j.out
#SBATCH --error=../jobs/pp_swap_%j.err
# PerPropHead Exp 1: Swap-only — DFT V-JEPA into the archived 0.831 recipe.

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"
python scripts/perprop_residual.py \
    --vjepa-source dft --vjepa-pca 20 --sup-pca 20 \
    --seeds 10 --epochs 300 --tag swap_only
echo "Done: $(date)"
