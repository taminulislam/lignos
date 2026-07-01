#!/bin/bash
#SBATCH --job-name=p1hybrid
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --exclude=gpua039
#SBATCH --output=../jobs/p1hybrid_%j.out
#SBATCH --error=../jobs/p1hybrid_%j.err
# Phase #1 Step G: Hybrid PerPropHead (G20+D20+S20) on the expanded
# 4637-sample training set, with NaN-masked multi-task MSE loss.

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"

python scripts/perprop_expanded.py --seeds 10 --epochs 300 --tag phase1_expanded_hybrid
echo "Done: $(date)"
