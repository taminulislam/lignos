#!/bin/bash
#SBATCH --job-name=p1vjepa
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --exclude=gpua039
#SBATCH --output=../jobs/p1vjepa_%j.out
#SBATCH --error=../jobs/p1vjepa_%j.err
# Phase #1 step F.2: extract Gasteiger+DFT V-JEPA CLS embeddings
# for all 157 unique SMILES in the expanded train/val/test splits.
# GPU version; CPU version took 3+ h per encoder.

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

python scripts/phase1_cache_vjepa_expanded.py
echo "Done: $(date)"
