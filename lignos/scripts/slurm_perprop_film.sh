#!/bin/bash
#SBATCH --job-name=pp_film
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/pp_film_%j.out
#SBATCH --error=../jobs/pp_film_%j.err
# Idea α: T-conditioned FiLM projection replaces PCA on V-JEPA streams.

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"
python scripts/perprop_advanced.py --mode film --seeds 10 --epochs 300 --tag film
echo "Done: $(date)"
