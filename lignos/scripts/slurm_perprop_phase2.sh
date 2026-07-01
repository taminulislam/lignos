#!/bin/bash
#SBATCH --job-name=pp_phase2
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/pp_phase2_%j.out
#SBATCH --error=../jobs/pp_phase2_%j.err
# Phase #2: hybrid V-JEPA PerPropHead on top of the stronger
# (v4 blend + GBT residual) base. Target = beat 0.8320.

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"
python scripts/perprop_residual.py \
    --hybrid-vjepa --hybrid-pca-each 20 --sup-pca 20 \
    --base-source stronger \
    --seeds 10 --epochs 300 --tag phase2_hybrid_G20_D20_S20_strongerbase
echo "Done: $(date)"
