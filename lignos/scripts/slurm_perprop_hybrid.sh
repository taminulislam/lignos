#!/bin/bash
#SBATCH --job-name=pp_hybrid
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/pp_hybrid_%j.out
#SBATCH --error=../jobs/pp_hybrid_%j.err
# Hybrid: PCA(Gasteiger V-JEPA, 10) + PCA(DFT V-JEPA, 10) + PCA(Supervised, 20) = 40D

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"
python scripts/perprop_residual.py \
    --hybrid-vjepa --hybrid-pca-each 10 --sup-pca 20 \
    --seeds 10 --epochs 300 --tag hybrid_G10_D10_S20
echo "Done: $(date)"
