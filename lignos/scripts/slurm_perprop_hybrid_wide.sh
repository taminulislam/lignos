#!/bin/bash
#SBATCH --job-name=pp_hybwide
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/pp_hybwide_%j.out
#SBATCH --error=../jobs/pp_hybwide_%j.err
# Final hybrid: PCA(Gasteiger V-JEPA, 20) + PCA(DFT V-JEPA, 20) + PCA(Supervised, 20) = 60D

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"
python scripts/perprop_residual.py \
    --hybrid-vjepa --hybrid-pca-each 20 --sup-pca 20 \
    --seeds 10 --epochs 300 --tag hybrid_G20_D20_S20
echo "Done: $(date)"
