#!/bin/bash
#SBATCH --job-name=pp_repro
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/pp_repro_%j.out
#SBATCH --error=../jobs/pp_repro_%j.err
# Reproduction check: run the archived 0.831 recipe with the archived
# Gasteiger V-JEPA features inside perprop_residual.py. If this produces
# R² ≈ 0.831, the harness is faithful and our DFT swap comparison is clean.

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"
python scripts/perprop_residual.py \
    --vjepa-source gasteiger --vjepa-pca 20 --sup-pca 20 \
    --seeds 10 --epochs 300 --tag gasteiger_repro
echo "Done: $(date)"
