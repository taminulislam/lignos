#!/bin/bash
#SBATCH --job-name=pp_surf
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/pp_surf_%j.out
#SBATCH --error=../jobs/pp_surf_%j.err
# PerPropHead Exp 3: swap + PCA(DFT surface_fp, 20) third stream (60-D features).

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"
python scripts/perprop_residual.py \
    --vjepa-source dft --vjepa-pca 20 --sup-pca 20 \
    --add-surface --surface-pca 20 \
    --seeds 10 --epochs 300 --tag swap_plus_surface
echo "Done: $(date)"
