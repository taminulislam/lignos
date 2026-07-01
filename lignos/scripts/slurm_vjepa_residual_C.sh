#!/bin/bash
#SBATCH --job-name=vres_C
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/vres_C_%j.out
#SBATCH --error=../jobs/vres_C_%j.err
# Suggestion C: B features + skip P

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"
python scripts/vjepa_residual_corrector.py \
    --seeds 10 --epochs 300 \
    --features vjepa,chemprop,thermo \
    --skip-properties P \
    --tag C_vjepa_chemprop_thermo_skipP
echo "Done: $(date)"
