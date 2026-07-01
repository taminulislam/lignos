#!/bin/bash
#SBATCH --job-name=vres_E
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/vres_E_%j.out
#SBATCH --error=../jobs/vres_E_%j.err
# Suggestion E: kitchen-sink + skip P

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"
python scripts/vjepa_residual_corrector.py \
    --seeds 10 --epochs 300 \
    --features vjepa,chemprop,thermo,surface \
    --skip-properties P \
    --tag E_kitchen_sink_skipP
echo "Done: $(date)"
