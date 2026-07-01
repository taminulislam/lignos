#!/bin/bash
#SBATCH --job-name=pp_frame
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/pp_frame_%j.out
#SBATCH --error=../jobs/pp_frame_%j.err
# Idea β: Frame-level training — treat each of 36 rotation frames as a
# separate training sample (152×36=5472 pairs), infer via frame-average.

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"
python scripts/perprop_advanced.py --mode frame-level --seeds 10 --epochs 300 --tag frame_level
echo "Done: $(date)"
