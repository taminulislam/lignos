#!/bin/bash
#SBATCH --job-name=vjepa_s5
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/train_vjepa_seed5_%j.out
#SBATCH --error=../jobs/train_vjepa_seed5_%j.err

source /u/kahmed2/miniconda3/bin/activate mmseg
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"
cd "${PROJECT_ROOT}"
echo "Date: $(date) Node: $(hostname)"
python "${V5_ROOT}/scripts/train_v5.py" --config configs/v5_vjepa.yaml --seeds 5-5
echo "Done: $(date)"
