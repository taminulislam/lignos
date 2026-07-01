#!/bin/bash
#SBATCH --job-name=raca
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --exclude=gpua039
#SBATCH --output=../jobs/raca_%j.out
#SBATCH --error=../jobs/raca_%j.err

source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry
echo "Date: $(date) Node: $(hostname)"
python scripts/raca.py
echo "Done: $(date)"
