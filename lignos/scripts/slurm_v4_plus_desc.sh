#!/bin/bash
#SBATCH --job-name=v4_desc
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/v4_desc_%j.out
#SBATCH --error=../jobs/v4_desc_%j.err

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

cd /work/nvme/bgte/kahmed2/Dataset_Chemistry

echo "v4 + Descriptor Path | $(date) | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
python lignos/scripts/train_v4_plus_descriptors.py --seeds 0-9
echo "Done: $(date)"
