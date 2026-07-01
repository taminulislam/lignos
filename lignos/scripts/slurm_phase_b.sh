#!/bin/bash
#SBATCH --job-name=phase_b
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/phase_b_%j.out
#SBATCH --error=../jobs/phase_b_%j.err

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry

echo "PHASE B: DAPT on 70 ILs + V-JEPA retraining + v4 pipeline | $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

python lignos/scripts/train_phase_b.py --seeds 0-9

echo "Done: $(date)"
