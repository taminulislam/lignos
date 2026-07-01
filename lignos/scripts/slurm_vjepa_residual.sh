#!/bin/bash
#SBATCH --job-name=vjepa_res
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4-preempt,gpuA40x4-preempt,gpuA100x4,gpuA40x4
#SBATCH --requeue
#SBATCH --output=../jobs/vjepa_residual_%j.out
#SBATCH --error=../jobs/vjepa_residual_%j.err

# Idea 4: V-JEPA residual corrector for v4 predictions.
# Extracts CLS embeddings from DFT-pretrained V-JEPA, trains a small MLP
# to predict v4 residuals, evaluates 10-seed ensemble R² on test.

source /u/kahmed2/miniconda3/bin/activate mmseg
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
cd "${PROJECT_ROOT}"

echo "Date: $(date) Node: $(hostname)"
python scripts/vjepa_residual_corrector.py --seeds 10 --epochs 300
echo "Done: $(date)"
