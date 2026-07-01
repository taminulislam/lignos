#!/bin/bash
#SBATCH --job-name=rendft
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4,gpuA40x4-preempt,gpuA100x4-preempt
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --requeue
#SBATCH --output=../jobs/logs/rendft_%A_%a.out
#SBATCH --error=../jobs/logs/rendft_%A_%a.err

# ============================================================
# Re-render cosmo_images/ for V-JEPA using DFT ESP.
# 243 compounds (geometry_status.csv minus header) split across
# 24 shards of ~11 compounds each.
# ============================================================

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
PYTHON="/u/kahmed2/miniconda3/envs/mmseg/bin/python"
SCRIPT="${PROJECT_ROOT}/scripts/pipeline/step4_render_cosmo_images.py"

N_TOTAL=243
N_SHARDS=24
SHARD_SIZE=$(( (N_TOTAL + N_SHARDS - 1) / N_SHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
END=$(( START + SHARD_SIZE ))
if [ $END -gt $N_TOTAL ]; then END=$N_TOTAL; fi

echo "============================================"
echo "DFT image render: shard ${SLURM_ARRAY_TASK_ID}"
echo "Indices [$START:$END)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================"

cd "${PROJECT_ROOT}"
"${PYTHON}" "${SCRIPT}" --start $START --end $END

echo "Done: $(date)"
