#!/bin/bash
#SBATCH --job-name=render_cacheb
#SBATCH --account=bgte-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-8%9
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/render_cacheb_%A_%a.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/render_cacheb_%A_%a.err
set -euo pipefail

# Render COSMO rotation frames for the 138 cache-ILs missing from the ViT bank.
# Partitioned into 9 array tasks of ~16 compounds each (138 / 9 ≈ 15.3).
# Each task produces data/pipeline/cosmo_images/{CACHEB_NNN}_frames/frame_*.png
# (the single-conformer location, where build_il_vit_bank.py looks).

source /u/kahmed2/miniconda3/etc/profile.d/conda.sh
# mmseg env has both skimage (needed by step4 marching cubes) and rdkit.
# The default psi4 env is missing skimage and rejects the import.
conda activate mmseg

REPO=/work/nvme/bgte/kahmed2/Dataset_Chemistry
cd "$REPO"
export PYTHONUNBUFFERED=1

# The merged geometry_status.csv has 381 rows: 0..242 are the old ILs
# (already rendered) and 243..380 are the 138 new CACHEB rows that need
# rendering. Partition those 138 rows across 9 array tasks.
OLD_END=243
N_TOTAL=138
N_TASKS=9
CHUNK=$(( (N_TOTAL + N_TASKS - 1) / N_TASKS ))  # ceil
START=$(( OLD_END + SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))
ABS_END=$(( OLD_END + N_TOTAL ))
if [ $END -gt $ABS_END ]; then END=$ABS_END; fi

echo "============================================"
echo "render_cacheb array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "absolute slice [$START:$END]  Date: $(date)"
echo "============================================"

# Writes into data/pipeline/cosmo_images_multi/conf_0/{CACHEB_NNN}_frames/
# (matching existing convention so build_il_vit_bank.py picks them up with
# a trivial path tweak if needed).
python3 scripts/pipeline/step4_render_cosmo_images.py \
    --start $START --end $END --frames-only

echo "Done: $(date)"
