#!/bin/bash
#SBATCH --job-name=cosmo_v5_gen
#SBATCH --account=bgte-delta-gpu
#SBATCH --array=1-99%50
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --output=../jobs/cosmo_v5_gen_%A_%a.out
#SBATCH --error=../jobs/cosmo_v5_gen_%A_%a.err

# ============================================================
# COSMO Image Generation Pipeline for iThermo ILs
# COSMOBridge v5 - Phase 3
# ============================================================
#
# Prerequisites:
#   1. Create missing_compounds.txt:
#      python -c "
#      import pandas as pd
#      from pathlib import Path
#      df = pd.read_csv('../../data/pipeline/ilthermo_compounds.csv')
#      existing = {p.stem for p in Path('../../data/pipeline/point_clouds').glob('*.npz')}
#      missing = df[~df['compound_id'].isin(existing)]
#      missing['compound_id'].to_csv('missing_compounds.txt', index=False, header=False)
#      print(f'{len(missing)} compounds to process')
#      "
#
#   2. Update --array above to match line count of missing_compounds.txt
#
#   3. Submit: sbatch slurm_cosmo_generation.sh
# ============================================================

# Load modules
module load nwchem/7.2 2>/dev/null || true
module load python/3.10 2>/dev/null || true

# Activate environment
# Activate conda environment
source /u/kahmed2/miniconda3/bin/activate mmseg

# Project paths
PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"
PIPELINE_DIR="${PROJECT_ROOT}/data/pipeline"

# Get compound ID for this array task
COMPOUND_LIST="${V5_ROOT}/scripts/missing_compounds.txt"
COMPOUND_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${COMPOUND_LIST}")

if [ -z "$COMPOUND_ID" ]; then
    echo "ERROR: No compound found for task ${SLURM_ARRAY_TASK_ID}"
    exit 1

echo "============================================"
echo "Processing: ${COMPOUND_ID}"
echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================"

cd "${PROJECT_ROOT}"

# Step 1: Geometry optimization (GFN2-xTB)
echo "[1/6] Geometry optimization..."
if [ -f "${PIPELINE_DIR}/geometries/${COMPOUND_ID}.xyz" ]; then
    echo "  Already exists, skipping."
else
    python scripts/pipeline/step2_geometry_optimization.py \
        --compound_id "${COMPOUND_ID}" 2>&1 || echo "  WARNING: geom opt failed"

# Step 2: DFT + COSMO calculation (NWChem)
echo "[2/6] DFT + COSMO calculation..."
if [ -f "${PIPELINE_DIR}/dft_output/${COMPOUND_ID}.cosmo" ]; then
    echo "  Already exists, skipping."
else
    timeout 14400 python scripts/pipeline/step3_dft_esp.py \
        --compound_id "${COMPOUND_ID}" 2>&1 || echo "  WARNING: DFT failed (timeout or error)"

# Step 3: Extract point cloud
echo "[3/6] Extracting point cloud..."
if [ -f "${PIPELINE_DIR}/point_clouds/${COMPOUND_ID}.npz" ]; then
    echo "  Already exists, skipping."
else
    python scripts/pipeline/step4_extract_pointcloud.py \
        --compound_id "${COMPOUND_ID}" 2>&1 || echo "  WARNING: point cloud extraction failed"

# Step 4: Render 36-view COSMO images
echo "[4/6] Rendering COSMO views..."
python "${V5_ROOT}/scripts/render_cosmo_views.py" \
    --compound_id "${COMPOUND_ID}" \
    --n_views 36 \
    --resolution 224 \
    --render_ep \
    --output_dir "${V5_ROOT}/data/cosmo_images" 2>&1 || echo "  WARNING: rendering failed"

# Step 5: Generate sigma-surface unfolding
echo "[5/6] Sigma-surface unfolding..."
python "${V5_ROOT}/scripts/unfold_surface.py" \
    --compound_id "${COMPOUND_ID}" \
    --output_dir "${V5_ROOT}/data/sigma_maps" 2>&1 || echo "  WARNING: unfolding failed"

# Step 6: Render cation/anion images
echo "[6/6] Rendering ion images..."
python "${V5_ROOT}/scripts/render_ion_images.py" \
    --compound_id "${COMPOUND_ID}" \
    --output_dir "${V5_ROOT}/data/ion_images" 2>&1 || echo "  WARNING: ion rendering failed"

echo ""
echo "Done: ${COMPOUND_ID}"
echo "Finished: $(date)"
