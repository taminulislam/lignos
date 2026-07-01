#!/bin/bash
#SBATCH --job-name=ligno_dft
#SBATCH --account=bgte-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-98%16
#SBATCH --output=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/ligno_dft_%A_%a.out
#SBATCH --error=/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs/ligno_dft_%A_%a.err
set -euo pipefail

# Activate the conda env that has psi4 installed.
source /u/kahmed2/miniconda3/etc/profile.d/conda.sh
conda activate psi4

REPO=/work/nvme/bgte/kahmed2/Dataset_Chemistry
export PYTHONUNBUFFERED=1
export PSI4_MEM_GB=16
export PSI4_SCRATCH=/tmp/psi4_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$PSI4_SCRATCH"
export PSI_SCRATCH="$PSI4_SCRATCH"
# PCM writes a parsed file ("@pcmsolver.inp"-style) into the current working
# directory; multiple concurrent tasks sharing cwd corrupt each other's zips.
# Give every task its own cwd under PSI4_SCRATCH.
cd "$PSI4_SCRATCH"

TASK_LIST=/work/nvme/bgte/kahmed2/Dataset_Chemistry/data/pipeline/lignoil_new_dft_task_list.txt
TASK_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST")

echo "============================================"
echo "LignoIL DFT  array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Task: ${TASK_ID}"
echo "Date: $(date)"
echo "============================================"

python3 "$REPO/scripts/pipeline/step3_psi4_cosmo.py" \
    --compound-id "${TASK_ID}" \
    --geom-dir "$REPO/data/pipeline/geometries" \
    --out-dir "$REPO/data/pipeline/dft_surface"

# Clean up scratch to avoid filling /tmp
rm -rf "$PSI4_SCRATCH"
echo "Done: $(date)"
