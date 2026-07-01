#!/bin/bash
#SBATCH --job-name=psi4retry
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4,gpuA40x4-preempt,gpuA100x4-preempt
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-18
#SBATCH --requeue
#SBATCH --output=../jobs/logs/psi4_retry_%A_%a.out
#SBATCH --error=../jobs/logs/psi4_retry_%A_%a.err

# ============================================================
# Retry array for the 18 compounds missed by 17496966/67.
# Same payload as slurm_psi4_cosmo.sh, but indexed against
# /tmp/missing_compounds.txt.
# ============================================================

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
PSI4_PY="/u/kahmed2/miniconda3/envs/psi4/bin/python"
SCRIPT="${PROJECT_ROOT}/scripts/pipeline/step3_psi4_cosmo.py"
COMPOUND_LIST="${PROJECT_ROOT}/data/pipeline/missing_compounds.txt"
OUTPUT_DIR="${PROJECT_ROOT}/data/pipeline/dft_surface"

mkdir -p "${OUTPUT_DIR}"

CID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${COMPOUND_LIST}")
if [ -z "$CID" ]; then
    echo "ERROR: No compound for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

if [ -f "${OUTPUT_DIR}/${CID}.npz" ]; then
    echo "SKIP: ${CID} already computed"
    exit 0
fi

echo "============================================"
echo "Psi4 CPCM (retry): ${CID}"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================"

export PSI_SCRATCH="/tmp/psi4_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${PSI_SCRATCH}"
trap 'rm -rf ${PSI_SCRATCH}' EXIT
cd "${PSI_SCRATCH}"

timeout 7200 "${PSI4_PY}" "${SCRIPT}" \
    --compound-id "${CID}" \
    --geom-dir "${PROJECT_ROOT}/data/pipeline/geometries" \
    --out-dir "${OUTPUT_DIR}"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: ${CID}"
else
    echo "FAILED: ${CID} exit code ${EXIT_CODE}"
fi

echo "Done: $(date)"
