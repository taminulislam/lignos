#!/bin/bash
# Submit LOIO-CV as 4-chunk array + dependent aggregator.
# Fresh run: clears partial results so folds are recomputed.
set -euo pipefail
JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTIAL_DIR="${JOBS_DIR}/../results/loio_partial"
if [[ "${1:-}" == "--fresh" ]]; then
    rm -rf "${PARTIAL_DIR}"
    echo "Cleared ${PARTIAL_DIR}"
fi
ARRAY_JID=$(sbatch --parsable "${JOBS_DIR}/slurm_loio_cv.sh")
echo "Submitted array job:  ${ARRAY_JID} (4 chunks × ~7 folds)"
AGG_JID=$(sbatch --parsable --dependency=afterok:${ARRAY_JID} "${JOBS_DIR}/slurm_loio_cv_aggregate.sh")
echo "Submitted aggregator: ${AGG_JID} (runs after array completes)"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ${JOBS_DIR}/logs/loio_cv_${ARRAY_JID}_*.out"
echo "Result:   lignos/results/loio_cv_core7.json"
