#!/bin/bash
# =============================================================================
# Evaluate a trained model on LUMI.
#
# Usage (job name selects the dataset):
#   sbatch lumi/evaluation.sh                  # default: mnist
#   sbatch -J mnist lumi/evaluation.sh         # evaluate on mnist
#   sbatch -J usps lumi/evaluation.sh          # evaluate on usps
#   sbatch -J svhn lumi/evaluation.sh          # evaluate on svhn
#   cd lumi && sbatch -J usps evaluation.sh    # or from inside lumi/
# =============================================================================

# --- Resources ---------------------------------------------------------------
#SBATCH --account=project_465002757
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00

# --- Experiment selection ----------------------------------------------------
# Default dataset; override per run with `sbatch -J eval-<dataset> evaluation.sh`.
#SBATCH --job-name=mnist

# Always run from repo root regardless of where sbatch was invoked.
# Redirect logs to lumi/logs/ so the path is correct from any submission dir.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p lumi/logs
exec > "lumi/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "lumi/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

set -euo pipefail
set -x

DATASET="${SLURM_JOB_NAME}"
CHECKPOINT="weights/${DATASET}.pth"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    echo "Available checkpoints:" >&2
    ls weights/*.pth 2>/dev/null >&2
    exit 1
fi

CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif

singularity run "$CONTAINER" \
    python -m src.evaluation \
        --dataset "$DATASET" \
        --checkpoint-path "$CHECKPOINT" \
        --device cuda
