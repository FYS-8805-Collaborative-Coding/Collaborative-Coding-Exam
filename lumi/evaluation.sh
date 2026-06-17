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

# --- Experiment selection and logs -------------------------------------------
# Default dataset; override per run with `sbatch -J eval-<dataset> evaluation.sh`.
#SBATCH --job-name=eval-mnist
#SBATCH --output=lumi/logs/%x_%j.out
#SBATCH --error=lumi/logs/%x_%j.err

# Always run from repo root regardless of where sbatch was invoked.
# BASH_SOURCE is unreliable under SLURM (script is copied to /var/spool/...);
# use SLURM_SUBMIT_DIR instead. Support both submission forms:
#   sbatch lumi/evaluation.sh        (SLURM_SUBMIT_DIR = repo root)
#   cd lumi && sbatch evaluation.sh  (SLURM_SUBMIT_DIR = lumi/)
if [[ -d "${SLURM_SUBMIT_DIR}/configs" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "${SLURM_SUBMIT_DIR}/.."
fi

set -euo pipefail
set -x

DATASET="${SLURM_JOB_NAME#eval-}"
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
