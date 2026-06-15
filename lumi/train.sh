#!/bin/bash
# =============================================================================
# Train a model on LUMI and save the resulting weights.
#
# Usage:
#   sbatch lumi/train.sh              # from repo root
#   sbatch -J usps lumi/train.sh      # from repo root, different dataset
#   cd lumi && sbatch train.sh        # or from inside lumi/
#
# How it works:
#   The job name (-J) selects the experiment. It is used to:
#     * load hyperparameters from   configs/<job-name>.cfg
#     * name the log files          lumi_logs/<job-name>_<job-id>.{out,err}
#   To add a dataset, drop a new configs/<name>.cfg next to the others.
# =============================================================================

# --- Resources (shared by all runs; not dataset-specific) --------------------
#SBATCH --account=project_465002757
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:30:00

# --- Experiment selection and logs -------------------------------------------
# Default dataset; override per run with `sbatch -J <dataset> train.sh`.
#SBATCH --job-name=mnist

# Always run from repo root regardless of where sbatch was invoked.
# Redirect logs to lumi/logs/ so the path is correct from any submission dir.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p lumi/logs
exec > "lumi/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "lumi/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

# --- Load the experiment config ----------------------------------------------
# The config is chosen by the job name (or CONFIG, if explicitly set).
CONFIG=${CONFIG:-$SLURM_JOB_NAME}
CONFIG_FILE="configs/${CONFIG}.cfg"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    echo "Available configs:" >&2
    ls configs/*.cfg 2>/dev/null >&2
    exit 1
fi

# Provides: dataset, epochs, batch_size, device
source "$CONFIG_FILE"

# --- Software environment ----------------------------------------------------
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif

# --- Run training ------------------------------------------------------------
set -xv
srun singularity run "$CONTAINER" \
    python -m src.training \
        --dataset "$dataset" \
        --epochs "$epochs" \
        --batch-size "$batch_size" \
        --device "$device"