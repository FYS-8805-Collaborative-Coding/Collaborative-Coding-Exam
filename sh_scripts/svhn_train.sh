#!/bin/bash
#SBATCH --account=project_465002757
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:30:00
#SBATCH --output=slurm-%j.out

DATASET=${DATASET:-svhn}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-64}

# Set up the software environment.
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Prebuilt LAIF PyTorch container (ships torch + torchvision, ROCm-enabled).
CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif

# Cache directories on fast parallel storage instead of $HOME.
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
export TORCH_HOME=$SCRATCH/torch-cache
mkdir -p $TORCH_HOME

set -xv  # echo the command so it is visible in the slurm log
srun singularity run $CONTAINER \
    python -m src.training \
        --dataset "$DATASET" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --data-dir datasets \
        --num-workers "${SLURM_CPUS_PER_TASK}"
