#!/bin/bash
#SBATCH --account=project_465002757
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:30:00
#SBATCH --output=lumi_logs/mnist_train_%j.out
#SBATCH --error=lumi_logs/mnist_train_%j.err

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif

set -xv
srun singularity run $CONTAINER \
    python -m src.training --dataset mnist --epochs 5 --batch-size 32 --device cuda
