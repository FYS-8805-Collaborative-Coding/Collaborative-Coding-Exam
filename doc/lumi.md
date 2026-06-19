# Running on LUMI

[LUMI](https://csc.fi/en/our-expertise/high-performance-computing/lumi-supercomputer/) is a EuroHPC supercomputer hosted at CSC's data center in Kajaani, Finland. It is owned by the European Commission's EuroHPC Joint Undertaking and operated by a consortium of 11 European nations. With a sustained computing power of 380 petaflops, LUMI ranks among the fastest supercomputers in the world (9th on the Top500 list, November 2025). The system runs on 100% renewable energy, and its waste heat is used to warm the city of Kajaani. All models in this repository (MNIST, USPS, SVHN) were trained on LUMI-G using AMD MI250x GPUs.

This page explains how to run training and evaluation on LUMI using the scripts provided in the `lumi/` directory. For a broader introduction to running AI workloads on LUMI, see the [Getting Started with AI on LUMI](https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop) workshop repository.

---

## Environment setup

For a detailed guide on setting up your environment on LUMI, see the [LUMI AI Guide — Setting up the environment](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/02-setting-up-environment).

The scripts in this repository use a Singularity container that provides PyTorch with ROCm support for AMD GPUs. The software environment is loaded inside each Slurm script automatically — no manual `module load` or `pip install` steps are needed.

---

## Training

The repository includes a ready-made Slurm script at `lumi/train.sh`. Submit it from the **repository root**:

```bash
# Train the default dataset (MNIST)
sbatch lumi/train.sh

# Train a different dataset by overriding the job name
sbatch -J svhn lumi/train.sh
sbatch -J usps lumi/train.sh
```

The job name (`-J`) selects the experiment: it loads `configs/<name>.cfg` for hyperparameters and names the log files `lumi/logs/<name>_<job-id>.{out,err}`.

To add a new dataset, create a matching `configs/<name>.cfg` file alongside the existing ones.

---

## Evaluation

A separate Slurm script at `lumi/evaluation.sh` evaluates a trained checkpoint on the test set. Submit it from the **repository root**:

```bash
# Evaluate the default dataset (MNIST)
sbatch lumi/evaluation.sh

# Evaluate a specific dataset
sbatch -J eval-svhn lumi/evaluation.sh
sbatch -J eval-usps lumi/evaluation.sh
```

The script derives the dataset name and checkpoint path from the job name (`eval-<dataset>` → `src/weights/<dataset>.pth`). If the checkpoint is not found, it exits with an error listing the available checkpoints.

> **Note:** Inference speed reported on LUMI may appear slower than on a local machine. This is expected — HPC environments introduce overhead from container startup, Slurm job scheduling, and parallel filesystem I/O that is not present in local runs.

---

## Monitoring jobs

```bash
# Check the status of your running jobs
squeue --me

# View the full log output of a job
cat lumi/logs/<job-name>_<job-id>.out

# Follow the log output in real time (while the job is running)
tail -f lumi/logs/<job-name>_<job-id>.out
```

---

## Tips

- Both scripts can also be submitted from inside the `lumi/` subdirectory:
  ```bash
  cd lumi && sbatch train.sh
  cd lumi && sbatch -J eval-svhn evaluation.sh
  ```
- The `--account` in the scripts is set to `project_465002757`. Update this if you are running under a different project allocation. Check your allocations with `csc-projects`.
- Training time is capped at 30 minutes and evaluation at 10 minutes in the default scripts. Adjust `#SBATCH --time` if needed.
