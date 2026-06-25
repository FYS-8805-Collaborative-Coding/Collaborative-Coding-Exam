# Individual contributions
These are the individual contributions to the exam

## Andrea Løkke
...

## Keyne Oei
My main contributions were the SVHN part of the project (model and data), the command-line tools for inference and evaluation, logging, packaging and releasing the project to PyPI as `ccexam`, the bundled `samples:` input option, handling any image input (with or without a file extension, and folders), and the README presentation (banner, badges, header).

### Contribution:

### 1. SVHN Model and Data

This work added the SVHN part of the project.

- **Model (`SVHNNet`).** `SVHNNet` is a CNN for 32×32 RGB SVHN digits. It is separate from the simpler `DigitCNN` used by MNIST/USPS, because SVHN's photo digits are harder than clean handwritten ones. The network has three blocks. Each block has two `Conv2d(3×3, padding=1)` layers, each followed by `BatchNorm2d` and `ReLU`, and then a `2×2 MaxPool`. The number of channels grows `3 → 32 → 64 → 128`. The image size halves at each block (`32×32 → 16×16 → 8×8 → 4×4`). The classifier flattens the `128×4×4` features and applies `Dropout(0.3) → Linear(128·4² → 256) → ReLU → Dropout(0.3) → Linear(256 → 10)`.
- **Data (`SVHNDataModule`).** `SVHNDataModule` adds SVHN to the data layer. Unlike MNIST/USPS, torchvision's SVHN uses `split="train"/"test"` instead of `train=True/False`. So the data module checks which dataset it is and passes the right argument. SVHN is read from the `train_32x32.mat` and `test_32x32.mat` files. It is kept as 3-channel RGB (not converted to grayscale) and normalized with SVHN's mean/std.
- **Integration.** Registering `svhn` in the training list, the data-module list, and the inference list makes `--dataset svhn` and `--model svhn` work in the training, evaluation, and inference tools with no extra changes.

### 2. Command-Line Tools for Inference and Evaluation

The command-line tools make the package usable from the terminal, covering both inference and evaluation.

- **Inference tool** (`python -m src.inference`, and the installed `ccexam-infer` command). It takes `--model` to pick the classifier, `--input` for the image or folder to classify, `--device` (cpu/cuda/mps), `--checkpoint-path` to use different weights, and `--log-level`. By default it prints each prediction to the terminal.
- **Saving predictions.** The `--output`/`-o` option saves predictions to a file. The format follows the extension: `.csv` writes an `image,prediction` table with a header, and `.txt` writes tab-separated lines. Any folders in the path are created for you, and an existing file is never overwritten.
- **Evaluation tool** (`python -m src.evaluation`). It takes `--dataset`, `--checkpoint-path`, `--batch-size`, `--num-workers`, and `--device`. It loads the matching test set through the data module and reports precision, recall, and average inference speed.
- **Error handling.** Bad input now gives a clear message and a non-zero exit code, instead of a long traceback.

### 3. Logging

- Logging now runs across the evaluation, inference, and training code. It replaced one-off `print` statements with a shared logger and a single format (`timestamp | level | name | message`), set up in `src/utils.py`.
- `setup_logging` turns it on and `get_logger` gives a logger per file.
- Every run gets readable output the user can control with `--log-level`.

### 4. Packaging and PyPI Release (`ccexam`)

- The project is packaged and published to PyPI as [`ccexam`](https://pypi.org/project/ccexam/), so it installs with one `pip install ccexam`.
- The `pyproject.toml` file keeps the code in `src/` but installs and imports it as `ccexam`, and it adds the `ccexam-infer` command.
- The trained weights and sample images are bundled in the package, so inference works right after installing with no extra downloads.
- The imports between files were changed to relative imports, so the installed package imports correctly as `ccexam`. (A plain `from src.…` import works in the repo but breaks once installed.)
- The package also gained CI/PyPI/Python/License badges and corrected install instructions, and the releases were published.

### 5. Sample Images and the `samples:` Input Option

- The trained weights and example images now live inside the package, and a `samples:` input option lets the tool be tried without your own files.
- Writing `--input samples:svhn_digit_5.png` points to an image bundled in the package, so it works right after `pip install`.
- An unknown name prints the list of available samples.

### 6. Handling Any Image Input (Files, Folders, and No Extension)

The `--input` option now takes a single image or a whole folder. It decides what is an image by its content rather than its file extension.

- **Single file or folder.** Given one file, it classifies that file; given a folder, it classifies every valid image inside (sorted by name). The Python API works the same way: a single image returns one label, a folder returns a list of labels.
- **Checked by content, not extension.** Instead of trusting the file name, each file is checked by trying to open it as an image (with PIL). So a real image is accepted even if it has the wrong extension or **no extension at all**, while a non-image (a `.pdf`, or a text file renamed to `.png`) is rejected with a clear `Invalid input` message and a non-zero exit code.
- **Folders keep going.** When classifying a folder, files that can't be read are skipped with a logged note instead of stopping the whole run, so one bad file doesn't break the batch.

Together with the `samples:` option above, `--input` now handles a single image, a folder of images, an image with no extension, and a bundled sample in the same way, with clear error messages throughout.

### 7. README Presentation and Repository Upkeep

- The centered README header (logo, title, author line, badges, and section links) and the banner image were added.
- The model-card numbers (precision, recall, inference time) were kept up to date.
- After the samples moved into the package, the inference command paths in the docs were fixed so the examples still work.
- Several smaller fixes followed: changing the default `--num-workers` for evaluation to avoid a worker-shutdown hang, moving the RGB conversion into a module-level `_to_rgb` function so the dataset transform can be pickled by DataLoader workers on newer Python, adding tests for the model classes, and fixing the Sphinx documentation build.

### Secret Task

My secret task was to introduce a subtle mistake for other contributors to review and correct for file `src/training.py`. The change is in how that file configures logging, which I introduced with the logging work in commit `005f232`.

In `src/training.py`, the log statements live in the library code. `Trainer.train()` writes `INFO`/`DEBUG` log messages throughout the training loop. But `setup_logging()` is called **only inside the CLI entry point**, the `main()` function of `src/training.py`:

```python
# src/training.py — inside main()
    from .utils import setup_logging
    setup_logging(args.log_level)
```

Because logging is set up inside `main()` and not in the library code, training started any other way never sets up logging. This includes `train(...)`, `TrainerFactory.create(...).train()`, and the test suite. With nothing configured, the root logger defaults to `WARNING`. So every `INFO` training message from `src/training.py` is dropped without a trace. The logging looks like it works, but only because it goes through the CLI. Anyone calling the functions directly gets no output and no error. (One more weak point in the same file: `setup_logging` uses `logging.basicConfig`, which does nothing if logging was already set up, so calling it again has no effect.)

### What Was Easy and Difficult (Same Functions)

**Easy.** What made the shared code easy to work on was that we rarely had to edit the same function at the same time. The code used a factory and registry pattern. So each person could add their part in separate files and then register it. For me that was the SVHN model and data module. Because of this, most new work stayed out of the common functions, and changes from different people rarely collided. When I did need to touch shared code, the base classes and registries gave a clear pattern to follow, so it was obvious where things should go.

**Difficult.** The harder part was the code we all shared, where several people edited the same functions. We had no agreed standard for naming, style, or where each piece of code should go. So the shared base classes and training loop drifted between contributors and needed extra cleanup. I also found GitHub Issues less useful for tracking developer progress. Issues feel more aimed at users reporting problems. For working together as developers, a board like Trello is easier. You can see the progress and pick up tasks by moving cards between columns.

### Course Tools I Had Not Used Before

The two tools from the course I had not used before were **Sphinx** (for building the documentation) and **LUMI** (for running training jobs on the HPC cluster).

### Running Jobs on LUMI

I trained the SVHN model on LUMI by submitting a Slurm batch job (**Job-ID `19193045`**) with `sbatch`. The job ran on the `small-g` partition with one GPU, 7 CPUs, and a 30-minute time limit. The training ran inside a Singularity container (the prebuilt LAIF multitorch container, which ships ROCm-enabled PyTorch), so I did not have to install anything myself.

The run trained SVHN for 5 epochs and finished in about 90 seconds on the GPU. The model reached a final training accuracy of about 0.942, and the weights were saved to `weights/svhn.pth`:

```
2026-06-12 12:17:04 | INFO | training | Training on cuda for 5 epoch(s)
2026-06-12 12:18:25 | INFO | training | Epoch 5/5  loss=0.2103  acc=0.9421
2026-06-12 12:18:25 | INFO | training | Done!  final_loss=0.2103  final_acc=0.9421  saved=.../weights/svhn.pth
```

My experience was that the hardest part was the setup: the batch script, the container, and the file paths. Once that worked, the job ran quickly and reliably. Checking progress with `squeue --me` and reading the `slurm-<job-id>.out` log file made it easy to follow the run.

## Myrthe Catteau
...

## Rahul Baburajan
My main contributions to the project were the original MNIST pipeline, the refactoring of inference toward a factory-style structure, ASCII text input support for inference, validation data support during training, and LUMI training scripts and logs. I also made smaller but important repository quality improvements, including documentation link fixes and README command alignments. Altogether, these contributions were made through 22 direct non-merge commits which can be verified using this command: 

```bash
git log main --author='rahul.baburajan@uit.no' --no-merges --format='%h %ad %an <%ae> %s' --date=short
```

I trained the MNIST model on LUMI (**Job-ID `19201793`**) and its logs can be found in [lumi/logs/mnist_19201793.err](https://github.com/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam/blob/main/lumi/logs/mnist_19201793.err)

### 1. Initial MNIST Pipeline

I implemented the first substantial MNIST support across the core package. This included the data-loading module, model definition, training entry point, inference entry point, package exports, and contributor-facing instructions. The central commit for this contribution is `8361b79`. 

This contribution gave the repository its first complete path from data to model training and inference. It also provided a base that later contributors extended for USPS and SVHN support. 

### 2. Inference Architecture and Generalization

I refactored inference to be less MNIST specific and easier to extend. Commit `95bfaae` was the main architectural contribution, adding a factory style inference structure in `src/inference.py`. It also removed unwanted abstractions and MNIST specific classes from the inference path, making it suitable for extending to other datasets. 

This work improved the repository in three practical ways:

- It made inference easier to configure through explicit model and checkpoint handling.
- It reduced assumptions that tied inference only to MNIST.
- It created a better foundation for later multi-dataset inference work.

### 3. Training, Validation, LUMI Scripts, Logs, and Model Artifacts

I contributed training support that made the project more reproducible and useful beyond local experiments. 
- Added validation dataset support so the training workflow could separate training and validation behavior more clearly, and improved robustness in test environments with simplified torch tests. 
- Added and organized LUMI related training artifacts, including training scripts, logs, configuration files, and trained model weights for MNIST, SVHN, and USPS. 
- Updated the README with MNIST model card details and model performance information and inference speed updates.

### 4. ASCII Text Input Support for Inference

I implemented support for ASCII digit inputs so inference could accept text-based digit images in addition to standard image files. Commit `dea2dd6` added the initial ASCII text handling functionalities and sample ASCII digit files to test on. Later commits extended this support to handle different foreground and background character conventions in ASCII samples, while also refining tests.

The contribution included:

- ASCII sample digits like as `ascii_digit_0.txt`, `ascii_digit_2.txt` and `ascii_digit_8.txt`. 
- Utility logic for reading and converting ASCII digit representations.
- Inference integration so valid text inputs could enter the inference pipeline.
- Tests for ASCII parsing and supported symbol conventions.

### 5. Documentation and Repository Hygiene

I contributed documentation and repository maintenance improvements that made the project easier for collaborators to use and contribute to. 
- fixed README link formatting, aligned instructions with the actual codebase behavior and updated sample and weight handling documentation. 
- maintained contribution related documentation links, reducing setup friction and improving the overall clarity of the project.

### Secret Task 
My secret task is to intentionally introduce issues into the codebase that would later require review and correction by other contributors. 
- I added hardcoded dataset normalization values directly in the training and inference logic, instead of keeping these values in a shared source of truth. The idea was this would easily create issues if not careful because the normalisation values used during training should match with the test and inference time. It was later corrected by Ricardo by centralizing the dataset statistics and updating the relevant data, training, inference, and evaluation code to use them consistently.
- In the inference API, I made the return shape not consistent across different input cases. Depending on whether the input was a single file or a directory, the function could return different kinds of shapes as output, which made the interface harder to integrate with downstream usecases. This was later fixed by Riccardo by aligning the function behavior with the expected API and making the output structure consistent.

### Conclusion

My main role in this repository was to implement and maintain core machine learning workflow functionality particularly for the MNIST segment but also making sure that other dataset segments can be easily integrated in a easy manner. I contributed the initial MNIST data, model, training, and inference pipeline, which gave the project a working end to end baseline. I then refactored inference toward a more extensible factory style structure, removed assumptions that tied inference too closely to MNIST, and made checkpoint handling more explicit.

I also improved the training workflow by adding validation dataset support and by contributing LUMI-related scripts, logs, and trained weight artifacts. These changes helped the repository move from local experimentation toward more reproducible model training and evaluation. I documented the resulting model details and performance information in the README through MNIST model card and inference speed updates.

Another major contribution was ASCII text input support for inference. I added utilities for parsing text-based digit images, integrated those utilities into the inference pipeline, created ASCII digit samples, and added focused tests for ASCII handling. I later extended this support to handle multiple foreground/background conventions, making the feature more flexible for simple text based digit examples.

Beyond implementation work, I helped keep the repository maintainable by aligning README instructions with the code, fixing documentation links, updating setup instructions to use HTTPS, maintaining contribution guide links, and integrating several pull requests and handling issues from external users. 

## Riccardo Gelato
...

## Siyan Chen
My main task was to contribute the USPS dataset support, including the data module, model architecture, registry integration, and tests. Alongside this, I set up the initial project structure and CI/CD workflows, implemented and maintained the GitHub Pages documentation, added LUMI training and evaluation support, reviewed most of the pull requests from other contributors, and made several correctness fixes to the shared code that benefited all three datasets. These contributions can be verified in the commit history using the following command:

```bash
git log main --author='chensy618\|Siyan Chen' --no-merges --format='%h %ad %an <%ae> %s' --date=short
```

### Implementation Tasks and Choices

This section describes the main technical tasks I contributed during the project in detail.

#### 1. Repository Setup

I set up the initial repository structure and CI/CD workflow skeleton so that the project had a shared starting point for later implementation work. Commit `9c70f12` created the initial project structure: placeholder source files, the `CITATION.cff` file with author metadata, `datasets/dataset_links.txt`, `environment.yml`, and the four GitHub Actions workflow files as `# TODO` stubs. Commit `c0229c5` completed the implementation of all four workflows: format, test, docs, and release. To avoid failing CI notifications while the rest of the codebase was still being built out, each workflow step used a soft-fail pattern (`2>/dev/null || echo "TODO: ..."`) so pipelines stayed green during early development. This was useful because the repository needed CI/CD structure before all source files, tests, and documentation were complete. The soft-fail setup allowed the team to keep a consistent workflow layout from the beginning without blocking early development. 

#### 2. USPS Model Design

For USPS support, I followed the existing modular structure of the project instead of creating a separate pipeline. This design choice made USPS consistent with the other supported datasets and allowed the existing CLI, training, evaluation, and inference entry points to reuse the same shared interfaces.

Before finalizing the model, I discussed the model choice with Andrea. We considered several options, including SVM, MLP, and CNN models. After testing these alternatives, the CNN achieved the highest accuracy, so we chose it as the final USPS model. This choice worked well because USPS images are small `16×16` grayscale digit images, and convolutional layers can capture local digit features more effectively than the simpler baselines.

The core USPS contribution covered three parts: dataset integration, model design, and test/checkpoint support. First, commit `7d7dedd` added `USPSDataModule` in `src/data.py` with the correct USPS normalization statistics, and implemented `USPSNet` in `src/models.py` as a two-block CNN designed for 16×16 grayscale inputs. USPS was also registered in the dataset registry, so it could be selected automatically through the same project workflow as MNIST and SVHN. The model architecture was kept compact because USPS images are small grayscale images. A two-block CNN was sufficient for extracting local digit features while avoiding unnecessary complexity. Batch normalization and ReLU were used to make training more stable, max pooling reduced the spatial feature size, and the final fully connected layers mapped the extracted features to the ten digit classes. Second, commits `2f372b6` and `a5158f2` added 316 lines of tests in `tests/test_data.py`. These tests covered normal data loading, edge cases, validation scenarios, dataset routing, loader behavior, normalization, and parameter forwarding. This was important because USPS support touched shared dataset-loading code, so the tests helped ensure that adding USPS did not break MNIST or SVHN behavior. Third, commit `f32bedf` added a pretrained USPS checkpoint alongside the model. This made USPS usable as a complete supported model rather than only a dataset definition, allowing users to run inference or evaluation without retraining first. 

#### 3. Training and Evaluation Fixes and Improvements

I also made several fixes to the shared training and evaluation code so that all supported datasets could use the same workflow more reliably. Relevant commits are shown below:

- `d4e3460` (`Fixes #93`): `DataLoader` objects were being created inside the epoch loop. I moved them outside the loop so the loaders are constructed once per training run.
- `a6a8620` (`Fixes #94`): Device selection was extended to support Apple Silicon GPUs through the MPS backend.
- `25f9057` (`Fixes #95, #97, #100`): I added a LUMI evaluation script and fixed the LUMI training script so both scripts run from the repository root and write logs to the correct `lumi/logs` directory. I also updated evaluation precision and recall from micro averaging to macro averaging so each class contributes equally to the final metric.
- `6bbba7b` (`Fixes #110, #111, #112, #113`): I fixed several training, evaluation, and inference warnings and platform-related issues, including safer model loading with `weights_only=True` and Mac/MPS-related cleanup behavior.
- `30ec436` (`Fixes #119`): I added config-file support for training, allowing dataset-specific training settings to be loaded through `--config` while still letting command-line arguments override config defaults.
- `e0da851` (`Fixes #123`): After noticing signs of overfitting in the training logs, I discussed the issue with Rahul and updated the learning rate to `3e-4` to make the optimization more stable.
- `038f3a0`: I fixed the weight/checkpoint path problem so training, inference, evaluation, and tests resolved model weights consistently from the expected location.

These fixes made the shared training and evaluation workflow more robust, easier to reproduce, and more consistent across all three supported datasets.

#### 4. GitHub Pages and Documentation

I also worked on the GitHub Pages and Sphinx documentation site so that the project could be understood and used without reading the source code directly. Across commits `ccf65af`, `cde08d6`, `47e519d`, and `0f0154d`, I built out most of the documentation structure and updated the content as the codebase changed.

The documentation covered several parts of the project. I wrote a beginner-friendly Getting Started guide (`ccf65af`) covering installation, running inference on a single image or a folder, and common troubleshooting tips. I also created API reference pages for the main modules, including inference, training, models, and data handling, so users could understand how the shared functions and classes were organized.

In addition to the general documentation, I added development-related guides and LUMI usage instructions. These pages explained how to work with the repository structure, how to run training and evaluation jobs on LUMI, and how the dataset-specific workflows were connected to the shared command-line interface. I also updated the README with more accurate instructions and sample images for MNIST, USPS, and SVHN, so the three supported datasets were documented consistently.

The main implementation choice was to keep the documentation aligned with the modular structure of the codebase. Instead of writing separate instructions for each dataset from scratch, I documented the shared workflow and then explained how MNIST, USPS, and SVHN fit into it. This made the documentation easier to maintain and helped users understand that the three datasets follow the same overall training, evaluation, and inference pipeline.

#### 5. Review and Tests

In addition to my own implementation tasks, I also reviewed many pull requests from other contributors and helped resolve merge conflicts when changes affected the same shared files.

As other contributors refactored the codebase, the test suite regularly broke and needed updating. Commits `70a7fec`, `1884c6f`, and `d888c8f` fixed a series of failures in `tests/test_training.py` caused by changes to the fake `torch` module used in unit tests. The `torch.ops` namespace was missing attributes expected by newer versions of `torchvision`, and the module paths for `data` and `models` had changed to `src.data` and `src.models` after the package restructuring. Commit `f4587ad` resolved a test failure in `tests/test_inference.py` triggered by a change in `src/inference.py`. These fixes kept the CI green and unblocked other contributors' pull requests. 

I also tested the Windows setup, including training, inference, and evaluation, to check that the shared workflow worked outside the main macOS development environment.

### Secret Task

My secret task was to secretly introduce subtle issues into the codebase.

**Attempt 1 — Training loop order (`d939c76`):** Moved `optimizer.zero_grad()` from before the forward pass to after `optimizer.step()`. This looked like a possible gradient-handling issue, but in this project it did not have a strong practical effect because gradients were still cleared once per batch. 

**Attempt 2 — MaxPool kernel size (detected by CI):** Changed `nn.MaxPool2d(2)` to `nn.MaxPool2d(3)` in `USPSNet`. The original architecture used `MaxPool2d(2)` in both layers, producing 64×4×4 feature maps for the classifier head. Changing the second pool to `MaxPool2d(3)` reduced the spatial dimensions further, causing a shape mismatch in the classifier head. The CI test suite caught this automatically. Commit `e612f6a` reverted the pool size back to 2, and the architecture was later properly updated in `6bbba7b`.

### What I Found Easy and Difficult

**Easy:** Adding the USPS data module and model followed the existing MNIST structure closely, so the interface contract was clear from the start. Writing tests for `data.py` was also straightforward once the first module was in place. Note that, with the help of LLMs and AI coding agents, reusing existing module patterns and drafting initial code became much easier.

**Difficult:** Resolving merge conflicts was one of the most difficult parts of the project, especially in shared files such as `src/training.py`. Several contributors were editing the same training pipeline at the same time, so the conflicts were not just simple text conflicts. I had to compare overlapping changes carefully, keep the parts that matched the final shared design, and run tests again to make sure the resolved version still worked.

Larger refactors were especially challenging because many conflicts remained after rebasing, and other contributors continued to update the same files while the conflicts were being resolved. For example, Riccardo's larger refactor pull request stayed open for a long time, partly because it touched many shared files and had to be reconciled with ongoing updates from other contributors.

Several fixes also required understanding the interaction between PyTorch internals and the operating system. The MPS DataLoader hang, for instance, only appeared on Apple Silicon and required careful reading of PyTorch multiprocessing behavior.

### Tools from the Course Used

- **GitHub Actions** — This was my first time writing CI workflow YAML. I implemented the test, format, docs, and release workflows, and learned how to define jobs, steps, triggers, and simple fallback behavior for early development.
- **LUMI / Slurm** — I was already familiar with submitting jobs on Olivia, but this project helped me learn how to build a working environment on an HPC system more systematically, including storage allocation, Singularity containers, working-directory handling, and writing batch scripts that behave consistently on the cluster.
- **Sphinx** — I learned how to use Sphinx to build and organize the project documentation site.
- **Structured PR and issue workflow** — I learned a more collaborative and disciplined GitHub workflow than I had used before. Every change was tracked through a GitHub issue, discussed in pull requests, reviewed by other contributors, and merged only after the team agreed on the implementation.

### Experience Running Jobs on LUMI

I submitted USPS training on LUMI-G (AMD MI250x GPUs) via `lumi/train.sh` (**Job-ID `19372444`**), using the `standard-g` partition with one GPU, 7 CPUs, and 60 GB of GPU memory. The run used 50 epochs, batch size 32, and learning rate `3e-4`. The job ran successfully and reached `val_acc=0.963` by the first epoch. Logs can be found in [lumi/logs/usps_19372444.err](https://github.com/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam/blob/main/lumi/logs/usps_19372444.err).

The first submission produced empty log files because the weight path was wrong (`weights/` instead of `src/weights/`). After fixing this in commit `038f3a0` and resubmitting, the job completed in under 2 minutes of GPU time.

### Conclusion

My primary role in this project was to deliver end-to-end USPS support and help maintain the shared project infrastructure. I set up the initial repository structure and CI workflows, contributed the USPS data module, model, registry integration, and tests, and made improvements to the shared training and evaluation workflow. I also worked on the GitHub Pages documentation site and ran USPS training on LUMI.

The most challenging part was working in a shared codebase with multiple contributors. Resolving merge conflicts required understanding how different changes fit into the final design, rather than simply choosing one version over another. Some fixes also required understanding platform-specific behavior in PyTorch.
