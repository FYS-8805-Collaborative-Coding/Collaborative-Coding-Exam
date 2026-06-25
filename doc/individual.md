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
...
