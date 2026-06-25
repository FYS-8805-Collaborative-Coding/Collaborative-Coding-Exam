# Individual contributions
These are the individual contributions to the exam

## Andrea Løkke
...

## Keyne Oei
...

## Myrthe Catteau
...

## Rahul Baburajan
...

## Riccardo Gelato

Me and Rahul Baburajan were responsible for Customer A: the MNIST Dataset.

### 1. Implementation Tasks

I worked on most of the core files (`training.py`, `data.py`, `models.py`, `inference.py`, and `evaluation.py`), focusing on increasing the modularity of the project and ensuring that modules could interact with each other with minimal knowledge of the underlying codebase. I also aimed to reduce code bloat and improve the overall extendability of the project. In addition, I contributed to writing `instructions.md` and the `README`, wrote test functions for `training.py` and `inference.py`, and contributed to the other test suites.

### 2. Implementation Choices

The main programming pattern I adopted to reduce bloat and increase modularity was the **Factory Pattern**, which allowed the creation of objects (trainers, loaders, inference pipelines) without tightly coupling the calling code to specific implementations. This pattern was applied across `training.py`, `inference.py`, and `data.py`. We also made heavy use of abstract base classes and polymorphism to further increase modularity and flexibility.

A specific focus of mine was increasing the coupling between `evaluation.py` and `inference.py` to reduce code repetition. I also worked to eliminate hardcoded variables across the different files, grouping them instead in a centralised `constants.py` script.

### 3. What I Found Easy / Difficult

After introducing the Factory Pattern, merge conflicts became easier to resolve, since module boundaries were clearer and responsibilities better separated.

The main challenge was introducing the unified `constants.py`. This change required simultaneous modifications across multiple files and proved far more complex than anticipated — it caused numerous conflicts and required updates to the test suite as well. Another significant challenge was increasing the coupling between `evaluation.py` and `inference.py`, as this required modifying files that were being concurrently edited by colleagues for different purposes, which meant carefully resolving conflicts while preserving all introduced functionality.

### 4. New Tools from the Course

I had used GitHub before, but had never worked with branching and pull requests in a collaborative setting. I also deepened my knowledge of conda environment management, running jobs on supercomputers such as LUMI, and software testing practices.

### 5. Experience Running Jobs on LUMI

I was previously accustomed to running jobs via the command line (PuTTY) on Karoline. LUMI's interface made this process considerably more intuitive, and once the job script was correctly configured, submitting training runs was straightforward. I ran **Job-ID 19344734**. Monitoring progress with `squeue --me` and reading the `slurm-<job-id>.out` log file made it easy to follow the run in real time.

### 6. Secret Task

My secret task was to introduce a subtle bug into `src/models.py`. I did so by hardcoding the batch size in two places.

The first instance was in the `forward` method:

```python
def forward(self, x):
    x = self.features(x)
    x = x.view(64, -1)  # batch size hardcoded to 64
    return self.classifier(x)
```

This bug was promptly detected and corrected by Rahul Baburajan, preventing further disruption.

The second instance was subtler — in `_build_classifier`, the `in_features` parameter was silently ignored in favour of a hardcoded value:

```python
def _build_classifier(self, in_features: int) -> nn.Sequential:
    """Override to adjust the input features based on the smaller feature map."""
    return super()._build_classifier(in_features=32 * self.feature_dim**2)
```

This bug went unnoticed for longer, only surfacing during more purposeful training and evaluation runs. Once caught, it was promptly fixed by Keyne Oei and Andrea Løkke by simply removing the override entirely and letting the parent class handle the method directly.

## Siyan Chen
...
