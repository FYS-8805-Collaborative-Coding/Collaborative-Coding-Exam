# Individual contributions
These are the individual contributions to the exam

## Andrea Dahlmo Løkke
My main contributions to the project included formulating the contribution workflow, setting up the backbone of the README, the initial implementation of the evaluation script, and generalizing the codebase both in terms of modularity and docstring standardization. I also contributed by actively reviewing and merging pull requests, with attention to upholding the contribution standards we had set. The contributions were made through 35 commits following the contribution standards, which can be confirmed by the following command:
```
git log main --author='andrea.d.lokke@uit.no' --no-merges --format='%h %ad %an <%ae> %s' --date=short | wc -l
```

### 1. Contribution Workflow and Repository Governance
My initial contribution was to set up branch protection on main with no admin bypass, so each pull request requires one approved review before merging. I additionally formulated the contribution workflow in CONTRIBUTING.md (commit `45d940b`), establishing a sustainable contribution standard for the repository. The workflow is based directly on the course material, using issues, feature branches, and pull requests as the collaboration mechanism. I also tried to uphold these standards during PR reviews; on a few occasions a PR was large or messy enough that I could have merged it as-is, but I asked the contributor to clean it up first so we actually followed the workflow we had set for ourselves.

### 2. README, Model Cards, and Project Documentation
I set up the initial barebone structure of the README file, including installation and inference instruction sections, and the model card structure (commits `9ce672b` and `f5161e7`). Later, I updated the USPS model card (commit `e7c0586`) and made general model card corrections regarding precision and recall (commit `c440077`, linked to my secret task), along with smaller README fixes.

### 3. Evaluation Pipeline
I set up a modular evaluation script for calculating precision, recall, and inference time (commit `6d7a460`). I made an initial effort on this script because it was important for my secret task (below). The script became the basis for the metric values reported in the model cards across all three datasets, and made it straightforward to re-evaluate models whenever weights changed. That is exactly what happened after the secret task bug was fixed (PR #135), when I re-ran the evaluation and updated the model cards with the corrected numbers.

### 4. Training Configuration and LUMI Scripts
The overall structure of the LUMI training script was already set up by one of my collaborators, but it only worked for one model. I implemented a generalized version that handles all models (commit `fe4e93d`) along with individual configuration files for each model's parameters (commit `1e6b523`). The configuration files also serve as a record of the training hyperparameters used for each model, instead of having to remember them or track them in a separate file.

### 5. USPS Dataset Support and Folder Standardization
I added test images from the USPS dataset as samples for testing the models (commit `e1b6e86`). I also standardized the folder structure of the downloaded datasets. The MNIST dataset was initially downloaded to `datasets/MNIST/`, so I ensured the same convention was followed for SVHN and USPS (commits `b977400` and `ff0e31a`). This standardization made the overall structure of the datasets folder much cleaner.

### 6. Codebase-wide Docstring Standardization
As one of my final tasks, I made a pass over the entire codebase to standardize docstrings. I used `interrogate` to assess docstring coverage, which initially sat at 60%. Many existing docstrings were also not optimal, so there was a lot to work through. The task was carried out in multiple PRs, where each touched between 1-3 files with a common theme (e.g. adding docstrings to functions missing one in `training.py` and `evaluation.py`). To avoid bloating the codebase with large docstrings, I implemented a multi-level scheme: every public API has full Parameters/Returns blocks, base classes have a short contract statement, and internal functions suffice with a single descriptive sentence. All docstrings were formatted consistently within their level. The task ended up spanning eight PRs, where some filled out missing docstrings (#171, #173, #177, #178, #179) and some improved the quality of existing ones (#183, #184, #185).

After passing through the docstrings, I restructured the API documentation to follow a logical workflow (commit `ef5128d`).

### Secret Task
My task was to make the `evaluation.py` script fail silently. I thought that an error producing the wrong numbers would be the most hidden, yet still runnable, kind of silent failure.
I implemented the script to calculate precision, recall, and inference time for our trained models. I used the `sklearn` functions `precision_score` and `recall_score`, and passed the argument `average='micro'` instead of `average='macro'`. This reduces both scores to plain accuracy, producing seemingly identical precision and recall values for the model being evaluated.
The bug was identified and fixed in PR #135. I helped by re-running the evaluation script and re-filling the model cards with the corrected values.

### What I found easy/difficult about working on the same functions 
What I found most difficult was that we did not plan out who was going to do what. The only way to know whether a task was being worked on was to check if there was an issue for it, and whether anyone was assigned. Sometimes issues had an assignee, sometimes not, and sometimes implementations came in without a preliminary issue. This happened mostly at the start of the exam, though, and it did not take long until everyone followed the workflow we had defined. I actually ended up enjoying looking through the issues to see the progress of the project. A conversation or group chat is not necessarily easier to manage, since discussions tend to get lost in the noise. Issues proved to be a solid way to keep a concise record of what needs to be done.

Another thing I found difficult was both conveying my own implementation ideas and understanding the other collaborators' ideas. Sometimes I felt overwhelmed by the many ways to solve a single problem.

I am grateful that the collaborators with an inclination towards coding took it upon themselves to implement the bulk of the codebase, as it would have taken me significantly longer to do the same task. Beyond commits and PRs, I also focused on contributing to discussions about modularity and general code design, where I often brought ideas that others implemented. The beauty of collaboration is that each collaborator can contribute with their expertise, which played to our advantage given the diversity of the group. I feel like I was able to identify and contribute to tasks that suited my base of knowledge, and the group was always positive about my efforts.

### Tools from the course I did not know before
I learned a lot in this course, as I was only getting started with git version control in my own projects. I was not familiar with how to use branches, pull requests, and merges, but now I feel confident navigating this workflow. There were many tools that were new to me, but I will mention a few that made the biggest difference:
- Branch protection rules on GitHub: protecting the main branch from unwanted or error-prone merges
- Cross-referencing issues in commit messages using keywords like `fixes #X` to automatically close issues on merge
- The distinction between cloning (for collaborators) and forking (for external contributors)
- Jupyter notebook sharing: "content-aware" version control using `nbdime`, and Binder for sharing notebooks
- Automated testing tools: writing test functions in separate test scripts was completely new to me
- Snakemake: being able to run scripts step-wise based on predefined rules was something I did not know about, but found extremely useful
- Creating pip-installable Python packages and deploying them to PyPI. I knew about this before the course, but I did not feel confident in how to do it.
- Containers: this is something I also knew of before the course, but I was more intimidated by it than inspired to use it.

### Experience with LUMI
During the project I wrote and ran the generalized `train.sh` script (commit `fe4e93d`) several times on LUMI to test the script. Getting a good overview of the SLURM parameters and what each one controlled was a bit difficult to begin with, but it became clearer once I had submitted a few jobs. I forgot to commit the resulting log files to the repository, and only realized this after LUMI access was closed before the end of the exam, so I no longer have my own Job-IDs to reference here. There is, however, an extra USPS training log committed by my collaborator under Job-ID `19201795`.

Beyond the project work itself, the LUMI workshop was very useful, and I learned a lot. Although some parts went above my head, I got a decent grip of what running programs on a supercomputer entails. Learning more about containers and how useful they are was the single most useful lesson from the workshop. I generally want to navigate in the CLI, but I rather used the GUI since it was easier to understand what was being done. That was especially important to me since I did not want to do anything wrong in the terminal of a supercomputer, just in case.

### Conclusion
My contributions were a combination of changes to the code base through commits and merged PRs, and a more conversational side where I opened a lot of issues (both to scope upcoming work and to flag bugs I found through testing) and engaged in discussions about modularity and standardization. I also think that defining the contribution workflow early gave a solid foundation for the collaboration to thrive on. These are habits I want to bring with me into future collaborative projects.


## Keyne Oei
...

## Myrthe Catteau
...

## Rahul Baburajan
My main contributions to the project were the original MNIST pipeline, the refactoring of inference toward a factory-style structure, ASCII text input support for inference, validation data support during training, and LUMI training scripts and logs. I also made smaller but important repository quality improvements, including documentation link fixes and README command alignments. Altogether, these contributions were made through 22 direct non-merge commits which can be verified using this command: 

```bash
git log main --author='rahul.baburajan@uit.no' --no-merges --format='%h %ad %an <%ae> %s' --date=short | wc -l
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
