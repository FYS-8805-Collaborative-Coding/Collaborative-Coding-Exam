# About this code

## Getting started

All you need is a terminal (command prompt) and Python 3.10 or newer. Follow the steps below in order.

**Step 1 — Install the package**

Open a terminal and run:

```bash
pip install ccexam
```

---

**Step 2 — Get a sample image to try**

Download this example digit image and save it somewhere easy to find (e.g. your Desktop or Downloads folder):

```{image} _static/sample_digit.png
:alt: Sample digit 8
:width: 80px
```

{download}`Download sample_digit.png <_static/sample_digit.png>`

---

**Step 3 — Run the tool on your image**

Open a terminal and run the command below, replacing the path with wherever you saved the image:

```bash
ccexam-infer --model svhn --input /path/to/sample_digit.png
```

The tool will print the digit (0–9) it thinks is in the image.

Want to classify all images in a folder at once?

```bash
ccexam-infer --model svhn --input path/to/your/folder
```

The predictions can be piped to an output file using the `--output` argument:

```bash
ccexam-infer --model svhn --input path/to/your/folder --output path/to/your/output_file.csv
```

By default the tool picks the best available device. You can override it with `--device`:

```bash
ccexam-infer --model svhn --input /path/to/sample_digit.png --device cpu   # any machine
ccexam-infer --model svhn --input /path/to/sample_digit.png --device mps   # Apple Silicon Mac
ccexam-infer --model svhn --input /path/to/sample_digit.png --device cuda  # NVIDIA GPU
```

Available models: `mnist`, `usps`, `svhn` — pick whichever matches your images.

---

**No image handy? Try a bundled sample**

Use the `samples:` prefix to run on an example image shipped inside the package — works immediately after install, no download needed:

```bash
ccexam-infer --model svhn  --input samples:svhn_digit_5.png
ccexam-infer --model mnist --input samples:ascii_digit_5.txt
```

If the name is not found, the error message will print the full list of available sample names.

---

**What inputs are accepted?**

| Input type | How to pass it | Notes |
|---|---|---|
| **Image file** | `--input /path/to/image.png` | Detected by content, not extension. Works with any extension or no extension at all. Non-images are rejected with an `Invalid input` message. |
| **ASCII-art digit** | `--input digit.txt` or `--input digit.ascii` | Auto-detected and converted before inference. Foreground chars: `#` `X` `1` `@` `*`; background: `.` space `0` `-`. All rows must be the same width. Results are best-effort. |
| **Folder** | `--input /path/to/folder/` | Classifies every readable image inside. Unreadable files are skipped silently. |

---

## Need help?

- Run `ccexam-infer --help` to see all available arguments and their descriptions.
- Make sure Python 3.10 or newer is installed — check with `python --version`.
- If `ccexam-infer` is not found after installing, close and reopen your terminal. If it still fails, use the module form as a fallback — it works regardless of `PATH`:
  ```bash
  python -m ccexam.inference --model svhn --input /path/to/image.png
  ```
- For further questions, reach out to the team listed on the [Individual contributions](individual.md) page.
