# About this code

## Getting started

All you need is a terminal (command prompt) and Python 3.10 or newer. Follow the steps below in order.

**Step 1 — Install the package**

Open a terminal and run:

```bash
pip install ccexam
```

You only need to do this once.

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

In your terminal, navigate to the folder where you saved the image, then run:

```bash
ccexam-infer --model usps --input sample_digit.png
```

The tool will print the digit (0–9) it thinks is in the image.

Want to classify all images in a folder at once?

```bash
ccexam-infer --model usps --input path/to/your/folder
```

The predictions can be piped to an output file using the `--output` argument:

```bash
ccexam-infer --model usps --input path/to/your/folder --output path/to/your/output_file.csv
```

On a laptop with no dedicated graphics card, add `--device cpu`:

```bash
ccexam-infer --model usps --input sample_digit.png --device cpu
```

Available models: `mnist`, `usps`, `svhn` — pick whichever matches your images.

---

**No image handy? Try a bundled sample**

Use the `samples:` prefix to run on an example image shipped inside the package — works immediately after install, no download needed:

```bash
ccexam-infer --model svhn  --input samples:svhn_digit_5.png
ccexam-infer --model mnist --input samples:ascii_digit_5.txt
```

An unknown name will list the available samples.

---

**What inputs are accepted?**

- **Any image, recognised by its content — not its file name.** A real image works even with an unusual extension or **no extension at all**; a non-image (a PDF, a text file renamed `.png`, etc.) is rejected with a clear `Invalid input` message.
- **ASCII-art digits** stored in a `.txt` file (a digit drawn with characters) are detected automatically and converted before inference — no special flag:
  ```bash
  ccexam-infer --model mnist --input ascii_digit.txt
  ```
  Note that ASCII art is coarse and out of distribution for these models, so results are best-effort.
- A **folder** classifies every readable image inside and simply skips files it can't read.

---

## Need help?

- Run `ccexam-infer --help` to see all available arguments and their descriptions.
- Make sure Python 3.10 or newer is installed — check with `python --version`.
- If `ccexam-infer` is not found after installing, close and reopen your terminal, then try again.
- Do **not** add extra text after the command (e.g. avoid `./` at the end).
- For further questions, reach out to the team listed on the [Individual contributions](individual.md) page.
