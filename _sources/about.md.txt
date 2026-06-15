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

On a laptop with no dedicated graphics card, add `--device cpu`:

```bash
ccexam-infer --model usps --input sample_digit.png --device cpu
```

Available models: `mnist`, `usps`, `svhn` — pick whichever matches your images.

---

## Need help?

- Make sure Python 3.10 or newer is installed — check with `python --version`.
- If `ccexam-infer` is not found after installing, close and reopen your terminal, then try again.
- Do **not** add extra text after the command (e.g. avoid `./` at the end).
- For further questions, reach out to the team listed on the [Individual contributions](individual.md) page.
