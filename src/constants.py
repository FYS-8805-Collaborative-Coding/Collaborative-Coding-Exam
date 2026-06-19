"""Per-dataset image statistics used by ``src.data`` to build input transforms."""

DATASET_STATS = {
    "mnist": {"mean": (0.1307,), "std": (0.3081,), "image_size": 28, "grayscale": True},
    "usps":  {"mean": (0.2471,), "std": (0.2994,), "image_size": 16, "grayscale": True},
    "svhn":  {"mean": (0.4377, 0.4438, 0.4728), "std": (0.1980, 0.2010, 0.1970), "image_size": 32, "grayscale": False},
}