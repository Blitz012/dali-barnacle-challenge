"""Loading images and annotation masks for the barnacle challenge."""
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_rgb(name):
    """Load an image as an RGB uint8 array (drops alpha if present)."""
    im = cv2.imread(str(DATA_DIR / name), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(DATA_DIR / name)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[..., :3]
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_outline_mask(name):
    """Load an annotation mask as a boolean array of contour-outline pixels.

    The contours are stored only in the alpha channel; RGB is white wherever
    alpha is 0, so reading without IMREAD_UNCHANGED loses the annotation.
    mask1 uses two outline colours (top and bottom halves); we ignore colour.
    """
    im = cv2.imread(str(DATA_DIR / name), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(DATA_DIR / name)
    if im.ndim != 3 or im.shape[2] != 4:
        raise ValueError(f"{name} has no alpha channel; expected RGBA mask")
    return im[..., 3] > 0


def describe(name):
    """Return shape, dtype and alpha summary for a file, for the data audit."""
    im = cv2.imread(str(DATA_DIR / name), cv2.IMREAD_UNCHANGED)
    info = {"file": name, "shape": im.shape, "dtype": str(im.dtype), "alpha": "none"}
    if im.ndim == 3 and im.shape[2] == 4:
        a = im[..., 3]
        info["alpha"] = "constant 255" if a.min() == 255 else f"varies ({(a > 0).mean():.1%} opaque)"
    return info
