"""Laplacian-of-Gaussian on inverted greyscale: the first detector, now rejected.

It assumed plates are dark spots. In img1 they are mid-grey and the darkest
spots are crevices between shells, so <1% of detections land on a plate.
Kept as the evidence for switching to template matching (src/templates.py).
Diameter convention: d ~= 2*sqrt(2)*sigma.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.feature import blob_log

from . import region

# Starting values (untuned). sigma 3.5-9 ~ diameters 10-25 px.
# threshold and overlap are blob_log's library defaults, not chosen values.
DEFAULTS = dict(min_sigma=3.5, max_sigma=9.0, num_sigma=10, threshold=0.2, overlap=0.5)


def sigma_to_diameter(sigma):
    return 2 * np.sqrt(2) * np.asarray(sigma)


def inverted_gray(rgb):
    """Greyscale in [0, 1], inverted so dark plates become bright blobs."""
    return 1.0 - rgb2gray(rgb)


def log_scores(inv, rows, cols, sigmas):
    """Scale-normalised LoG response at each detection's centre and sigma.

    ndi.gaussian_laplace blurs with a Gaussian of width sigma and takes the
    Laplacian; multiplying by -sigma**2 makes bright blobs positive and makes
    responses comparable across scales. This is a ranking score, not a
    probability.
    """
    scores = np.empty(len(rows))
    for s in np.unique(sigmas):
        sel = sigmas == s
        resp = -(s ** 2) * ndi.gaussian_laplace(inv, s)
        scores[sel] = resp[rows[sel], cols[sel]]
    return scores


def fraction_on_plates(rows, cols, mask_name):
    """Share of detections whose centre falls inside an annotated plate."""
    from .dataio import load_outline_mask
    lines = load_outline_mask(mask_name)
    interiors = ndi.binary_fill_holes(lines) & ~lines
    return interiors[rows, cols].mean(), int(interiors[rows, cols].sum())


def detect(rgb, corners=None, **params):
    """Detect on the whole image, then keep centres inside the frame polygon.

    skimage.feature.blob_log builds a stack of LoG-filtered images over a range
    of sigmas and returns local maxima above `threshold` as (row, col, sigma);
    blobs overlapping by more than `overlap` are pruned to the larger one.
    Returns a dict of arrays: rows, cols, sigma, diameter, score.
    """
    p = {**DEFAULTS, **params}
    inv = inverted_gray(rgb).astype(np.float32)
    blobs = blob_log(inv, **p)
    rows, cols = blobs[:, 0].astype(int), blobs[:, 1].astype(int)
    sig = blobs[:, 2]
    if corners is not None:
        keep = region.inside(np.c_[rows, cols], corners)
        rows, cols, sig = rows[keep], cols[keep], sig[keep]
    return {
        "rows": rows, "cols": cols, "sigma": sig,
        "diameter": sigma_to_diameter(sig),
        "score": log_scores(inv, rows, cols, sig),
    }
