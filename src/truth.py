"""Turn contour-outline masks into a ground-truth list of barnacles.

A barnacle is a hole enclosed by an outline (decision 1: option A, 10 px cutoff).
Optionally the outline layer is morphologically closed first so that outlines
with 1-2 px gaps become enclosed holes.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

from .dataio import load_outline_mask

MIN_HOLE_AREA = 10  # px; below this a hole is a gap inside a thick line


def close_outlines(outline, ksize):
    """Seal small gaps in the outline layer with a morphological closing.

    cv2.morphologyEx(MORPH_CLOSE) dilates then erodes with the same kernel,
    which bridges gaps narrower than the kernel while keeping line width.
    """
    if ksize <= 1:
        return outline
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(outline.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0


def label_holes(outline, min_area=MIN_HOLE_AREA):
    """Label background regions fully enclosed by outline pixels.

    ndi.label gives each 4-connected background region an integer id; regions
    touching the image border are the outside, not holes.
    """
    lab, _ = ndi.label(~outline)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    areas = np.bincount(lab.ravel())
    keep = areas >= min_area
    keep[border] = False
    keep[0] = False
    lab = np.where(keep[lab], lab, 0)
    lab, n = ndi.label(lab > 0)
    return lab, n


def unclosed_outline_hulls(outline):
    """Convex-hull masks of outline blobs that enclose no hole (open outlines)."""
    m = outline.astype(np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    parents = {h[3] for h in hier[0] if h[3] != -1}
    hulls = []
    for i, (c, h) in enumerate(zip(cnts, hier[0])):
        if h[3] == -1 and i not in parents:
            hm = np.zeros(m.shape, np.uint8)
            cv2.fillPoly(hm, [cv2.convexHull(c)], 1)
            hulls.append(hm > 0)
    return hulls


def compare_closing(outline, ksize):
    """Diagnose what closing with kernel ksize does relative to no closing."""
    lab0, n0 = label_holes(outline)
    lab1, n1 = label_holes(close_outlines(outline, ksize))
    hulls = unclosed_outline_hulls(outline)

    # which original holes overlap each new hole
    pairs = np.unique(np.stack([lab1[lab1 > 0], lab0[lab1 > 0]]), axis=1)
    per_new = {}
    for new, old in pairs.T:
        per_new.setdefault(new, set())
        if old:
            per_new[new].add(old)
    merged = [k for k, v in per_new.items() if len(v) >= 2]
    survived_old = set().union(*per_new.values()) if per_new else set()
    lost = n0 - len(survived_old)
    fresh = [k for k, v in per_new.items() if not v]

    areas1 = np.bincount(lab1.ravel())
    recovered, spurious, closed_blobs = 0, 0, set()
    for k in fresh:
        region = lab1 == k
        hit = [j for j, h in enumerate(hulls) if (region & h).sum() >= 0.5 * areas1[k]]
        if hit:
            recovered += 1
            closed_blobs.update(hit)
        else:
            spurious += 1
    return {
        "ksize": ksize, "holes": n1, "change": n1 - n0,
        "open_outlines_closed": f"{len(closed_blobs)}/{len(hulls)}",
        "recovered_holes": recovered, "spurious_new_holes": spurious,
        "merged_holes": len(merged), "lost_holes": lost,
        "max_hole_area": int(areas1[1:].max()) if n1 else 0,
    }


def mask_instances(name, min_area=MIN_HOLE_AREA):
    """Turn a mask PNG into barnacle instances: centroids (row, col) and areas.

    1. Binarize: outline pixels are where alpha > 0.
    2. Fill: ndi.binary_fill_holes sets every background pixel that cannot be
       reached from the image border to True, i.e. fills each closed outline.
    3. Interiors = filled minus the lines. Labelling interiors (not the filled
       blobs) keeps touching barnacles separate, since their outlines merge
       but their interiors do not.
    4. Label: cv2.connectedComponentsWithStats gives each 4-connected interior
       an id plus its pixel area and centroid in one pass.
    """
    lines = load_outline_mask(name)
    interiors = ndi.binary_fill_holes(lines) & ~lines
    n, _, stats, cents = cv2.connectedComponentsWithStats(
        interiors.astype(np.uint8), connectivity=4)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas >= min_area
    return cents[1:][keep][:, ::-1], areas[keep]


def equivalent_diameter(areas):
    """Diameter of the circle with the same area: sqrt(4A/pi)."""
    return np.sqrt(4 * np.asarray(areas) / np.pi)


def barnacle_centroids(outline, ksize=1):
    """Ground-truth barnacle centers (row, col) and areas from an outline mask."""
    lab, n = label_holes(close_outlines(outline, ksize))
    idx = np.arange(1, n + 1)
    cents = np.array(ndi.center_of_mass(np.ones_like(lab), lab, idx))
    areas = ndi.sum(np.ones_like(lab), lab, idx)
    return cents, areas
