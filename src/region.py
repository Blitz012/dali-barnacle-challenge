"""Evaluation region: the inner square of the wire frame, marked by hand.

Corners are stored in data/corners.json as {image: [[x, y], ...]} in the order
top-left, top-right, bottom-right, bottom-left, at the inner edge of the wire.

Mark or re-mark corners from a terminal (opens a window):
    .venv/bin/python -m src.region img1.png
Click near a corner on the full image, then click precisely on the zoomed view.
"""
import json
import sys

import numpy as np
from matplotlib.path import Path as MplPath

from .dataio import DATA_DIR, load_rgb

CORNERS_FILE = DATA_DIR / "corners.json"
CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]


def load_corners(name):
    """Return the 4x2 array of (x, y) inner-frame corners for an image."""
    return np.array(json.loads(CORNERS_FILE.read_text())[name], dtype=float)


def save_corners(name, corners):
    data = json.loads(CORNERS_FILE.read_text()) if CORNERS_FILE.exists() else {}
    data[name] = [[int(round(x)), int(round(y))] for x, y in corners]
    CORNERS_FILE.write_text(json.dumps(data, indent=2))


def inside(points_rc, corners):
    """Boolean mask: which (row, col) points fall inside the corner polygon.

    matplotlib's Path.contains_points does a point-in-polygon test; it takes
    (x, y), so rows and cols are swapped first.
    """
    pts = np.asarray(points_rc)[:, ::-1]
    return MplPath(corners).contains_points(pts)


def roi_mask(shape, corners):
    """Boolean image mask of pixels whose centre is inside the corner polygon.

    Uses the same point-in-polygon test as `inside`, evaluated only on the
    polygon's bounding box, so pixel and centroid filtering agree exactly.
    """
    m = np.zeros(shape[:2], bool)
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, shape[1] - 1), min(y1, shape[0] - 1)
    rr, cc = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    m[y0:y1 + 1, x0:x1 + 1] = inside(np.c_[rr.ravel(), cc.ravel()], corners).reshape(rr.shape)
    return m


PROVISIONAL_FILE = DATA_DIR / "corners_provisional.json"


def load_provisional_corners(name):
    """Corners read off by eye for an image without verified corners.

    These are a provisional review region, not verified ground truth; see the
    "status" field in data/corners_provisional.json.
    """
    d = json.loads(PROVISIONAL_FILE.read_text())
    return np.array(d["corners"][name], dtype=float)


def mark_corners(name, zoom=80):
    """Interactive two-stage click: rough click on full image, precise on zoom."""
    import matplotlib.pyplot as plt

    img = load_rgb(name)
    corners = []
    for cname in CORNER_NAMES:
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(img)
        ax.set_title(f"{name}: click near the INNER {cname} corner")
        (x, y), = plt.ginput(1, timeout=0)
        plt.close(fig)
        x0, y0 = int(max(x - zoom, 0)), int(max(y - zoom, 0))
        fig, ax = plt.subplots(figsize=(8, 8))
        view = img[y0:y0 + 2 * zoom, x0:x0 + 2 * zoom]  # may be truncated at the image edge
        h, w = view.shape[:2]
        ax.imshow(view, extent=(x0 - 0.5, x0 + w - 0.5, y0 + h - 0.5, y0 - 0.5))
        ax.set_title(f"click exactly where the inner wire edges meet ({cname})")
        (x, y), = plt.ginput(1, timeout=0)
        plt.close(fig)
        corners.append((x, y))
    save_corners(name, corners)
    print(name, np.round(corners).astype(int).tolist())


if __name__ == "__main__":
    for n in sys.argv[1:]:
        mark_corners(n)


def wire_intrusions(rgb, quad, margin=8, group_px=15, min_area=40):
    """Boxes around teal frame hardware (tabs, ring clips) that pokes inside `quad`.

    Teal pixels are those with G and B both more than 25 above R. Pieces
    within `group_px` of each other are grouped (a ring clip's shiny centre is
    not teal), and each group's bounding box is padded by `margin`. Returns
    (row0, row1, col0, col1) boxes. Only a starting point: every box must be
    checked visually before use.
    """
    from scipy import ndimage as ndi
    from skimage.measure import label, regionprops

    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    teal = (g - r > 25) & (b - r > 25) & roi_mask(rgb.shape, quad)
    groups = label(ndi.binary_dilation(teal, iterations=group_px))
    boxes = []
    for p in regionprops(groups * teal):
        if p.area >= min_area:
            r0, c0, r1, c1 = p.bbox
            boxes.append((r0 - margin, r1 - 1 + margin, c0 - margin, c1 - 1 + margin))
    return boxes


def notched_polygon(quad, boxes):
    """Cut rectangular notches out of a quadrilateral, keeping it one simple polygon.

    `quad` is [x, y] TL, TR, BR, BL. Each box (row0, row1, col0, col1) is assigned
    to the side it touches; the notch follows the side's line between the box's
    edges and steps inward to the box's inner edge. Returns an (N, 2) [x, y] array.
    """
    q = np.asarray(quad, float)

    def x_on(p, s, y):  # x of side p->s at row y
        return p[0] + (s[0] - p[0]) * (y - p[1]) / (s[1] - p[1])

    def y_on(p, s, x):
        return p[1] + (s[1] - p[1]) * (x - p[0]) / (s[0] - p[0])

    cy, cx = q[:, 1].mean(), q[:, 0].mean()
    sides = {"top": [], "right": [], "bottom": [], "left": []}
    for r0, r1, c0, c1 in boxes:
        d = {"top": r0 - q[[0, 1], 1].max(), "bottom": q[[2, 3], 1].min() - r1,
             "left": c0 - q[[0, 3], 0].max(), "right": q[[1, 2], 0].min() - c1}
        sides[min(d, key=d.get)].append((r0, r1, c0, c1))
    pts = [q[0]]
    for r0, r1, c0, c1 in sorted(sides["top"], key=lambda b: b[2]):
        pts += [(c0, y_on(q[0], q[1], c0)), (c0, r1), (c1, r1), (c1, y_on(q[0], q[1], c1))]
    pts.append(q[1])
    for r0, r1, c0, c1 in sorted(sides["right"], key=lambda b: b[0]):
        pts += [(x_on(q[1], q[2], r0), r0), (c0, r0), (c0, r1), (x_on(q[1], q[2], r1), r1)]
    pts.append(q[2])
    for r0, r1, c0, c1 in sorted(sides["bottom"], key=lambda b: -b[3]):
        pts += [(c1, y_on(q[3], q[2], c1)), (c1, r0), (c0, r0), (c0, y_on(q[3], q[2], c0))]
    pts.append(q[3])
    for r0, r1, c0, c1 in sorted(sides["left"], key=lambda b: -b[1]):
        pts += [(x_on(q[0], q[3], r1), r1), (c1, r1), (c1, r0), (x_on(q[0], q[3], r0), r0)]
    return np.array(pts, float)  # unrounded: rounding could push a slanted side outward


CORRECTED_FILE = DATA_DIR / "roi_corrected.json"


def load_corrected_polygon(name):
    """Agent-verified inner polygon (wire and clips excluded); see data/roi_corrected.json."""
    d = json.loads(CORRECTED_FILE.read_text())
    return np.array(d["images"][name]["polygon"], dtype=float)


def load_corrected(name):
    """Full data/roi_corrected.json entry for an image (quad, notch boxes, polygon, notes)."""
    return json.loads(CORRECTED_FILE.read_text())["images"][name]


def distance_to_edge(points_rc, polygon):
    """Signed distance (px) from (row, col) points to a polygon edge; positive inside.

    cv2.pointPolygonTest with measureDist=True returns the signed Euclidean
    distance from a point to the nearest polygon edge.
    """
    import cv2

    cnt = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(cnt, (float(c), float(r)), True) for r, c in np.asarray(points_rc)])
