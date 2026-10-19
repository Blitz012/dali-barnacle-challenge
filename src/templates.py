"""Whole-shell template matching (decision 3, revised: option A).

A barnacle in img1 is a bright white shell around a mid-grey plate. Templates
are square crops of whole shells, centred on an annotated plate centroid.
Correlation scores are a review ranking, not a probability.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.feature import peak_local_max

from . import region
from .dataio import DATA_DIR

TEMPLATE_FILE = DATA_DIR / "templates.json"


def gray(rgb):
    return rgb2gray(rgb).astype(np.float32)


def shell_diameters(g, centres, plate_diam, max_r=40):
    """Estimate whole-shell diameter around each plate from its radial profile.

    ndi.map_coordinates samples the image at fractional (row, col) positions
    with bilinear interpolation, giving the mean brightness on rings of radius
    0..max_r. Outward from the plate edge, brightness peaks on the white shell
    then falls toward the dark gap; the shell edge is where it drops halfway
    from that peak to the minimum. NaN where no drop is found.
    """
    radii = np.arange(max_r + 1)
    theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    out = np.full(len(centres), np.nan)
    for i, ((r0, c0), dp) in enumerate(zip(centres, plate_diam)):
        rr = r0 + radii[:, None] * np.sin(theta)
        cc = c0 + radii[:, None] * np.cos(theta)
        p = ndi.map_coordinates(g, [rr, cc], order=1).mean(1)
        lo = int(np.ceil(dp / 2))
        pk = lo + int(np.argmax(p[lo:lo + 15]))
        tail = p[pk:]
        below = np.nonzero(tail < (p[pk] + tail.min()) / 2)[0]
        if len(below):
            out[i] = 2 * (pk + below[0])
    return out


def crop(g, centre, size):
    """Square crop of side `size` (odd) centred on a (row, col) point."""
    h = size // 2
    r, c = int(round(centre[0])), int(round(centre[1]))
    return g[r - h:r + h + 1, c - h:c + h + 1]


def rotation_centre(shape):
    """Pixel-grid centre (x, y) of an array: ((w-1)/2, (h-1)/2).

    For an odd side 2h+1 this is pixel (h, h), the crop's centre pixel; w/2
    would sit half a pixel off and shift every rotated copy.
    """
    h, w = shape[:2]
    return ((w - 1) / 2, (h - 1) / 2)


def circle_mask(side):
    """Float mask of the disc inscribed in a side x side square."""
    h = (side - 1) / 2
    yy, xx = np.mgrid[:side, :side]
    return (np.hypot(yy - h, xx - h) <= h).astype(np.float32)


def variants(tmpl, scales=(1.0,), angles=(0,)):
    """Rescaled and rotated copies of a template, each with its support mask.

    cv2.resize rescales with area interpolation; cv2.warpAffine with a
    rotation matrix from cv2.getRotationMatrix2D rotates about the pixel-grid
    centre. Square corners rotated by 45 degrees would need invented border
    pixels, so every variant is scored only over the inscribed disc, which a
    rotation maps onto itself: all angles compare the same real pixels.
    Returns a list of (template, mask) pairs.
    """
    out = []
    for s in scales:
        t = cv2.resize(tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else tmpl
        h, w = t.shape
        if h != w or h % 2 == 0:
            raise ValueError(f"template must be square with odd side, got {t.shape}")
        m = circle_mask(h)
        for a in angles:
            M = cv2.getRotationMatrix2D(rotation_centre(t.shape), a, 1.0)
            r = cv2.warpAffine(t, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            out.append((np.ascontiguousarray(r, np.float32), m))
    return out


def score_map(g, variants_):
    """Per-pixel best normalised correlation over all template variants.

    cv2.matchTemplate with TM_CCOEFF_NORMED and a mask computes, at each
    offset, the Pearson correlation between template and image over the mask
    pixels only (-1 to 1). Each map is placed back at image size so its value
    sits at the template's centre pixel (offset + side//2); positions where
    the template does not fit stay -1. Non-finite values (flat windows) are
    set to -1. The pixelwise maximum over variants is the score map; it does
    not by itself give one peak per barnacle (see `detect`).
    """
    best = np.full(g.shape, -1.0, np.float32)
    for t, m in variants_:
        h, w = t.shape
        r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED, mask=m)
        r[~np.isfinite(r)] = -1.0
        full = np.full(g.shape, -1.0, np.float32)
        full[h // 2:h // 2 + r.shape[0], w // 2:w // 2 + r.shape[1]] = r
        np.maximum(best, full, out=best)
    return best


def find_peaks(sm, roi, min_distance, threshold):
    """Local maxima of the score map inside the ROI, at least min_distance apart.

    The map is set to -1 outside the ROI first, so an outside-frame peak can
    never suppress an inside-frame one. skimage.feature.peak_local_max keeps
    maxima of a (2*min_distance+1)-square neighbourhood above `threshold`, then
    drops any weaker peak within min_distance (Chebyshev) of a stronger one;
    this is the spatial suppression across templates and rotations.
    Returns (row, col) int array sorted by descending score, and the scores.
    """
    s = np.where(roi, sm, -1.0)
    pk = peak_local_max(s, min_distance=min_distance, threshold_abs=threshold,
                        exclude_border=False)
    pk = pk[roi[pk[:, 0], pk[:, 1]]]
    sc = s[pk[:, 0], pk[:, 1]]
    order = np.argsort(-sc, kind="stable")
    return pk[order], sc[order]


def build_templates(g, selection, angles=None, scales=None):
    """Variants for the chosen gallery IDs, cropped at their recorded side."""
    angles = tuple(selection.get("angles", (0,))) if angles is None else angles
    scales = tuple(selection.get("scales", (1.0,))) if scales is None else scales
    by_id = {c["id"]: c for c in selection["candidates"]}
    out = []
    for i in selection["chosen_ids"]:
        c = by_id[i]
        t = crop(g, (c["row"], c["col"]), c["side"])
        out += variants(t, scales, angles)
    return out


def detect(rgb, selection, corners, min_distance, threshold):
    """Frozen detector: score map from the chosen templates, peaks inside ROI.

    Returns (peaks (row, col), scores, score map).
    """
    g = gray(rgb)
    sm = score_map(g, build_templates(gray_source(selection), selection))
    roi = region.roi_mask(g.shape, corners)
    pk, sc = find_peaks(sm, roi, min_distance, threshold)
    return pk, sc, sm


_SOURCE_CACHE = {}


def gray_source(selection):
    """Greyscale of the image the templates were cut from (img1)."""
    from .dataio import load_rgb
    name = selection.get("image", "img1.png")
    if name not in _SOURCE_CACHE:
        _SOURCE_CACHE[name] = gray(load_rgb(name))
    return _SOURCE_CACHE[name]


SELECTION_CRITERIA = [
    "exactly one annotated plate centroid inside the estimated shell circle",
    "a visible whole shell, contained in the crop with a margin",
    "limited neighbouring material (other plates) inside the crop",
]


def candidate_table(g, centres, plate_diam, shell_diam, interiors_lab, ids,
                    targets=(25, 33, 41), tol=3, margin=4):
    """Rows describing every annotated barnacle eligible as a template.

    Eligible: shell diameter within `tol` px of a target size, and exactly one
    plate centroid inside its shell circle. The crop side is the target plus
    2*margin, enlarged when that shell is larger, and made odd so the plate
    centroid is the centre pixel. `neighbour_frac` is the share of crop pixels
    belonging to other annotated plates (shown, not filtered on).
    """
    rows = []
    for k, ((r, c), dp, ds, i) in enumerate(zip(centres, plate_diam, shell_diam, ids)):
        if not np.isfinite(ds):
            continue
        group = next((t for t in targets if abs(ds - t) <= tol), None)
        if group is None:
            continue
        others = np.hypot(*(centres - (r, c)).T) < ds / 2
        if others.sum() != 1:
            continue
        side = int(max(group, ds) + 2 * margin) | 1
        lab = crop(interiors_lab, (r, c), side)
        if lab.shape != (side, side):
            continue
        own = interiors_lab[int(round(r)), int(round(c))]
        neigh = ((lab > 0) & (lab != own)).mean()
        rows.append(dict(id=int(i), group=group, row=float(r), col=float(c),
                         plate_d=float(dp), shell_d=float(ds), side=side,
                         neighbour_frac=round(float(neigh), 3)))
    return rows


def sample_candidates(rows, per_group=40, seed=0):
    """Fixed-seed random sample per size group, so the gallery is reproducible."""
    rng = np.random.default_rng(seed)
    out = []
    for gsize in sorted({r["group"] for r in rows}):
        pool = [r for r in rows if r["group"] == gsize]
        pick = rng.choice(len(pool), size=min(per_group, len(pool)), replace=False)
        out += sorted((pool[j] for j in pick), key=lambda r: r["id"])
    return out


def rotation_symmetry(g, cand):
    """Normalised correlation between a crop and its 180-degree rotation.

    1.0 means the shell looks identical turned half way; this checks the
    assumption that rotations 0-135 degrees cover all orientations.
    """
    t = crop(g, (cand["row"], cand["col"]), cand["side"])
    return float(cv2.matchTemplate(t, np.rot90(t, 2).copy(), cv2.TM_CCOEFF_NORMED)[0, 0])


def save_selection(candidates, chosen_ids, notes=""):
    """Record candidate IDs, chosen IDs and criteria in data/templates.json."""
    import json
    TEMPLATE_FILE.write_text(json.dumps({
        "source": "img1.png / mask1.png (development data)",
        "criteria": SELECTION_CRITERIA,
        "candidates": candidates,
        "chosen_ids": sorted(chosen_ids),
        "notes": notes,
    }, indent=2))


def load_selection():
    import json
    return json.loads(TEMPLATE_FILE.read_text())


def masked_corr(a, b, mask):
    """Pearson correlation of two equal-size crops over mask pixels only."""
    x, y = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def rotation_check(g, cands, angles=(90, 180), seed=0):
    """Shell-only correlation of each crop with its rotations, vs unrelated pairs.

    Restricting to the estimated shell circle removes neighbouring shells, so
    a low score reflects the shell itself (plate shape, shading direction).
    The baseline correlates each crop with a different candidate of the same
    side, measuring how alike two unrelated shells are.
    """
    rng = np.random.default_rng(seed)
    out = {a: [] for a in angles}
    out["unrelated"] = []
    for c in cands:
        t = crop(g, (c["row"], c["col"]), c["side"])
        h = c["side"] // 2
        yy, xx = np.mgrid[:c["side"], :c["side"]]
        m = np.hypot(yy - h, xx - h) <= c["shell_d"] / 2
        for a in angles:
            M = cv2.getRotationMatrix2D((h, h), a, 1.0)
            out[a].append(masked_corr(t, cv2.warpAffine(t, M, t.shape[::-1]), m))
        peers = [p for p in cands if p["side"] == c["side"] and p["id"] != c["id"]]
        if peers:
            p = peers[rng.integers(len(peers))]
            out["unrelated"].append(masked_corr(t, crop(g, (p["row"], p["col"]), p["side"]), m))
    return {k: np.median(v) for k, v in out.items()}
