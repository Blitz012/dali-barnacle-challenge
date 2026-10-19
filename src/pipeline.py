"""Frozen detection pipeline: settings chosen on img1 only, then reused unchanged.

data/frozen_params.json holds every value chosen during development (match
radius, peak spacing, threshold, low-confidence cut, template IDs). It is
written once, from img1, before img2 is evaluated; later code only reads it.
"""
import json

import numpy as np

from . import region, templates as tm
from .dataio import DATA_DIR, load_rgb

FROZEN_FILE = DATA_DIR / "frozen_params.json"


def freeze(params):
    """Write the frozen settings. Refuses to overwrite different values."""
    if FROZEN_FILE.exists():
        old = json.loads(FROZEN_FILE.read_text())
        if old != params:
            raise RuntimeError("frozen_params.json exists with different values; "
                               "delete it deliberately if development is reopened")
        return old
    FROZEN_FILE.write_text(json.dumps(params, indent=2))
    return params


def load_frozen():
    return json.loads(FROZEN_FILE.read_text())


def check_selection(fz):
    """Load templates.json and refuse to run if it drifted from the frozen record.

    Template IDs, source image and each crop's (row, col, side) must match
    what was frozen; angles and scales are always taken from the frozen file.
    """
    sel = tm.load_selection()
    if sorted(sel["chosen_ids"]) != sorted(fz["template_ids"]):
        raise RuntimeError("templates.json chosen_ids differ from the frozen template IDs")
    if sel.get("image", "img1.png") != fz["template_source"]:
        raise RuntimeError("template source image differs from the frozen one")
    if [list(sel.get("angles", [0])), list(sel.get("scales", [1.0]))] != [fz["angles"], fz["scales"]]:
        raise RuntimeError("templates.json angles/scales differ from the frozen ones")
    by_id = {c["id"]: c for c in sel["candidates"]}
    for rec in fz["template_crops"]:
        if {k: by_id[rec["id"]][k] for k in rec} != rec:
            raise RuntimeError(f"template {rec['id']} crop record differs from the frozen one")
    return sel


def frozen_variants(fz=None):
    """Template variants exactly as frozen (IDs, crops, angles, scales)."""
    fz = load_frozen() if fz is None else fz
    sel = check_selection(fz)
    return tm.build_templates(tm.gray_source(sel), sel,
                              angles=tuple(fz["angles"]), scales=tuple(fz["scales"]))


def run(name, corners, frozen=None, score_map=None):
    """Detect barnacles in one image with the frozen settings.

    A precomputed `score_map` must come from `frozen_variants`.
    Returns dict with peaks (row, col), scores, score map, roi mask and the
    low-confidence flag per detection.
    """
    fz = load_frozen() if frozen is None else frozen
    sel = check_selection(fz)
    rgb = load_rgb(name)
    g = tm.gray(rgb)
    if score_map is None:
        score_map = tm.score_map(g, frozen_variants(fz))
    sm = score_map
    roi = region.roi_mask(g.shape, corners)
    pk, sc = tm.find_peaks(sm, roi, fz["min_distance"], fz["threshold"])
    return {"rgb": rgb, "peaks": pk, "scores": sc, "score_map": sm, "roi": roi,
            "low": sc < fz["low_confidence_below"]}
