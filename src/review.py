"""Human-review triage: which parts of the frame hold the least confident detections.

The ROI bounding box is cut into an n x n grid; each tile counts the
detections inside it (all detections are already inside the ROI) and the
share of them that are low-confidence. This ranks where to look first when
checking detections. It is not an estimate of missed barnacles: a tile with
no detections has an undefined share, not a good one.
"""
import numpy as np
import pandas as pd


def tile_table(peaks_rc, low, roi, n=5, min_roi_frac=0.25, min_dets=5, tp=None):
    """One row per tile of the ROI bounding box.

    status: 'empty' (no detections, share undefined), 'few' (< min_dets),
    'mostly outside' (< min_roi_frac of its pixels in the ROI) or 'eligible'.
    Only eligible tiles are ranked. `tp` (optional bool per detection) adds a
    matched-to-GT count for context; it is never used for ranking.
    """
    rr, cc = np.nonzero(roi)
    r_edges = np.linspace(rr.min(), rr.max() + 1, n + 1).round().astype(int)
    c_edges = np.linspace(cc.min(), cc.max() + 1, n + 1).round().astype(int)
    pk = np.asarray(peaks_rc).reshape(-1, 2)
    low = np.asarray(low, bool)
    ti = np.clip(np.searchsorted(r_edges, pk[:, 0], side="right") - 1, 0, n - 1)
    tj = np.clip(np.searchsorted(c_edges, pk[:, 1], side="right") - 1, 0, n - 1)
    rows = []
    for i in range(n):
        for j in range(n):
            r0, r1, c0, c1 = r_edges[i], r_edges[i + 1], c_edges[j], c_edges[j + 1]
            roi_frac = roi[r0:r1, c0:c1].mean()
            sel = (ti == i) & (tj == j)
            nd, nl = int(sel.sum()), int(low[sel].sum())
            if nd == 0:
                status = "empty"
            elif roi_frac < min_roi_frac:
                status = "mostly outside"
            elif nd < min_dets:
                status = "few"
            else:
                status = "eligible"
            row = {"tile": f"r{i}c{j}", "row0": r0, "row1": r1, "col0": c0, "col1": c1,
                   "roi_frac": round(float(roi_frac), 2), "detections": nd, "low_conf": nl,
                   "low_share": nl / nd if nd else np.nan, "status": status}
            if tp is not None:
                row["matched_to_GT"] = int(np.asarray(tp)[sel].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def worst_tiles(table, k=6):
    """The k eligible tiles with the highest low-confidence share."""
    t = table[table.status == "eligible"]
    return t.sort_values(["low_share", "low_conf"], ascending=False).head(k)


def review_queue(peaks_rc, scores, low):
    """Detections to check, least confident first."""
    pk = np.asarray(peaks_rc).reshape(-1, 2)
    df = pd.DataFrame({"row": pk[:, 0], "col": pk[:, 1], "score": scores, "low_conf": low})
    return df.sort_values("score").reset_index(drop=True)
