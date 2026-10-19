"""One-to-one matching of detections to ground truth, and count/detection metrics.

Coordinates are (row, col) in ORIGINAL image pixels throughout; nothing here
uses (x, y). A detection and a ground-truth centroid may be paired only if
their Euclidean distance is <= radius (inclusive). The gate is part of the
assignment problem: among all pairings that use only allowed pairs, choose
one with the most pairs, and among those the smallest total distance. This
differs from running unrestricted Hungarian assignment and discarding far
pairs afterwards, which can throw away pairs that a gated assignment keeps.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .truth import equivalent_diameter


def match_radius(gt_areas_in_roi, factor=0.5):
    """Gate radius: factor x median equivalent plate diameter of the ROI GT."""
    return float(factor * np.median(equivalent_diameter(gt_areas_in_roi)))


def _solve_block(d, radius):
    """Max-cardinality, then min-total-distance matching on one dense block.

    Allowed pairs get cost d - BIG and forbidden pairs cost 0, which is the
    same as leaving both unmatched. BIG exceeds any possible sum of allowed
    distances, so one extra pair always outweighs any distance saving:
    minimising the total first maximises the pair count, then minimises the
    distance. scipy's linear_sum_assignment (Hungarian-type solver) handles
    rectangular blocks; pairs it assigns at cost 0 are discarded as dummies.
    """
    ok = d <= radius
    big = radius * (min(d.shape) + 1) + 1.0
    cost = np.where(ok, d - big, 0.0)
    r, c = linear_sum_assignment(cost)
    keep = ok[r, c]
    return r[keep], c[keep]


def match(pred_rc, gt_rc, radius):
    """Gated one-to-one matching. Returns (pred_idx, gt_idx, distances).

    cKDTree.query_ball_point lists all GT within `radius` of each detection;
    the allowed-pair graph is split into connected components
    (scipy.sparse.csgraph.connected_components) and each component is solved
    exactly, which gives the same optimum as one global solve.
    """
    pred = np.asarray(pred_rc, float).reshape(-1, 2)
    gt = np.asarray(gt_rc, float).reshape(-1, 2)
    empty = (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    if len(pred) == 0 or len(gt) == 0:
        return empty
    # small epsilon so the inclusive gate is not lost to float rounding in the tree
    nbrs = cKDTree(gt).query_ball_point(pred, radius + 1e-9)
    pi = np.repeat(np.arange(len(pred)), [len(n) for n in nbrs])
    gi = np.fromiter((j for n in nbrs for j in n), int, count=len(pi))
    if len(pi) == 0:
        return empty
    n_p = len(pred)
    graph = coo_matrix((np.ones(len(pi)), (pi, n_p + gi)), shape=(n_p + len(gt),) * 2)
    _, comp = connected_components(graph, directed=False)
    out_p, out_g = [], []
    for k in np.unique(comp[pi]):
        ps = np.nonzero(comp[:n_p] == k)[0]
        gs = np.nonzero(comp[n_p:] == k)[0]
        d = np.hypot(*(pred[ps, None, :] - gt[None, gs, :]).transpose(2, 0, 1))
        r, c = _solve_block(d, radius)
        out_p.append(ps[r])
        out_g.append(gs[c])
    p_idx = np.concatenate(out_p)
    g_idx = np.concatenate(out_g)
    order = np.argsort(p_idx)
    p_idx, g_idx = p_idx[order], g_idx[order]
    return p_idx, g_idx, np.hypot(*(pred[p_idx] - gt[g_idx]).T)


def metrics(n_pred, n_gt, n_tp):
    """Count and detection metrics from the three totals."""
    fp, fn = n_pred - n_tp, n_gt - n_tp
    prec = n_tp / n_pred if n_pred else float("nan")
    rec = n_tp / n_gt if n_gt else float("nan")
    f1 = 2 * n_tp / (n_pred + n_gt) if n_pred + n_gt else float("nan")
    return {
        "predicted": int(n_pred), "ground truth": int(n_gt),
        "signed count error %": 100 * (n_pred - n_gt) / n_gt if n_gt else float("nan"),
        "TP": int(n_tp), "FP": int(fp), "FN": int(fn),
        "precision": prec, "recall": rec, "F1": f1,
    }


def evaluate(pred_rc, gt_rc, radius):
    """Match, then return (metrics dict, match result)."""
    p, g, d = match(pred_rc, gt_rc, radius)
    return metrics(len(pred_rc), len(gt_rc), len(p)), (p, g, d)


def evaluate_excluding(pred_rc, gt_rc, radius, exclude_gt):
    """Metrics with some GT instances (template exemplars) removed fairly.

    Protocol: (1) match the full prediction set to the full GT set; (2) remove
    each excluded GT instance together with the detection matched to it, if
    any; (3) also remove unmatched detections within `radius` of an excluded
    GT centroid, since they sit on an exemplar and would otherwise count as
    false positives only because their GT was deleted; (4) report metrics on
    what remains, using the step-1 matches. Simply deleting exemplar GT and
    re-matching would score the exemplar detections as false positives.
    Returns (metrics dict, number of detections removed).
    """
    pred = np.asarray(pred_rc, float).reshape(-1, 2)
    gt = np.asarray(gt_rc, float).reshape(-1, 2)
    exclude_gt = np.asarray(sorted(set(int(i) for i in exclude_gt)), int)
    p, g, _ = match(pred, gt, radius)
    drop_p = np.zeros(len(pred), bool)
    drop_p[p[np.isin(g, exclude_gt)]] = True
    unmatched = np.ones(len(pred), bool)
    unmatched[p] = False
    if len(exclude_gt) and len(pred):
        near = cKDTree(gt[exclude_gt]).query_ball_point(pred, radius + 1e-9)
        drop_p |= unmatched & np.array([len(n) > 0 for n in near])
    keep_gt = np.ones(len(gt), bool)
    keep_gt[exclude_gt] = False
    tp = int(np.sum(~np.isin(g, exclude_gt)))
    return metrics(int((~drop_p).sum()), int(keep_gt.sum()), tp), int(drop_p.sum())


def precision_curve(pred_rc, scores, gt_rc, radius, steps=None):
    """Cumulative precision and recall when accepting the top-k detections.

    For each prefix size k (detections sorted by descending score), the
    accepted set is re-matched from scratch one-to-one against all GT, so a
    lower-ranked detection can never take credit that a higher one would get
    within that prefix. Prefixes never split detections with equal scores.
    Returns a dict of arrays: k, score threshold (score of
    the k-th detection), TP, precision, recall.
    """
    pred = np.asarray(pred_rc, float).reshape(-1, 2)
    order = np.argsort(-np.asarray(scores), kind="stable")
    pred, sc = pred[order], np.asarray(scores)[order]
    n = len(pred)
    if n == 0:
        z = np.zeros(0)
        return {"k": np.zeros(0, int), "threshold": z, "TP": np.zeros(0, int),
                "precision": z, "recall": z}
    ks = np.unique(np.r_[np.arange(1, n + 1, max(1, n // 200)), n]) if steps is None else np.asarray(steps)
    # extend each prefix to the end of its tie group, so "score >= threshold"
    # describes exactly the accepted set
    last_of_value = np.searchsorted(-sc, -sc[ks - 1], side="right")
    ks = np.unique(last_of_value)
    tp = np.array([len(match(pred[:k], gt_rc, radius)[0]) for k in ks])
    return {"k": ks, "threshold": sc[ks - 1], "TP": tp,
            "precision": tp / ks, "recall": tp / len(gt_rc)}


def random_baseline(roi, n, gt_rc, radius, seed=0):
    """Metrics for n points drawn uniformly at random from ROI pixels.

    A floor for comparison: how much a detector beats placing points blindly.
    """
    rng = np.random.default_rng(seed)
    rr, cc = np.nonzero(roi)
    pick = rng.choice(len(rr), size=n, replace=False)
    return evaluate(np.c_[rr[pick], cc[pick]], gt_rc, radius)[0]
