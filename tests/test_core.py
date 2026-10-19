"""Checks for matching, metrics, template rotation and peak suppression.

Run: .venv/bin/python -m tests.test_core   (no pytest needed)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from src import evaluate as ev
from src import templates as tm
from src import region


def test_empty_inputs():
    for p, g in [([], [[1, 1]]), ([[1, 1]], []), ([], [])]:
        pi, gi, d = ev.match(p, g, 5)
        assert len(pi) == len(gi) == len(d) == 0
    m = ev.metrics(0, 3, 0)
    assert m["recall"] == 0 and np.isnan(m["precision"]) and m["FN"] == 3


def test_threshold_boundary_inclusive():
    # 3-4-5 triangle: distance exactly 5
    assert len(ev.match([[0, 0]], [[3, 4]], 5.0)[0]) == 1
    assert len(ev.match([[0, 0]], [[3, 4]], 4.999)[0]) == 0


def test_duplicates_one_to_one():
    # two detections on one GT: one TP, one FP; two identical GT, one detection: one TP
    p, g, _ = ev.match([[10, 10], [10, 11]], [[10, 10]], 3)
    assert len(p) == 1 and p[0] == 0
    p, g, _ = ev.match([[10, 10]], [[10, 10], [10, 10]], 3)
    assert len(p) == 1
    m, _ = ev.evaluate([[10, 10], [10, 11]], [[10, 10]], 3)
    assert (m["TP"], m["FP"], m["FN"]) == (1, 1, 0)


def test_counterexample_post_filtered_hungarian():
    # P0 is near both G0 (0) and G1 (4.24); P1 is near only G0 (4.9), radius 5.
    # Unrestricted min-distance assignment prefers P0-G0 + P1-G1 (0 + 8.45)
    # over P0-G1 + P1-G0 (9.14); post-filtering then keeps one pair.
    # Gated matching keeps two.
    pred = np.array([[0, 0], [0, -4.9]])
    gt = np.array([[0, 0], [3, 3]])
    r = 5.0
    d = np.hypot(*(pred[:, None] - gt[None]).transpose(2, 0, 1))
    a, b = linear_sum_assignment(d)
    naive = int((d[a, b] <= r).sum())
    gated = len(ev.match(pred, gt, r)[0])
    assert naive == 1 and gated == 2, (naive, gated)


def test_min_distance_among_max_matchings():
    # two valid perfect matchings; the one with smaller total distance wins
    pred = [[0, 0], [0, 2]]
    gt = [[0, 0.5], [0, 2.5]]
    p, g, d = ev.match(pred, gt, 3)
    assert list(zip(p, g)) == [(0, 0), (1, 1)]
    assert np.isclose(d.sum(), 1.0)


def test_row_col_convention():
    # (row, col) points; distance is symmetric so check against roi masks instead
    corners = np.array([[10, 0], [20, 0], [20, 5], [10, 5]], float)  # x 10-20, y 0-5
    assert region.inside([[2, 15]], corners)[0]       # row 2, col 15
    assert not region.inside([[15, 2]], corners)[0]   # swapped
    m = region.roi_mask((30, 30), corners)
    assert m[2, 15] and not m[15, 2]


def test_exclusion_protocol():
    gt = [[0, 0], [100, 100], [200, 200]]
    pred = [[0, 1], [0, 2], [100, 101], [300, 300]]  # second det duplicates exemplar 0
    m, dropped = ev.evaluate_excluding(pred, gt, 5, exclude_gt=[0])
    assert dropped == 2
    assert (m["predicted"], m["ground truth"], m["TP"], m["FP"], m["FN"]) == (2, 2, 1, 1, 1)


def test_prefix_curve_rematches():
    gt = [[0, 0]]
    pred = [[0, 1], [0, 0]]
    c = ev.precision_curve(pred, [0.9, 0.5], gt, 3)
    assert list(c["TP"]) == [1, 1] and np.allclose(c["precision"], [1, 0.5])


def test_prefix_curve_empty():
    c = ev.precision_curve([], [], [[0, 0]], 5)
    assert len(c["k"]) == 0 and len(c["precision"]) == 0


def test_prefix_curve_ties():
    # equal scores: the threshold 0.5 accepts both, so no prefix of size 1 is reported
    c = ev.precision_curve([[0, 0], [100, 100]], [0.5, 0.5], [[0, 0]], 1)
    assert list(c["k"]) == [2] and np.allclose(c["precision"], [0.5])


def test_empty_outline_mask():
    from src import truth
    assert truth.unclosed_outline_hulls(np.zeros((10, 10), bool)) == []
    assert truth.compare_closing(np.zeros((10, 10), bool), 3)["holes"] == 0


def test_frozen_selection_drift_detected():
    import copy
    from src import pipeline
    fz = pipeline.load_frozen()
    pipeline.check_selection(fz)  # current files agree
    bad = copy.deepcopy(fz)
    bad["template_crops"][0]["side"] += 2
    bad_angles = copy.deepcopy(fz)
    bad_angles["angles"] = [0]
    for b in (bad, bad_angles):
        try:
            pipeline.check_selection(b)
        except RuntimeError:
            continue
        raise AssertionError("drift from the frozen record was not detected")


def test_rotation_centre_and_variants():
    t = np.zeros((9, 9), np.float32)
    t[4, 4] = 1.0
    t[4, 6] = 0.5   # off-centre marker to see rotation direction
    for v, m in tm.variants(t, angles=(0, 45, 90, 135, 180, 225, 270, 315)):
        assert v.shape == (9, 9)
        assert np.argmax(v) == 4 * 9 + 4, "centre pixel moved under rotation"
    v180 = tm.variants(t, angles=(180,))[0][0]
    assert np.isclose(v180[4, 2], 0.5) and np.isclose(v180[4, 4], 1.0)
    m = tm.circle_mask(9)
    assert m[4, 0] == 1 and m[0, 0] == 0


def test_score_map_centre_and_suppression():
    rng = np.random.default_rng(0)
    img = rng.random((80, 80)).astype(np.float32) * 0.1
    t = img[30:41, 50:61].copy()                     # 11x11 crop centred at (35, 55)
    sm = tm.score_map(img, tm.variants(t, angles=(0, 90)))
    assert np.unravel_index(np.argmax(sm), sm.shape) == (35, 55)
    # two strong maxima 3 px apart: only one survives min_distance 5
    s = np.zeros((40, 40), np.float32)
    s[20, 20], s[20, 23] = 0.9, 0.8
    roi = np.ones_like(s, bool)
    pk, sc = tm.find_peaks(s, roi, min_distance=5, threshold=0.5)
    assert pk.tolist() == [[20, 20]]
    # an outside-ROI peak must not suppress an inside one
    roi[:, :22] = False
    pk, _ = tm.find_peaks(s, roi, min_distance=5, threshold=0.5)
    assert pk.tolist() == [[20, 23]]


def test_notched_polygon():
    # 100 x 100 square with one notch on each side; notch interiors excluded, rest kept
    quad = [[0, 0], [100, 0], [100, 100], [0, 100]]
    boxes = [(-5, 10, 40, 60), (40, 60, 90, 105), (90, 105, 20, 30), (70, 80, -5, 15)]
    poly = region.notched_polygon(quad, boxes)
    assert len(poly) == 4 + 4 * len(boxes)
    out = [[5, 50], [50, 95], [97, 25], [75, 10]]   # (row, col) inside each notch
    keep = [[20, 50], [50, 80], [80, 25], [75, 20], [50, 50], [5, 5]]
    assert not region.inside(out, poly).any()
    assert region.inside(keep, poly).all()


def test_corrected_roi_file_consistent():
    # stored polygons are reproducible from quad + notch boxes and lie within the quad
    for name in ("unseen_img1.png", "img2.png"):
        e = region.load_corrected(name)
        poly = region.notched_polygon(e["quad"], e["notch_boxes_row0_row1_col0_col1"])
        assert np.array_equal(poly, np.array(e["polygon"], float)), name
        m_poly = region.roi_mask((2000, 3000), poly)
        m_quad = region.roi_mask((2000, 3000), np.array(e["quad"], float))
        assert not (m_poly & ~m_quad).any() and m_poly.sum() < m_quad.sum(), name


if __name__ == "__main__":
    n = 0
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            n += 1
            print("ok", name)
    print(f"{n} tests passed")
