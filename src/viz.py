"""Matplotlib display helpers. All inputs are RGB (never BGR)."""
import matplotlib.pyplot as plt
import numpy as np


def px_extent(r0, c0, h, w):
    """imshow extent so a crop starting at (r0, c0) keeps full-image pixel coordinates.

    Pixel (r, c) covers r-0.5..r+0.5, so scatter markers at integer (row, col)
    land on pixel centres, exactly as in an uncropped imshow.
    """
    return (c0 - 0.5, c0 + w - 0.5, r0 + h - 0.5, r0 - 0.5)


def show_grid(images, titles, ncols=3, size=5, cmap=None):
    """Show a list of RGB or 2-D images side by side."""
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, im, t in zip(np.ravel(axes), images, titles):
        ax.imshow(im, cmap=cmap if im.ndim == 2 else None)
        ax.set_title(t)
    fig.tight_layout()
    return fig


def overlay_outline(rgb, outline, color=(255, 0, 255)):
    """Paint outline pixels onto a copy of an RGB image."""
    out = rgb.copy()
    out[outline] = color
    return out


def show_region(rgb, corners, title="", zoom=90):
    """Full image with the evaluation polygon, plus a zoom on each corner."""
    closed = np.vstack([corners, corners[:1]])
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(rgb)
    ax.plot(closed[:, 0], closed[:, 1], "-", color="magenta", lw=1.5)
    ax.set_title(title)
    ax.axis("off")
    for i, (x, y) in enumerate(corners.astype(int)):
        a = fig.add_subplot(2, 4, [3, 4, 7, 8][i])
        r0, c0 = max(y - zoom, 0), max(x - zoom, 0)
        view = rgb[r0:y + zoom, c0:x + zoom]  # clipped at the image edge
        a.imshow(view, extent=px_extent(r0, c0, *view.shape[:2]))
        a.plot(closed[:, 0], closed[:, 1], "-", color="magenta", lw=1.5)
        a.plot(x, y, "+", color="yellow", ms=14, mew=2)
        a.set_xlim(x - zoom - 0.5, x + zoom - 0.5)
        a.set_ylim(y + zoom - 0.5, y - zoom - 0.5)
        a.set_title(f"corner {i}: ({x}, {y})", fontsize=9)
        a.axis("off")
    fig.tight_layout()
    return fig


def show_points(rgb, pts_rc, crops, title="", color="cyan", size=6, outline=None):
    """Full image with points, plus zoomed crops (r0, c0, side) to inspect them.

    If `outline` is given, mask pixels are painted magenta under the points.
    """
    base = overlay_outline(rgb, outline) if outline is not None else rgb
    pts = np.asarray(pts_rc)
    fig, axes = plt.subplots(1, 1 + len(crops), figsize=(6 * (1 + len(crops)), 6))
    axes[0].imshow(base)
    axes[0].scatter(pts[:, 1], pts[:, 0], s=1, c=color)
    for r0, c0, side in crops:
        axes[0].add_patch(plt.Rectangle((c0, r0), side, side, fill=False, ec="yellow", lw=1.5))
    axes[0].set_title(title)
    for ax, (r0, c0, side) in zip(axes[1:], crops):
        ax.imshow(base[r0:r0 + side, c0:c0 + side], extent=px_extent(r0, c0, side, side))
        sel = (pts[:, 0] >= r0) & (pts[:, 0] < r0 + side) & (pts[:, 1] >= c0) & (pts[:, 1] < c0 + side)
        ax.scatter(pts[sel, 1], pts[sel, 0], s=size * 4, c=color, marker="+")
        ax.set_title(f"rows {r0}-{r0 + side}, cols {c0}-{c0 + side}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def histogram(values, bins, xlabel, title="", marks=()):
    """Histogram with optional dashed vertical reference lines."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bins, color="#4a7fb0", edgecolor="white")
    for x, lbl in marks:
        ax.axvline(x, ls="--", color="#c0504d")
        ax.text(x, ax.get_ylim()[1] * 0.95, f" {lbl}", color="#c0504d", va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def show_circles(rgb, centres, diameters, crops, title="", color="cyan"):
    """Zoomed crops (r0, c0, side) with a circle of given diameter at each centre."""
    fig, axes = plt.subplots(1, len(crops), figsize=(6 * len(crops), 6))
    for ax, (r0, c0, side) in zip(np.atleast_1d(axes), crops):
        ax.imshow(rgb[r0:r0 + side, c0:c0 + side], extent=px_extent(r0, c0, side, side))
        for (r, c), d in zip(centres, diameters):
            if r0 <= r < r0 + side and c0 <= c < c0 + side and np.isfinite(d):
                ax.add_patch(plt.Circle((c, r), d / 2, fill=False, ec=color, lw=1))
        ax.set_xlim(c0 - 0.5, c0 + side - 0.5)
        ax.set_ylim(r0 + side - 0.5, r0 - 0.5)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def show_gallery(g, cands, crop_fn, ncols=10, size=1.6):
    """Grid of candidate crops labelled with ID and neighbour fraction.

    A dashed circle marks the estimated shell so containment can be checked.
    """
    n = len(cands)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * 1.15 * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, c in zip(np.ravel(axes), cands):
        t = crop_fn(g, (c["row"], c["col"]), c["side"])
        ax.imshow(t, cmap="gray", vmin=0, vmax=1)
        h = c["side"] // 2
        ax.add_patch(plt.Circle((h, h), c["shell_d"] / 2, fill=False, ec="cyan", ls="--", lw=0.8))
        ax.set_title(f"#{c['id']}  n={c['neighbour_frac']:.2f}", fontsize=7)
    fig.tight_layout()
    return fig


MATCH_COLOURS = {"TP": "#00e05a", "FP": "#ff2d2d", "FN": "#ffd400"}


def _crop_axes(ax, r0, c0, side):
    ax.set_xlim(c0 - 0.5, c0 + side - 0.5)
    ax.set_ylim(r0 + side - 0.5, r0 - 0.5)
    ax.axis("off")


def show_thresholds(rgb, results, crop, gt_rc=None, ms=40):
    """One zoomed panel per (title, peaks) result, on the same crop (r0, c0, side).

    Ground-truth centroids, if given, are small white dots for reference.
    """
    r0, c0, side = crop
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))
    for ax, (title, pk) in zip(np.atleast_1d(axes), results):
        ax.imshow(rgb[r0:r0 + side, c0:c0 + side], extent=px_extent(r0, c0, side, side))
        if gt_rc is not None:
            ax.scatter(gt_rc[:, 1], gt_rc[:, 0], s=6, c="white", label="GT centroid")
        ax.scatter(pk[:, 1], pk[:, 0], s=ms, facecolors="none", edgecolors="cyan", lw=1.2, label="detection")
        ax.set_title(title)
        _crop_axes(ax, r0, c0, side)
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def match_overlay(rgb, pred_rc, gt_rc, p_idx, g_idx, corners, crops=(), title=""):
    """Full image and zooms coloured by outcome: TP detections, FP detections, missed GT."""
    pred, gt = np.asarray(pred_rc), np.asarray(gt_rc)
    tp = np.zeros(len(pred), bool)
    tp[p_idx] = True
    fn = np.ones(len(gt), bool)
    fn[g_idx] = False
    groups = [("TP", pred[tp], "o"), ("FP", pred[~tp], "x"), ("FN", gt[fn], "s")]
    closed = np.vstack([corners, corners[:1]])
    fig, axes = plt.subplots(1, 1 + len(crops), figsize=(7 * (1 + len(crops)), 7))
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        ax.imshow(rgb)
        ax.plot(closed[:, 0], closed[:, 1], "-", color="magenta", lw=1)
        for name, pts, mk in groups:
            lbl = f"{name} ({len(pts)})" if name != "FN" else f"missed GT ({len(pts)})"
            s = 3 if k == 0 else 36
            if mk == "x" or k == 0:
                ax.scatter(pts[:, 1], pts[:, 0], s=s, c=MATCH_COLOURS[name], marker=mk, lw=1.2, label=lbl)
            else:
                ax.scatter(pts[:, 1], pts[:, 0], s=s, facecolors="none", edgecolors=MATCH_COLOURS[name],
                           marker=mk, lw=1.4, label=lbl)
        if k == 0:
            for r0, c0, side in crops:
                ax.add_patch(plt.Rectangle((c0, r0), side, side, fill=False, ec="white", lw=1.5))
            ax.set_title(title)
            ax.axis("off")
        else:
            r0, c0, side = crops[k - 1]
            ax.set_title(f"rows {r0}-{r0 + side}, cols {c0}-{c0 + side}")
            _crop_axes(ax, r0, c0, side)
    axes[-1].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


def precision_curves(curves, ref_precision=None):
    """Cumulative precision (and recall) versus number of accepted detections."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for (label, c), col in zip(curves.items(), ["#4a7fb0", "#c0504d", "#6a9a3a"]):
        axes[0].plot(c["k"], c["precision"], color=col, label=label)
        axes[1].plot(c["threshold"], c["precision"], color=col, label=f"{label}: precision")
        axes[1].plot(c["threshold"], c["recall"], color=col, ls="--", label=f"{label}: recall")
    axes[0].set_xlabel("accepted detections (highest scores first)")
    axes[0].set_ylabel("cumulative precision")
    axes[1].set_xlabel("score threshold (accept score >= t)")
    axes[1].invert_xaxis()
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def show_tiles(rgb, table, peaks_rc, low, roi, title=""):
    """Panels of the given tiles with high- and low-confidence detections."""
    pk = np.asarray(peaks_rc).reshape(-1, 2)
    n = len(table)
    if n == 0:
        print(f"{title}: no eligible tiles to show")
        return None
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.4))
    for ax, (_, t) in zip(np.atleast_1d(axes), table.iterrows()):
        r0, r1, c0, c1 = int(t.row0), int(t.row1), int(t.col0), int(t.col1)
        view = rgb[r0:r1, c0:c1].copy()
        view[~roi[r0:r1, c0:c1]] //= 3
        ax.imshow(view, extent=px_extent(r0, c0, r1 - r0, c1 - c0))
        sel = (pk[:, 0] >= r0) & (pk[:, 0] < r1) & (pk[:, 1] >= c0) & (pk[:, 1] < c1)
        ax.scatter(pk[sel & ~low, 1], pk[sel & ~low, 0], s=14, c="cyan", label="score >= cut")
        ax.scatter(pk[sel & low, 1], pk[sel & low, 0], s=14, c="#ff8c00", label="low confidence")
        ax.set_title(f"{t.tile}: {t.low_conf}/{t.detections} low ({t.low_share:.0%})", fontsize=9)
        ax.axis("off")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def tile_grid(rgb, table, roi, title=""):
    """ROI with the tile grid; each tile labelled with its low-confidence share or status."""
    fig, ax = plt.subplots(figsize=(8, 8))
    r0, r1 = table.row0.min(), table.row1.max()
    c0, c1 = table.col0.min(), table.col1.max()
    view = rgb[r0:r1, c0:c1].copy()
    view[~roi[r0:r1, c0:c1]] //= 3
    ax.imshow(view, extent=px_extent(r0, c0, r1 - r0, c1 - c0))
    for _, t in table.iterrows():
        ax.add_patch(plt.Rectangle((t.col0, t.row0), t.col1 - t.col0, t.row1 - t.row0,
                                   fill=False, ec="white", lw=1))
        txt = (f"{t.low_share:.0%}\n{t.low_conf}/{t.detections}" if t.status == "eligible"
               else ("empty\n(undefined)" if t.status == "empty" else f"{t.status}\n{t.low_conf}/{t.detections}"))
        ax.text((t.col0 + t.col1) / 2, (t.row0 + t.row1) / 2, txt, ha="center", va="center",
                color="white", fontsize=9, bbox=dict(fc="black", alpha=0.55, lw=0))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def show_patches(rgb, pts_rc, titles, half=30, ncols=6, mark=None):
    """Square patches of side 2*half+1 around (row, col) points, centre marked."""
    pts = np.asarray(pts_rc).astype(int)
    n = len(pts)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (r, c), t in zip(np.ravel(axes), pts, titles):
        ax.imshow(rgb[max(r - half, 0):r + half + 1, max(c - half, 0):c + half + 1])
        ax.plot(min(half, c), min(half, r), "+", color=mark or "cyan", ms=12, mew=1.5)
        ax.set_title(t, fontsize=8)
    fig.tight_layout()
    return fig


ROI_STYLES = [("yellow", "-"), ("magenta", "--"), ("cyan", ":"), ("orange", "-.")]


def _draw_polys(ax, polys, lw=1.5):
    for (label, p), (c, ls) in zip(polys.items(), ROI_STYLES):
        p = np.asarray(p, float)
        closed = np.vstack([p, p[:1]])
        ax.plot(closed[:, 0], closed[:, 1], ls, color=c, lw=lw, label=label)


def roi_whole(rgb, polys, title=""):
    """Whole image with one or more labelled ROI polygons ([x, y] vertex arrays)."""
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.imshow(rgb)
    _draw_polys(ax, polys, lw=1.2)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def roi_zooms(rgb, polys, centres_xy, labels, half=100, ncols=4, title=""):
    """Zooms (2*half px square) at [x, y] centres with ROI polygons and a 20 px grid."""
    nrows = int(np.ceil(len(centres_xy) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows + 0.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(centres_xy):]:
        ax.axis("off")
    for ax, (x, y), lab in zip(axes, np.asarray(centres_xy).astype(int), labels):
        r0, c0 = max(y - half, 0), max(x - half, 0)
        view = rgb[r0:r0 + 2 * half, c0:c0 + 2 * half]
        h, w = view.shape[:2]
        ax.imshow(view, extent=px_extent(r0, c0, h, w), interpolation="nearest")
        _draw_polys(ax, polys)
        ax.set_xlim(c0 - 0.5, c0 + w - 0.5)
        ax.set_ylim(r0 + h - 0.5, r0 - 0.5)
        ax.set_xticks(np.arange((c0 // 20 + 1) * 20, c0 + w, 20), minor=True)
        ax.set_yticks(np.arange((r0 // 20 + 1) * 20, r0 + h, 20), minor=True)
        ax.grid(which="minor", color="white", alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{lab}  (x {x}, y {y})", fontsize=9)
    axes[0].legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def roi_figures(rgb, polys, entry, stem, outdir):
    """Save ROI verification figures; return {kind: path}.

    `polys` maps labels to [x, y] polygons, the first being the one checked.
    `entry` is the data/roi_corrected.json record (quad corners, notch boxes).
    Writes whole-frame, four-corner (200 px), notch (140 px) and straight-edge
    (120 px, at 1/6, 1/2 and 5/6 of each side) views.
    """
    q = np.asarray(entry["quad"], float)
    edges, elab = [], []
    for i, s in enumerate(["top", "right", "bottom", "left"]):
        a, b = q[i], q[(i + 1) % 4]
        for t in (1 / 6, 1 / 2, 5 / 6):
            edges.append(a + (b - a) * t)
            elab.append(f"{s} edge t={t:.2f}")
    boxes = entry["notch_boxes_row0_row1_col0_col1"]
    cen = [((c0 + c1) / 2, (r0 + r1) / 2) for r0, r1, c0, c1 in boxes]
    name = stem + ".png"
    figs = {
        "whole": (roi_whole(rgb, polys, f"{name}: corrected inner ROI vs earlier ROI"), 90),
        "corners": (roi_zooms(rgb, polys, q, ["top-left", "top-right", "bottom-right", "bottom-left"],
                              half=100, ncols=2, title=f"{name}: four corners (200 px zooms)"), 90),
        "notches": (roi_zooms(rgb, polys, cen, [f"notch {i}" for i in range(len(cen))], half=70,
                              title=f"{name}: clip/tab notches (140 px zooms)"), 80),
        "edges": (roi_zooms(rgb, polys, edges, elab, half=60, ncols=6,
                            title=f"{name}: straight-edge spot checks (120 px zooms)"), 70),
    }
    paths = {}
    for kind, (fig, dpi) in figs.items():
        paths[kind] = f"{outdir}/roi_{stem}_{kind}.png"
        fig.savefig(paths[kind], dpi=dpi)
        if kind != "corners":
            plt.close(fig)
    return paths


def inline_formats(dpi=72, jpeg_quality=85):
    """Show photo figures inline as JPEG and plots as PNG, keeping the notebook small.

    A figure counts as a photo if any axes holds an imshow image. IPython keeps
    one formatter per output type; a formatter that returns None leaves that
    type out, so each figure is stored once, in one format. Saved files in
    figures/ are unaffected.
    """
    from IPython import get_ipython
    from IPython.core.pylabtools import print_figure
    from matplotlib.figure import Figure

    def is_photo(fig):
        return any(ax.get_images() for ax in fig.axes)

    def png(fig):
        return None if is_photo(fig) else print_figure(fig, "png", dpi=dpi)

    def jpeg(fig):
        if not is_photo(fig):
            return None
        return print_figure(fig, "jpeg", dpi=dpi, pil_kwargs={"quality": jpeg_quality})

    formatters = get_ipython().display_formatter.formatters
    formatters["image/png"].for_type(Figure, png)
    formatters["image/jpeg"].for_type(Figure, jpeg)
