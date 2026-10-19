# Review log

Codex reviews read-only through the official `codex@openai-codex` plugin (`codex-companion.mjs task`, no `--write`, default model). Claude applies fixes. Each entry records the finding as reported, Claude's assessment and the outcome.

## Milestone 1: core src (templates, evaluate, region, pipeline, review, truth) and tests

Codex review, round 1 of 3.

| # | Finding (Codex) | Severity | Assessment | Outcome |
|---|---|---|---|---|
| 1 | `src/pipeline.py:40-45`: `run` checked only template IDs and built templates from the mutable `templates.json`, so its angles, scales, crop coordinates or source image could change after freezing without being detected. No evidence of actual img2-driven tuning. | Medium | Valid. The frozen file stored angles and scales, but `run` did not use them. | Fixed. Added the frozen template source and per-ID crop records (id, row, col, side) to `frozen_params.json`; this added fields only and changed no existing value. New `pipeline.check_selection` refuses on any drift, and `run` passes the frozen angles and scales explicitly. Regression test `test_frozen_selection_drift_detected`. |
| 2 | `src/evaluate.py:142`: `precision_curve` with zero detections built `ks=[0]` and raised `IndexError`. | Medium | Valid, reproduced. | Fixed: returns empty arrays. Regression test `test_prefix_curve_empty`. |
| – | Rotation centre and masked centre placement; ROI isolation in peak suppression; gated matching (BIG-cost priority, components, inclusive radius, row/col); exclusion protocol; prefix re-matching; tile logic; GT extraction and double counting. | – | Codex found no substantiated bug in any of these. It noted that both GT extraction implementations return 1,694 instances for mask1. | No change. |

Tests after the fixes: 12 of 12 pass (`.venv/bin/python -m tests.test_core`).

## Issues found by Claude during implementation

| Issue | Outcome |
|---|---|
| Template rotation used (w/2, h/2), half a pixel off the centre pixel of an odd crop. | Fixed to ((w-1)/2, (h-1)/2). Test: a 180° rotation maps (4, 6) to (4, 2) and keeps the centre pixel. |
| Rotations of 45° filled the square's corners with reflected, invented pixels. | All variants are scored over the inscribed disc with masked `TM_CCOEFF_NORMED`. Masked correlation checked against a hand computation (0.088649 vs 0.088649); a full mask reproduces unmasked output to 8e-7. |
| Peaks outside the frame could suppress peaks inside it (peaks were filtered after suppression). | The score map is set to −1 outside the ROI before `peak_local_max`. Tested. |
| The first version of the post-filtered-Hungarian counterexample test was a cost tie, so it passed by luck. | Replaced with a non-degenerate case (unrestricted costs 8.45 vs 9.14). |
| The img2 corners in `data/corners.json` sit on the wire centreline, not the inner edge; noticed because the top img2 false positives were on the wire. | `corners.json` left unchanged. A post-hoc inner-edge sensitivity row was added (F1 0.080 to 0.079); no detector setting changed. Flagged for the user. |
| The provisional unseen_img1 ROI includes a few pixels of wire (queue items 0 and 10). | Documented in the notebook at the time. Superseded by milestone 3: the agent-verified ROI excludes the wire. |

## Milestone 2: final review of src (deliverable 10)

Codex review, round 1 of 3. Areas found clean: (a) GT extraction and double counting, with all 120 candidate coordinates confirmed to match their IDs as zero-based indices into the filtered `mask_instances` output, and the notebook's exemplar `searchsorted` mapping correct; (b) no image resizing or warping, checked with a resize-call trap; (c) no img2 feedback path into frozen settings, apart from finding 1; (d) the matching gate and row/col conversions; (e) template centres and rotations. Codex could not run the full test file in its read-only sandbox (Matplotlib cache), so it ran the tests in memory and claimed no full-suite pass.

| # | Finding (Codex) | Severity | Assessment | Outcome |
|---|---|---|---|---|
| 1 | `src/pipeline.py:61`: a precomputed score map bypassed the frozen angles and scales. The notebook built `sm1` from mutable `templates.json` settings, and `check_selection` did not compare angles or scales. | Medium | Valid. No drift had actually occurred: templates.json and the frozen file both list 8 angles and scale 1.0. | Fixed: `check_selection` compares angles and scales, the new `pipeline.frozen_variants()` builds the variants, and the notebook builds `sm1` with it. Test extended with an angle-drift case. |
| 2 | `src/evaluate.py:146` / `src/viz.py:196`: prefixes could split detections with tied scores, while the plot labels the x-axis "accept score >= t". | Low | Valid; reproduced with two equal scores. | Fixed: each prefix extends to the end of its tie group. Test `test_prefix_curve_ties`. |
| 3 | `src/viz.py:63,140,215`: cropped `imshow` extents shifted pixels half a pixel against scatter markers. | Low | Valid, display only; numeric matching unaffected. | Fixed: `viz.px_extent` and half-pixel axis limits in every cropped view; same correction in `region.mark_corners`, where it also affected where clicked corners land. |
| 4 | `src/viz.py:210`: `show_tiles` crashed on an empty eligible-tile table. | Low | Valid. | Fixed: prints a message and returns None. |
| 5 | `src/truth.py:49,92`: `compare_closing` crashed on an empty outline mask. | Low | Valid; current masks unaffected. | Fixed. Test `test_empty_outline_mask`. |

Tests after the fixes: 14 of 14 pass.

Round 2 verified fixes 1, 2, 4 and 5. Fix 3 was incomplete: `src/region.py:86` built the zoom extent from the requested size even when the image edge truncated the crop. With a 100×100 image, click (95, 95) and zoom 20, the 25×25 crop spanned 40×40 units, giving wrong click coordinates. Claude fixed this by deriving the extent from the sliced view, and fixed the same pattern in `viz.show_region` with a clamped start and an extent from the view shape. A local check gave extent [76.5, 99.5, 99.5, 74.5] for a 25×23 edge crop.

Round 3 (the last allowed round) verified both edits as correct, with no new bug; Codex reported 30 in-memory geometry checks passing. No unresolved findings or disagreements remain.

## Milestone 3: ROI correction for img2 (post hoc) and unseen_img1 (operational)

Scope: Codex (parent) looked at 200 px corner zooms. It found that the existing img2 `corners.json` polygon and the earlier eye-read unseen_img1 quad touch the wire at its rounded junctions. It proposed starting quads, and Claude implemented and verified them. Per the parent's instruction, no nested Codex review was run; the parent reviews this milestone.

| # | Issue | Found by | Outcome |
|---|---|---|---|
| 1 | The ROI corners touch the wire at rounded junctions (img2, unseen_img1). | Codex (parent) | Codex-suggested quads adopted as the starting point. |
| 2 | Both frames have 8 teal clips/tabs (2 per side: top/bottom tabs, side ring clips with shiny metal centres). They protrude 25–40 px past the straight wire edge. The suggested quads (and every earlier ROI) included parts of them. | Claude, on midpoint zooms; confirmed by a teal-colour scan | `region.wire_intrusions` groups teal pixels and pads each group by 8 px. `region.notched_polygon` cuts those boxes out as notches. All 16 notches were checked visually. |
| 3 | img2 suggested top edge (y=235) lay on the shaded underside of the top wire (no margin at x≈1770–2310). Its bottom corners sat at the start of the rounded fillets. | Claude, edge and corner zooms | Top 235→242, bottom 1850→1845, corners nudged inward. Based only on frame geometry; no detector output was consulted. |
| 4 | Rounding vertices interpolated on slanted sides pushed those sides up to 0.5 px outward (415 / 137 polygon pixels outside the quad). Caught by the new test. | Claude (test `test_corrected_roi_file_consistent`) | Vertices kept unrounded; JSON regenerated (≤0.5 px vertex change); test passes. |
| 5 | Status said "COMPLETE" while listing optional and out-of-scope items as blockers. | User / Codex (parent) | Status rewritten. unseen_img2 is out of scope, and re-marking corners is optional. Neither is a blocker. |

Verification results:
- unseen_img1: smallest straight-edge gap about 4–5 px (left edge and bottom-left). Nearest notch clearance about 3 px (left side of the ring at notch 2). No wire inside the polygon at any corner, notch or edge zoom.
- img2: smallest gap about 5 px (upper right edge, left side of notch 5). No wire inside.
- Of the 12 highest-scoring img2 false positives under the corrected ROI, none is centred on wire; the closest, 1 px inside the edge, is centred on a shell below the top wire.
- The unseen_img1 queue items within 2 px of the edge (#0, #2, #5) are centred on shells beside the wire. Their template windows overlap the wire.

Figures: `figures/roi_{img2,unseen_img1}_{whole,corners,notches,edges}.png`.

Metric impact (no retuning; `frozen_params.json` unchanged): img2 F1 0.080 (result of record) → 0.083 (post-hoc ROI correction). unseen_img1 detections 2,197 → 2,150.

Tests: 16 of 16 pass. The notebook was re-executed end to end with 0 errors and 0 stderr or warning outputs.
