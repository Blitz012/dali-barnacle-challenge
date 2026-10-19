# Final report: ROI correction milestone and project status

2026-10-18, Claude (implementation). Codex (parent) reviews this milestone. Nothing published.

## Status
Technically complete with no open blockers. **Personally authored, not technical:** README "Conclusions" and "What I Learned" are intentionally empty for the project author.

## What changed
- **`data/roi_corrected.json`** (new) holds the agent-verified inner polygons. Each records its Codex starting quad, any refinement, its notch boxes, the polygon itself and its caveats.
  - **unseen_img1** is the operational review ROI. Quad [[1010,225],[2550,195],[2540,1730],[1035,1740]] (Codex's suggestion, unchanged), plus 8 notches.
  - **img2** is a post-hoc ROI correction. Quad [[968,242],[2577,242],[2537,1845],[978,1845]], refined from Codex's [[965,235],[2580,235],[2540,1850],[975,1850]], plus 8 notches. The top edge touched the wire's shaded underside and the bottom corners sat on the fillets.
- **Clip/tab notches.** Both frames have 8 teal clips/tabs (2 per side) protruding 25–40 px inward, which every earlier ROI included. `src/region.py` gains `wire_intrusions` (finds them by colour), `notched_polygon`, `load_corrected`, `load_corrected_polygon` and `distance_to_edge`.
- **Figures.** `src/viz.py` gains `roi_whole`, `roi_zooms` and `roi_figures`; the notebook regenerates all verification figures.
- **Historical records kept.** `data/corners.json` is unchanged and is still the img2 result of record. `data/corners_provisional.json` keeps its coordinates and is marked historical.
- **Frozen settings untouched.** `data/frozen_params.json` and `data/templates.json` are unchanged (SHA-256 9697b359… and be8ecf15…). There was no detector retuning.
- **Queue file.** `results/unseen_img1_review_queue.csv` (new) lists 2,150 detections, least confident first, each with its distance to the ROI edge.
- **Docs.** README, notebook (sections 0, 3, 8.2, 8.3, 9.1, 10, 11), COLLABORATION_STATUS, REVIEW_LOG (milestone 3), COLLABORATION (decision update) and AGENTS all updated. unseen_img2 is documented as out of scope, not as a blocker.

## Results
| row | predicted | GT | count err | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| img1 development | 2,071 | 1,689 | +22.6 % | 0.390 | 0.478 | 0.429 |
| img2 held-out, corners.json (**result of record**) | 2,850 | 396 | +619.7 % | 0.046 | 0.328 | 0.080 |
| img2 earlier eye-read ROI (historical sensitivity) | 2,854 | 396 | +620.7 % | 0.045 | 0.323 | 0.079 |
| img2 **post-hoc ROI correction**, no retuning | 2,712 | 395 | +586.6 % | 0.048 | 0.327 | 0.083 |
| random points, img2 corrected ROI | 2,712 | 395 | +586.6 % | 0.028 | 0.190 | 0.048 |
| unseen_img1, agent-verified ROI | 2,150 | – | – | – | – | – |
| unseen_img1, earlier eye-read ROI (historical) | 2,197 | – | – | – | – | – |

## ROI verification (by the agents, not the user; the polygons are not exact)
Claude inspected every corner (200 px zooms), notch (140 px) and straight-edge sample (120 px, at 1/6, 1/2 and 5/6 of each side).
- **unseen_img1.** No wire inside the polygon anywhere. The tightest straight-edge gap is about 4–5 px (left edge, bottom-left); the tightest notch clearance is about 3 px (left side of the ring at notch 2).
- **img2.** No wire inside after the refinement. The tightest gap is about 5 px (upper right edge, notch 5).
- **Detections near the edge.**
  - None of the 12 highest-scoring img2 false positives under the corrected ROI is centred on wire.
  - unseen queue items #0, #2 and #5 lie 0–2 px inside the edge. They are centred on shells beside the wire, but their template windows overlap it.
- **Cost of the margin.** A thin strip of barnacles beside the wire and around each clip is excluded.

Figures for the parent's review:
- `figures/roi_unseen_img1_whole.png`, `figures/roi_unseen_img1_corners.png`, `figures/roi_unseen_img1_notches.png`, `figures/roi_unseen_img1_edges.png`
- `figures/roi_img2_whole.png`, `figures/roi_img2_corners.png`, `figures/roi_img2_notches.png`, `figures/roi_img2_edges.png`

## Checks run
- `.venv/bin/python -m tests.test_core`: **16 tests passed**. Two are new, `test_notched_polygon` and `test_corrected_roi_file_consistent`; the second caught a 0.5 px rounding overreach, which is fixed.
- `jupyter nbconvert --execute --inplace notebook.ipynb`: 43/43 code cells executed in sequence, **0 errors, 0 stderr or warning outputs**, no warning filters.
- **Source PNG SHA-256.** All 8 `data/*.png` hashes are unchanged from the start of this run and identical to the files in `Barnacles.zip` (the challenge download).

## Out of scope / optional
- `unseen_img2.png` is out of scope: frameless, a different scale, and not required by the brief.
- Re-marking corners by hand is optional and not needed for reproduction.
