# DALI Data Challenge: Barnacle Counting

Counting barnacles inside the inner square of a wire frame with classical computer vision (OpenCV, scikit-image, scipy), plus a confidence ranking for human review. No deep learning.

## Summary

- **Ground truth.** Each closed outline in a mask is one barnacle: the alpha-channel outline is filled, the lines are subtracted, and interiors of at least 10 px are labelled. mask1 gives 1,694 barnacles (1,689 inside the img1 frame) and mask2 gives 396. Inspection found 22 barnacles this misses in mask1 (about 1.3 %). Barnacles with no outline at all are not quantified.
- **Evaluation region.** At original image scale with no warping. img1 uses the hand-marked corners in `data/corners.json`, as does the img2 result of record. unseen_img1 uses an agent-verified inner polygon (`data/roi_corrected.json`), and so does a separate, labelled post-hoc img2 row. That polygon excludes the wire and the eight clips and tabs that stick into the opening.
- **Rejected baseline.** An inverted Laplacian-of-Gaussian blob detector assumed plates are dark spots. Only 5 of its 960 detections (0.52 %) fell inside an annotated plate, because the darkest spots are crevices between shells.
- **Detector.** Masked normalised correlation (`TM_CCOEFF_NORMED`) with nine whole-shell templates from img1 (three each at about 25, 33 and 41 px), at eight rotations and native scale. Each rotation is scored over the template's inscribed disc, so no invented corner pixels are compared. Detections are score-map peaks inside the ROI, at least 12 px apart, with a score of at least 0.30.
- **Matching.** One-to-one matching with the distance gate inside the assignment (maximum pairs, then minimum total distance). The radius is 7.42 px, half the median img1 plate diameter.
- **Protocol.** All settings were chosen on img1 and frozen in `data/frozen_params.json` before img2 was scored once. img1 numbers are development results, not validation.

## Results

| image | predicted | ground truth | signed count error | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| img1, development (full ROI) | 2,071 | 1,689 | +22.6 % | 0.390 | 0.478 | 0.429 |
| img1, excluding the 9 template exemplars | 2,062 | 1,680 | +22.7 % | 0.387 | 0.475 | 0.427 |
| img2, held-out test (frozen) | 2,850 | 396 | +619.7 % | 0.046 | 0.328 | 0.080 |
| img2, earlier eye-read inner-edge ROI (post-hoc sensitivity, historical) | 2,854 | 396 | +620.7 % | 0.045 | 0.323 | 0.079 |
| img2, **post-hoc ROI correction** (agent-verified polygon, no retuning) | 2,712 | 395 | +586.6 % | 0.048 | 0.327 | 0.083 |
| uniform random points at img1 detection count | 2,071 | 1,689 | +22.6 % | 0.124 | 0.152 | 0.137 |
| uniform random points at img2 detection count | 2,850 | 396 | +619.7 % | 0.024 | 0.169 | 0.041 |
| uniform random points, img2 corrected ROI | 2,712 | 395 | +586.6 % | 0.028 | 0.190 | 0.048 |
| unseen_img1 (agent-verified ROI, no ground truth) | 2,150 | – | – | – | – | – |
| unseen_img1 (earlier eye-read ROI, historical) | 2,197 | – | – | – | – | – |

- img2 differs from img1: fewer, larger, brown plated barnacles on dark algae, against img1's packed small white shells. The img1 templates do not transfer. No setting was changed after seeing img2.
- Confidence: on img1 the 50 highest-scoring detections are 86 % correct, falling to about 45 % by 500 and flattening after that. On img2 with `corners.json`, precision is 0.05–0.09 at every prefix and the top scores land on the frame wire. With the post-hoc corrected ROI, precision is 0.09–0.11 for the first 500 and 0.048 overall. Low confidence means score < 0.50 (fixed on img1). It applies to 80 % of img1 detections, so the 5 × 5 tile triage separates tiles only weakly.
### Follow-up: frame ROI correction (post-hoc; the frozen results above are unchanged)

- **img2 ROI.** `data/corners.json` places the img2 corners on the wire centreline, so the wire and its clips fall inside the polygon. The held-out result of record (F1 0.080) keeps those corners. A post-hoc ROI correction gives F1 0.083: an inner polygon that excludes the wire and clips, made after seeing img2 results but from frame geometry only, with `data/frozen_params.json` and every detector setting unchanged. The poor img2 result comes from the change in barnacle appearance, not from the ROI.
- **unseen_img1 ROI.** No hand-marked corners exist, so the review queue uses the agent-verified inner polygon in `data/roi_corrected.json`. Codex picked the starting quadrilateral and checked its corners on 200 px zooms. Claude cut notches around the clips and checked every corner, notch and edge on zooms (`figures/roi_unseen_img1_*.png`). The polygon has not been checked by the user and is not exact: it sits about 3–15 px inside the wire, so a thin strip of barnacles beside the wire is excluded. The full queue, least confident first, is `results/unseen_img1_review_queue.csv`.
- **Out of scope: `unseen_img2.png`.** It is a 500 × 500 frameless close-up of img2-type shells. Processing it would need new scale and region choices made on unlabelled data, and the brief asks only for unseen_img1. This is a scope decision, not a blocker.

## Status

All technical deliverables are complete. **Conclusions** and **What I Learned** below are written by the project author.

## Conclusions

The approach works partly. On img1, where the templates came from, the detector reaches F1 0.429 (precision 0.390, recall 0.478). On img2, evaluated with frozen settings, it reaches F1 0.080, about twice the random-point baseline of 0.041 but far from useful as an automatic counter.

The difference in appearance between the images is a likely reason for the poor transfer. img1 contains tightly packed small white shells around mid-grey plates; img2 contains larger, brown plated barnacles on dark algae. Every template came from img1, so the template collection covered a narrow range of appearances and sizes. Against mask2's 396 annotated instances, only 130 of the 2,850 detections match, leaving 2,720 unmatched. Because mask2 does not outline every barnacle, these cannot all be called confirmed biological false positives. The visible differences in shell size, surface texture and surrounding material were warning signs that one image's templates might not generalize.

The original frame boundary also included some wire. A separately reported, post-hoc correction raises img2 F1 only from 0.080 to 0.083, with the detector unchanged. Correcting the frame does not fix the poor result.

The ranking carries some signal on img1: 43 of the 50 highest-scoring detections are correct (86%), falling to 45.4% among the first 500. This suggests a possible assisted workflow, but does not establish that it saves time. For example, accepting the first 50 as a batch and deleting the seven errors would take eight actions rather than 43 individual marks: a hypothetical saving of 35 clicks, about 2% of a full count on an image with about 1,700 barnacles. That excludes inspection time and assumes every error is recognized. A timed user test would be needed before claiming a practical benefit.

The ranking does not transfer reliably to img2. With the original frame region, only three of the first 50 detections are correct (6%), and 42 of the first 500 (8.4%). Some of the highest scores fall on the frame wire. Removing the wire improves the top of the ranking, but does not make the detector reliable.

Next, I would test templates or a small classifier trained on examples from several development images, keeping separate images for final evaluation. I would also test per-image calibration: a scientist marks five to ten representative barnacles, and those examples guide template and scale selection. This fits the existing workflow because scientists already inspect images and identify barnacles by hand. Whether that setup saves more time than it costs remains an experiment, not a demonstrated result.

## What I Learned

What surprised me most was how little the correlation score separated annotated plate centres from the image overall: the median was 0.32 at true plates, against 0.28 over all frame pixels. I had expected the visible shell-and-plate pattern to produce a clearer separation. An overlay can look plausible without showing whether the score distinguishes objects reliably. Next time, I would compare scores at labelled objects and at carefully chosen background locations before trusting the overlays, since the all-pixel figure is not a pure background sample: it includes the plates themselves.

Every detector setting was frozen on img1 in `data/frozen_params.json` before evaluating img2. The drop from F1 0.43 to 0.08 made retuning, or cutting new templates from img2, an obvious next experiment. But doing that would turn img2 into development data, so an improved score would no longer measure transfer to an untouched image. Keeping the original result made the failure informative. The later frame correction is reported separately rather than replacing it.

I also learned that normalized correlation compares pixel patterns, not the concept of a barnacle. `TM_CCOEFF_NORMED` subtracts each patch's mean and normalizes its variation, which reduces sensitivity to uniform changes in brightness and contrast. It remains sensitive to differences in shape, alignment, shading and texture, so two shells can look similar to a person while correlating poorly. Fine texture may contribute to the weak separation here, but the experiment does not isolate it as the cause.

Claude Code implemented the code and notebook. Codex performed read-only reviews in bounded fix-and-review rounds, and chose the starting corners for the post-hoc frame correction. The findings and fixes are recorded in `REVIEW_LOG.md`. I used both agents to develop the approach and examine its evidence; the implementation and diagnostics were agent-assisted. Those checks exposed problems in the proposed methods: the inverted LoG detector mainly selected crevices, and closing the annotation outlines removed small real interiors.

Next time, I would establish the shared project instructions, evaluation split and file-editing ownership before implementation. I would keep one editing session active at a time and give the other agent a read-only review role. I would also inspect a broader set of development examples before choosing a method, so assumptions about appearance and scale are tested earlier.

## How to run

Requires Python 3.9 or newer (tested on 3.9.6, macOS).

```bash
git clone https://github.com/Blitz012/dali-barnacle-challenge.git
cd dali-barnacle-challenge
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m tests.test_core     # matching, rotation, suppression and ROI checks
.venv/bin/jupyter notebook notebook.ipynb
```

The images and masks are in `data/`, so nothing else needs downloading. To re-execute the whole notebook from the command line instead (about 3 minutes):

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

To re-mark frame corners interactively (opens a window; writes `data/corners.json`): `.venv/bin/python -m src.region <image>.png`. This is optional, not needed to reproduce the results.

## Files

| path | contents |
|---|---|
| `notebook.ipynb` | The full story: data audit, ground truth, metrics, rejected baseline, template selection, tuning, frozen results, failures, confidence and review queue. |
| `src/dataio.py` | Image and alpha-channel mask loading. |
| `src/truth.py` | Ground-truth instances from outline masks, plus the closing diagnostic. |
| `src/region.py` | Frame corners, point-in-polygon tests, ROI masks, the wire-clip finder and notched-polygon builder, and loaders for the corrected and historical ROIs. |
| `src/detect.py` | Rejected inverted-LoG baseline. |
| `src/templates.py` | Shell-size estimate, candidate gallery, rotation checks, masked template matching and ROI peak finding. |
| `src/evaluate.py` | Gated one-to-one matching, metrics, exemplar-exclusion protocol, prefix precision curve and random baseline. |
| `src/pipeline.py` | Frozen-settings loader and the end-to-end detector; refuses to run if the template records have drifted. |
| `src/review.py` | 5 × 5 tile triage and review queue. |
| `src/viz.py` | Matplotlib figures, including the ROI verification figures. |
| `tests/test_core.py` | 16 checks: matching, metrics, rotation centre, score-map centre, peak suppression, notched polygons and the corrected-ROI file. |
| `data/templates.json` | All 120 gallery candidates, the 9 chosen IDs and the reasons for each. |
| `data/frozen_params.json` | Every setting chosen on img1, frozen before img2. |
| `data/corners.json` | Hand-marked inner-frame corners for img1 and img2 ([x, y]). |
| `data/roi_corrected.json` | Agent-verified inner polygons (wire and clips excluded): the unseen_img1 review ROI and the post-hoc img2 correction, with starting quads, notch boxes, provenance and caveats. |
| `data/corners_provisional.json` | Historical: the earlier eye-read quadrilaterals behind the original unseen_img1 queue (2,197) and the img2 sensitivity row. |
| `results/unseen_img1_review_queue.csv` | All 2,150 unseen_img1 detections, least confident first, with score, low-confidence flag and distance to the ROI edge. |
| `figures/` | Template candidate galleries; ROI verification figures `roi_{img2,unseen_img1}_{whole,corners,notches,edges}.png`. |
| `REVIEW_LOG.md` | Codex review findings and their outcomes. |
| `FINAL_REPORT.md` | Compact report of the final (ROI) milestone, its verification results and figure paths. |
| `CLAUDE.md`, `AGENTS.md` | The working rules given to Claude Code and Codex, kept to show how the agents were run. They mention `COLLABORATION.md` and `COLLABORATION_STATUS.md`, private coordination notes not included in this repo. |
