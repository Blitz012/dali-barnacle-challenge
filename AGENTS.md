# Agents working on this project (Claude and Codex)

Shared rules, distilled from CLAUDE.md and COLLABORATION.md (those files win on conflict).

- Classical CV only (OpenCV, scikit-image, scipy). No deep learning, SAM, Mask R-CNN or Streamlit.
- Claude implements; Codex reviews read-only. Only one agent edits at a time.
- Never modify data/img*.png, data/mask*.png, data/masked_img*.png or data/unseen_img*.png.
- Original-scale images; no perspective warping. ROI via src/region.py: data/corners.json (img1; img2 result of record), data/roi_corrected.json (unseen_img1 operational ROI; img2 post-hoc correction, agent-verified, not user-verified).
- Tune only on img1. data/frozen_params.json is written once from img1 before img2 is evaluated; do not change it from img2 results.
- Coordinates: (row, col) in src unless a function says (x, y) (corners files are [x, y]).
- Logic in src/*.py; notebook.ipynb only calls functions and shows results.
- Do not write README Conclusions or What I Learned.
- At most three review/fix rounds per milestone; log findings and outcomes in REVIEW_LOG.md.
- Progress and blockers: COLLABORATION_STATUS.md.
