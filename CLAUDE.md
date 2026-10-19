# DALI Data Challenge: barnacle counting
Approach: classical CV (OpenCV, scikit-image, scipy) + confidence-driven human review.
No deep learning, SAM, Mask R-CNN, or Streamlit.

## Rules
- The user authorizes Claude and Codex to resolve routine design decisions together without pausing. Follow COLLABORATION.md for current decisions and bounded implementation/review iterations.
- Explain any new OpenCV/skimage function in 1-2 sentences.
- Logic lives in src/*.py; notebook.ipynb only calls functions and shows results.
- Tune only on img1. Never use img2 to choose parameters.
- Do not write README conclusions or the learning reflection.
- Display images with matplotlib (cv2 loads BGR; convert to RGB).
- Masks are contour outlines, not filled regions; check for alpha channels.
