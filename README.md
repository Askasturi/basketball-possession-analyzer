
# Basketball Possession Analyzer

A modular computer vision project for analyzing basketball possessions frame by frame.

The project is built to mirror a Roboflow-style basketball analytics pipeline:

1. Load basketball video clips.
2. Detect players, ball, rim, and jersey numbers.
3. Track players across frames.
4. Classify team identity.
5. Recognize jersey numbers.
6. Match jersey numbers to tracked players.
7. Resolve player identities.
8. Project player locations onto a top-down court using homography.
9. Detect shot events.
10. Render annotated output video.

The current implementation is intentionally lightweight and testable. It uses deterministic mock/simple components first, while leaving clean adapter stubs for future heavy models such as RF-DETR, SAM2, SigLIP, UMAP/K-means, and VLM-based jersey number recognition.

## Status

Current verified status:

```text
python -m ruff check .  ✅
python -m pytest        ✅
159 tests passing
demo pipeline           ✅
