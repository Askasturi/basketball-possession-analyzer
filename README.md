# Basketball Possession Analyzer

A from-scratch basketball possession analysis project inspired by Roboflow-style basketball video analytics.

## Goal

The final system will:

1. Load basketball video clips.
2. Detect players, ball, rim, and jersey numbers.
3. Track players across frames.
4. Identify teams.
5. Read jersey numbers.
6. Match jersey numbers to tracked players.
7. Map players to a top-down court using homography.
8. Detect shot attempts.
9. Classify make/miss.
10. Render an annotated output video.

## Tech Stack

- Python 3.12
- OpenCV
- NumPy
- Pytest
- Ruff

Future optional adapters may support:

- RF-DETR
- SAM2
- SigLIP
- UMAP
- K-means
- SmolVLM2

Heavy ML models are optional adapters and are not required for basic tests.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```
