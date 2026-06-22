"""Synthesize test clips so the pipeline can be exercised without real footage.

Each clip renders a static sky/ground background with a single bright object
moving along a scripted trajectory:

  - ``helicopter``: near-vertical ascent from the ground (steep climb angle).
  - ``airplane``:   fast horizontal run with a shallow climb, exiting the side.
  - ``none``:       a slow drifting blob that never ascends (should not flag).

Usage:
    python scripts/make_sample.py --out data/incoming
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
FPS = 25
DURATION_S = 6


def _background() -> np.ndarray:
    """Sky gradient (top) over a darker ground band (bottom)."""
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = y / H
        # light blue sky fading to pale near horizon
        bg[y, :] = (200 - int(60 * t), 170 - int(30 * t), 120 + int(40 * t))
    ground_y = int(H * 0.8)
    bg[ground_y:, :] = (60, 90, 70)
    return bg


def _trajectory(kind: str, n: int) -> list[tuple[int, int]]:
    ground_y = int(H * 0.78)
    pts: list[tuple[int, int]] = []
    for i in range(n):
        t = i / (n - 1)
        if kind == "helicopter":
            x = int(W * 0.5 + 30 * np.sin(t * 3))  # mostly vertical, slight sway
            y = int(ground_y - t * (ground_y - H * 0.1))
        elif kind == "airplane":
            x = int(W * 0.08 + t * (W * 0.9))       # fast horizontal sweep
            y = int(ground_y - t * (H * 0.22))      # shallow climb
        else:  # none
            x = int(W * 0.4 + t * (W * 0.05))       # barely drifts
            y = int(ground_y - t * (H * 0.02))
        pts.append((x, y))
    return pts


def render(kind: str, out_dir: Path) -> Path:
    n = FPS * DURATION_S
    bg = _background()
    traj = _trajectory(kind, n)
    out_path = out_dir / f"sample_{kind}.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    for (x, y) in traj:
        frame = bg.copy()
        cv2.circle(frame, (x, y), 9, (240, 240, 240), -1)
        cv2.circle(frame, (x, y), 9, (30, 30, 30), 2)
        writer.write(frame)
    writer.release()
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/incoming", help="output directory")
    ap.add_argument(
        "--kinds",
        nargs="+",
        default=["helicopter", "airplane", "none"],
        choices=["helicopter", "airplane", "none"],
    )
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in args.kinds:
        path = render(kind, out_dir)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
