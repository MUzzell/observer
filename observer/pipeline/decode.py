"""Frame decoding and sampling via OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return VideoInfo(fps=fps, frame_count=count, width=width, height=height)


@dataclass
class SampledFrame:
    index: int          # index within the sampled stream (0,1,2,...)
    t_seconds: float    # timestamp in the source video
    image: np.ndarray   # BGR, possibly downscaled


def iter_frames(
    path: Path, sample_fps: float, max_width: int
) -> Iterator[SampledFrame]:
    """Yield frames sampled down to roughly ``sample_fps``, downscaled to ``max_width``."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / sample_fps))) if sample_fps > 0 else 1
    src_idx = 0
    out_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_idx % step == 0:
                t = src_idx / src_fps
                if max_width and frame.shape[1] > max_width:
                    scale = max_width / frame.shape[1]
                    frame = cv2.resize(
                        frame,
                        (max_width, int(frame.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                yield SampledFrame(index=out_idx, t_seconds=t, image=frame)
                out_idx += 1
            src_idx += 1
    finally:
        cap.release()
