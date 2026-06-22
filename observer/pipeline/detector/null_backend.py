"""No-op detector: pure-trajectory mode.

The takeoff/type decision is made entirely from motion trajectory, so the
detector is only an extra confidence signal. Selecting ``detector_backend=none``
skips object detection entirely — no torch/ultralytics needed — which is ideal
for a fast first pass over a large clip archive.
"""

from __future__ import annotations

import numpy as np

from observer.config import Settings
from observer.pipeline.detector.base import Detection


class NullDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []
