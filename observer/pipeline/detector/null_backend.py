"""No-op detector for tests and dry runs (no model, no torch)."""

from __future__ import annotations

import numpy as np

from observer.config import Settings
from observer.pipeline.detector.base import Detection


class NullDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []

    def classify_type(self, frame: np.ndarray) -> tuple[str | None, float]:
        return None, 0.0
