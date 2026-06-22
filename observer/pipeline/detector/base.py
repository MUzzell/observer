"""Detector protocol shared by all backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    label: str
    confidence: float


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a BGR frame and return aircraft candidate detections."""
        ...

    def classify_type(self, frame: np.ndarray) -> tuple[str | None, float]:
        """Best-effort airplane-vs-helicopter guess for a frame. ``(None, 0.0)``
        if the backend can't tell."""
        ...
