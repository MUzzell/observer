"""Detector protocol shared by all backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    class_id: int
    confidence: float


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run object detection on a BGR frame and return detections."""
        ...
