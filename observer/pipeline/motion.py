"""Motion-based candidate extraction using MOG2 background subtraction.

The camera is fixed and footage is daytime, so the background is largely static.
MOG2 cheaply isolates moving objects (aircraft) against the sky/runway, giving us
candidate boxes that the tracker then stitches into trajectories. This is far
lighter than running a detector on every full frame and works well on an RPi.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionCandidate:
    xyxy: tuple[float, float, float, float]
    area: float


class MotionDetector:
    def __init__(
        self,
        history: int,
        var_threshold: float,
        min_area_frac: float,
        max_area_frac: float,
    ) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self._min_area_frac = min_area_frac
        self._max_area_frac = max_area_frac

    def apply(self, frame: np.ndarray) -> list[MotionCandidate]:
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        mask = self._bg.apply(frame)
        # Clean up speckle and merge nearby fragments of one object.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        out: list[MotionCandidate] = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = float(bw * bh)
            frac = area / frame_area
            if frac < self._min_area_frac or frac > self._max_area_frac:
                continue
            out.append(MotionCandidate(xyxy=(x, y, x + bw, y + bh), area=area))
        return out
