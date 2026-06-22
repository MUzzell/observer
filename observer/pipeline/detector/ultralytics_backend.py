"""Portable detector backend using Ultralytics YOLO (CPU or CUDA).

Used to confirm that a moving track is aircraft-like and to capture the
``airplane`` class confidence as an extra signal for the trajectory classifier.
The model is lazily loaded on first use.
"""

from __future__ import annotations

import numpy as np

from observer.config import Settings
from observer.pipeline.detector.base import Detection


class UltralyticsDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._settings.yolo_weights)
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        model = self._ensure_model()
        results = model.predict(
            frame, conf=self._settings.detect_conf, verbose=False
        )
        out: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                out.append(
                    Detection(
                        xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        class_id=int(box.cls[0]),
                        confidence=float(box.conf[0]),
                    )
                )
        return out
