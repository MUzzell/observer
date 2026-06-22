"""Open-vocabulary detector backend (YOLO-World).

Verified on real footage: YOLO-World-X, prompted with the single word
"aircraft" and run full-frame at imgsz=1280, reliably detects the small distant
helicopters (peak conf ~0.7-0.86) while staying quiet on bird-only clips — where
stock COCO YOLO and the smaller YOLO-World scored at the noise floor (~0.03).

The model is loaded lazily on first use. ``classify_type`` re-prompts the same
model with airplane/helicopter to provide an optional type hint.
"""

from __future__ import annotations

import numpy as np

from observer.config import Settings
from observer.pipeline.detector.base import Detection


class YoloWorldDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._current_classes: tuple[str, ...] | None = None

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLOWorld

            self._model = YOLOWorld(self._settings.yoloworld_weights)
        return self._model

    def _set_classes(self, classes: tuple[str, ...]) -> None:
        if self._current_classes != classes:
            self._ensure_model().set_classes(list(classes))
            self._current_classes = classes

    def detect(self, frame: np.ndarray) -> list[Detection]:
        model = self._ensure_model()
        self._set_classes(self._settings.aircraft_prompt)
        result = model.predict(
            frame,
            imgsz=self._settings.detect_imgsz,
            conf=self._settings.detect_conf,
            verbose=False,
        )[0]
        out: list[Detection] = []
        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                out.append(
                    Detection(
                        xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        label=model.names[int(box.cls[0])],
                        confidence=float(box.conf[0]),
                    )
                )
        return out

    def classify_type(self, frame: np.ndarray) -> tuple[str | None, float]:
        model = self._ensure_model()
        self._set_classes(self._settings.type_prompts)
        result = model.predict(
            frame, imgsz=self._settings.detect_imgsz, conf=0.05, verbose=False
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None, 0.0
        best = max(result.boxes, key=lambda b: float(b.conf[0]))
        return model.names[int(best.cls[0])], float(best.conf[0])
