"""Process one clip: sample frames, detect aircraft, decide presence, save evidence.

Kept free of database/web concerns: returns a :class:`ClipResult` and reports
progress via an optional callback. The worker persists rows and publishes updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from observer.config import Settings
from observer.pipeline import decode
from observer.pipeline.aggregate import decide
from observer.pipeline.detector.base import Detector
from observer.storage import files

ProgressCb = Callable[[float], None]


@dataclass
class ClipResult:
    duration_s: float
    has_aircraft: bool
    confidence: float
    num_hits: int
    num_frames: int
    aircraft_type: Optional[str] = None
    type_confidence: float = 0.0
    evidence_path: Optional[Path] = None
    best_time_s: float = 0.0


def _draw_box(frame: np.ndarray, box: tuple, label: str) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(
        out, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 220, 0), 2, cv2.LINE_AA,
    )
    return out


def process_video(
    path: Path,
    settings: Settings,
    detector: Detector,
    on_progress: Optional[ProgressCb] = None,
    media_key: Optional[str] = None,
) -> ClipResult:
    key = media_key or path.stem
    info = decode.probe(path)
    total = max(1, int(info.duration_s * settings.detect_sample_fps))

    frame_confidences: list[float] = []
    # Track the single best detection across the clip for the evidence image.
    best_conf = 0.0
    best_frame: Optional[np.ndarray] = None
    best_box: Optional[tuple] = None
    best_label = ""
    best_time = 0.0
    num_frames = 0

    # max_width=0 -> no downscale; the detector wants full native resolution.
    for sf in decode.iter_frames(path, settings.detect_sample_fps, max_width=0):
        num_frames += 1
        dets = detector.detect(sf.image)
        if dets:
            top = max(dets, key=lambda d: d.confidence)
            frame_confidences.append(top.confidence)
            if top.confidence > best_conf:
                best_conf = top.confidence
                best_frame = sf.image.copy()
                best_box = top.xyxy
                best_label = top.label
                best_time = sf.t_seconds
        if on_progress:
            on_progress(min(0.95, num_frames / total))

    decision = decide(frame_confidences, settings)
    result = ClipResult(
        duration_s=info.duration_s,
        has_aircraft=decision.has_aircraft,
        confidence=decision.confidence,
        num_hits=decision.num_hits,
        num_frames=num_frames,
        best_time_s=best_time,
    )

    if decision.has_aircraft and best_frame is not None and best_box is not None:
        # Optional airplane-vs-helicopter hint from the best frame.
        if settings.enable_type_hint:
            atype, tconf = detector.classify_type(best_frame)
            result.aircraft_type = atype
            result.type_confidence = tconf
        label = result.aircraft_type or best_label
        annotated = _draw_box(best_frame, best_box, f"{label} {best_conf:.2f}")
        evidence = files.evidence_path(key)
        cv2.imwrite(str(evidence), annotated)
        result.evidence_path = evidence

    if on_progress:
        on_progress(1.0)
    return result
