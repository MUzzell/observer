"""Process one clip: scan frames for aircraft, decide presence, save evidence.

Kept free of database/web concerns: returns a :class:`ClipResult` and reports
progress via an optional callback. The per-frame scan is factored out as
:func:`scan_clip` so the evaluator can reuse it to sweep decision thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
class BestDetection:
    confidence: float
    frame: np.ndarray
    box: tuple
    label: str
    t_seconds: float


@dataclass
class Scan:
    confidences: list[float] = field(default_factory=list)  # per-frame top conf
    best: Optional[BestDetection] = None
    duration_s: float = 0.0
    num_frames: int = 0


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


def scan_clip(
    path: Path,
    settings: Settings,
    detector: Detector,
    on_progress: Optional[ProgressCb] = None,
) -> Scan:
    """Run the detector over sampled frames and return per-frame confidences plus
    the single best detection (for the evidence image). No decision is made here."""
    info = decode.probe(path)
    total = max(1, int(info.duration_s * settings.detect_sample_fps))
    scan = Scan(duration_s=info.duration_s)

    # max_width=0 -> no downscale; the detector wants full native resolution.
    for sf in decode.iter_frames(path, settings.detect_sample_fps, max_width=0):
        scan.num_frames += 1
        dets = detector.detect(sf.image)
        if dets:
            top = max(dets, key=lambda d: d.confidence)
            scan.confidences.append(top.confidence)
            if scan.best is None or top.confidence > scan.best.confidence:
                scan.best = BestDetection(
                    top.confidence, sf.image.copy(), top.xyxy, top.label, sf.t_seconds
                )
        if on_progress:
            on_progress(min(0.95, scan.num_frames / total))
    if on_progress:
        on_progress(1.0)
    return scan


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
    scan = scan_clip(path, settings, detector, on_progress)
    decision = decide(scan.confidences, settings)

    result = ClipResult(
        duration_s=scan.duration_s,
        has_aircraft=decision.has_aircraft,
        confidence=decision.confidence,
        num_hits=decision.num_hits,
        num_frames=scan.num_frames,
        best_time_s=scan.best.t_seconds if scan.best else 0.0,
    )

    if decision.has_aircraft and scan.best is not None:
        if settings.enable_type_hint:
            result.aircraft_type, result.type_confidence = detector.classify_type(
                scan.best.frame
            )
        label = result.aircraft_type or scan.best.label
        annotated = _draw_box(scan.best.frame, scan.best.box, f"{label} {scan.best.confidence:.2f}")
        evidence = files.evidence_path(key)
        cv2.imwrite(str(evidence), annotated)
        result.evidence_path = evidence

    return result
