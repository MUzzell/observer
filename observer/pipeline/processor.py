"""Process one clip: scan frames for aircraft, decide presence, save evidence.

Kept free of database/web concerns: returns a :class:`ClipResult` and reports
progress via an optional callback. The per-frame scan is factored out as
:func:`scan_clip` so the evaluator can reuse it to sweep decision thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import cv2
import numpy as np

from observer.config import Settings
from observer.pipeline import decode
from observer.pipeline.aggregate import decide
from observer.pipeline.detector.base import Detector
from observer.storage import files

if TYPE_CHECKING:
    from observer.pipeline.audio.base import AudioDetector

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
    # A representative frame (≈ middle of the clip) kept for a thumbnail when no
    # detection produces an annotated evidence image.
    poster: Optional[np.ndarray] = None


@dataclass
class ClipResult:
    duration_s: float
    has_aircraft: bool          # final (fused) verdict
    confidence: float           # final (fused) confidence
    num_hits: int               # video frames hit
    num_frames: int
    aircraft_type: Optional[str] = None
    type_confidence: float = 0.0
    evidence_path: Optional[Path] = None
    best_time_s: float = 0.0
    # Audio sub-verdict (populated in "audio"/"fusion" modes when a sidecar WAV
    # is present).
    audio_has_aircraft: bool = False
    audio_confidence: float = 0.0
    audio_num_hits: int = 0
    audio_windows: int = 0


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
    mid = max(1, total // 2)
    scan = Scan(duration_s=info.duration_s)
    last_image: Optional[np.ndarray] = None

    # max_width=0 -> no downscale; the detector wants full native resolution.
    for sf in decode.iter_frames(path, settings.detect_sample_fps, max_width=0):
        scan.num_frames += 1
        last_image = sf.image
        # Grab a mid-clip frame for the fallback thumbnail (cheap single copy).
        if scan.num_frames == mid:
            scan.poster = sf.image.copy()
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
    # If the clip was shorter than estimated, fall back to the last frame seen.
    if scan.poster is None and last_image is not None:
        scan.poster = last_image.copy()
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
    audio_detector: Optional["AudioDetector"] = None,
) -> ClipResult:
    key = media_key or path.stem
    scan = scan_clip(path, settings, detector, on_progress)
    visual = decide(
        scan.confidences, settings.present_conf, settings.min_hit_frames,
        settings.strong_conf,
    )

    result = ClipResult(
        duration_s=scan.duration_s,
        has_aircraft=visual.has_aircraft,
        confidence=visual.confidence,
        num_hits=visual.num_hits,
        num_frames=scan.num_frames,
        best_time_s=scan.best.t_seconds if scan.best else 0.0,
    )

    # Audio sub-verdict from the sidecar WAV, when enabled.
    if settings.detection_mode in ("audio", "fusion") and audio_detector is not None:
        wav = path.with_suffix(".wav")
        if wav.exists():
            aconfs = audio_detector.scan(wav)
            adec = decide(
                aconfs, settings.audio_present_conf,
                settings.audio_min_hit_frames, settings.audio_strong_conf,
            )
            result.audio_has_aircraft = adec.has_aircraft
            result.audio_confidence = adec.confidence
            result.audio_num_hits = adec.num_hits
            result.audio_windows = len(aconfs)

    # Fuse the final verdict per mode ("visual" leaves the video result as-is).
    if settings.detection_mode == "audio":
        result.has_aircraft = result.audio_has_aircraft
        result.confidence = result.audio_confidence
    elif settings.detection_mode == "fusion":
        result.has_aircraft = visual.has_aircraft or result.audio_has_aircraft
        result.confidence = max(visual.confidence, result.audio_confidence)

    evidence = files.evidence_path(key)
    if result.has_aircraft and scan.best is not None:
        if settings.enable_type_hint:
            result.aircraft_type, result.type_confidence = detector.classify_type(
                scan.best.frame
            )
        label = result.aircraft_type or scan.best.label
        annotated = _draw_box(scan.best.frame, scan.best.box, f"{label} {scan.best.confidence:.2f}")
        cv2.imwrite(str(evidence), annotated)
        result.evidence_path = evidence
    elif scan.poster is not None:
        # No detection to annotate — still save a plain screenshot so the clip
        # has a thumbnail rather than a bare "done" status label.
        cv2.imwrite(str(evidence), scan.poster)
        result.evidence_path = evidence

    return result
