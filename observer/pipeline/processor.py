"""Process one clip end-to-end: motion -> tracking -> detect -> classify -> artifacts.

Kept free of database/web concerns: it returns a :class:`ProcessingResult` and
reports progress via an optional callback. The worker is responsible for
persisting rows and publishing live updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from observer.config import Settings
from observer.pipeline import decode
from observer.pipeline.detector.base import Detector
from observer.pipeline.motion import MotionDetector
from observer.pipeline.tracking import Track, TrackBuilder
from observer.pipeline.trajectory import Classification, classify
from observer.storage import files

ProgressCb = Callable[[float], None]


@dataclass
class EventResult:
    index: int
    classification: Classification
    start_time_s: float
    end_time_s: float
    clip_path: Optional[Path] = None
    thumb_path: Optional[Path] = None
    annotated_path: Optional[Path] = None


@dataclass
class ProcessingResult:
    duration_s: float
    frame_w: int
    frame_h: int
    events: list[EventResult] = field(default_factory=list)


def _detector_signals(
    track: Track, keyframes: dict[int, np.ndarray], detector: Detector, settings: Settings
) -> tuple[float, float]:
    """Return ``(airplane_conf, bird_conf)`` for the track.

    Crops a padded region around the object at the keyframe nearest the track
    midpoint and upscales it before detection, so small/distant aircraft (and
    birds) are large enough for the detector to classify. Returns the best
    airplane and bird confidences seen in that crop.
    """
    if not keyframes:
        return 0.0, 0.0
    mid = track.points[len(track.points) // 2]
    key_idx = min(keyframes, key=lambda k: abs(k - mid.frame_index))
    frame = keyframes[key_idx]
    pt = min(track.points, key=lambda p: abs(p.frame_index - key_idx))
    fh, fw = frame.shape[:2]

    # Pad generously around the object so the detector has surrounding context.
    half = max(pt.w, pt.h, 48.0) * 1.5
    x1 = max(0, int(pt.cx - half)); y1 = max(0, int(pt.cy - half))
    x2 = min(fw, int(pt.cx + half)); y2 = min(fh, int(pt.cy + half))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, 0.0

    long_edge = max(crop.shape[0], crop.shape[1])
    if long_edge < settings.detect_crop_min_size:
        scale = settings.detect_crop_min_size / long_edge
        crop = cv2.resize(
            crop,
            (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    airplane = bird = 0.0
    for det in detector.detect(crop):
        if det.class_id == settings.airplane_class_id:
            airplane = max(airplane, det.confidence)
        elif det.class_id == settings.bird_class_id:
            bird = max(bird, det.confidence)
    return airplane, bird


def _draw_annotation(frame: np.ndarray, box: tuple, label: str) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(
        out, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 220, 0), 2, cv2.LINE_AA,
    )
    return out


def _write_event_clip(
    source: Path, dest: Path, start_s: float, end_s: float, pad_s: float = 0.5
) -> Optional[Path]:
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 0 or h == 0:
        cap.release()
        return None
    lo = max(0.0, start_s - pad_s)
    hi = end_s + pad_s
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            if lo <= t <= hi:
                writer.write(frame)
            idx += 1
    finally:
        cap.release()
        writer.release()
    return dest


def process_video(
    path: Path,
    settings: Settings,
    detector: Detector,
    on_progress: Optional[ProgressCb] = None,
    media_key: Optional[str] = None,
) -> ProcessingResult:
    key = media_key or path.stem
    info = decode.probe(path)
    total_sampled = max(1, int(info.frame_count / max(1, info.fps / settings.sample_fps)))
    # Retain ~40 evenly spaced keyframes for detection/thumbnails (bounds memory).
    key_stride = max(1, total_sampled // 40)

    motion = MotionDetector(
        history=settings.mog2_history,
        var_threshold=settings.mog2_var_threshold,
        min_area_frac=settings.min_blob_area_frac,
        max_area_frac=settings.max_blob_area_frac,
    )
    builder = TrackBuilder(
        max_match_distance_frac=settings.track_match_distance_frac,
        max_age_frames=settings.track_max_age_frames,
    )
    keyframes: dict[int, np.ndarray] = {}
    frame_w = frame_h = 0

    for sf in decode.iter_frames(path, settings.sample_fps, settings.max_frame_width):
        frame_h, frame_w = sf.image.shape[:2]
        candidates = motion.apply(sf.image)
        if sf.index < settings.mog2_warmup_frames:
            candidates = []  # ignore noise while the background model primes
        builder.update(sf.index, sf.t_seconds, candidates, frame_w, frame_h)
        if sf.index % key_stride == 0:
            keyframes[sf.index] = sf.image
        if on_progress and total_sampled:
            on_progress(min(0.8, 0.8 * sf.index / total_sampled))

    tracks = builder.finish(settings.min_track_frames)
    tracks.sort(key=lambda t: t.points[0].t_seconds)

    result = ProcessingResult(
        duration_s=info.duration_s, frame_w=frame_w, frame_h=frame_h
    )
    event_index = 0
    for track in tracks:
        airplane_conf, bird_conf = _detector_signals(track, keyframes, detector, settings)
        cls = classify(track.points, frame_w, frame_h, settings, airplane_conf)
        if not cls.is_takeoff:
            continue
        # Detector-based bird rejection (only meaningful when a real detector is
        # in use; the null backend returns zeros and this is a no-op).
        if bird_conf >= settings.bird_reject_conf and bird_conf > airplane_conf:
            continue

        start_s = track.points[0].t_seconds
        end_s = track.points[-1].t_seconds
        evt = EventResult(
            index=event_index,
            classification=cls,
            start_time_s=start_s,
            end_time_s=end_s,
        )

        # Thumbnail + annotated preview from the keyframe nearest the midpoint.
        mid = track.points[len(track.points) // 2]
        if keyframes:
            key_idx = min(keyframes, key=lambda k: abs(k - mid.frame_index))
            frame = keyframes[key_idx]
            box = (
                mid.cx - mid.w / 2, mid.cy - mid.h / 2,
                mid.cx + mid.w / 2, mid.cy + mid.h / 2,
            )
            label = f"{cls.type.value} {cls.confidence:.2f}"
            annotated = _draw_annotation(frame, box, label)
            ann_path = files.event_annotated_path(key, event_index)
            cv2.imwrite(str(ann_path), annotated)
            evt.annotated_path = ann_path

            x1, y1, x2, y2 = (max(0, int(v)) for v in box)
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                thumb_path = files.event_thumb_path(key, event_index)
                cv2.imwrite(str(thumb_path), crop)
                evt.thumb_path = thumb_path

        clip_path = files.event_clip_path(key, event_index)
        evt.clip_path = _write_event_clip(path, clip_path, start_s, end_s)

        result.events.append(evt)
        event_index += 1

    if on_progress:
        on_progress(1.0)
    return result
