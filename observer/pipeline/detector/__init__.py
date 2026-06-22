"""Pluggable object-detector backends.

The pipeline only depends on the :class:`Detector` protocol, so the portable
Ultralytics backend and the optional Hailo (RPi) backend are interchangeable and
selected by ``settings.detector_backend``.
"""

from __future__ import annotations

from observer.config import Settings
from observer.pipeline.detector.base import Detector


def build_detector(settings: Settings) -> Detector:
    backend = settings.detector_backend.lower()
    if backend == "none":
        from observer.pipeline.detector.null_backend import NullDetector

        return NullDetector(settings)
    if backend == "ultralytics":
        from observer.pipeline.detector.ultralytics_backend import UltralyticsDetector

        return UltralyticsDetector(settings)
    if backend == "hailo":
        from observer.pipeline.detector.hailo_backend import HailoDetector

        return HailoDetector(settings)
    raise ValueError(f"Unknown detector_backend: {settings.detector_backend!r}")
