"""Pluggable detector backends, selected by ``settings.detector_backend``."""

from __future__ import annotations

from observer.config import Settings
from observer.pipeline.detector.base import Detector


def build_detector(settings: Settings) -> Detector:
    backend = settings.detector_backend.lower()
    if backend == "yoloworld":
        from observer.pipeline.detector.yoloworld_backend import YoloWorldDetector

        return YoloWorldDetector(settings)
    if backend == "none":
        from observer.pipeline.detector.null_backend import NullDetector

        return NullDetector(settings)
    raise ValueError(f"Unknown detector_backend: {settings.detector_backend!r}")
