"""Pluggable audio aircraft-detection backends.

Mirrors the visual detector design: the pipeline depends only on the
:class:`AudioDetector` protocol, so the PANNs (AudioSet) backend and the no-op
backend are interchangeable, selected by ``settings.audio_backend``.
"""

from __future__ import annotations

from observer.config import Settings
from observer.pipeline.audio.base import AudioDetector


def build_audio_detector(settings: Settings) -> AudioDetector:
    backend = settings.audio_backend.lower()
    if backend == "panns":
        from observer.pipeline.audio.panns_backend import PannsDetector

        return PannsDetector(settings)
    if backend == "none":
        from observer.pipeline.audio.null_backend import NullAudioDetector

        return NullAudioDetector(settings)
    raise ValueError(f"Unknown audio_backend: {settings.audio_backend!r}")
