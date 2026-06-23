"""No-op audio detector for tests and dry runs (no model, no audio deps)."""

from __future__ import annotations

from pathlib import Path

from observer.config import Settings


class NullAudioDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def scan(self, wav_path: Path) -> list[float]:
        return []
