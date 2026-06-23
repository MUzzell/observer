"""Audio detector protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class AudioDetector(Protocol):
    def scan(self, wav_path: Path) -> list[float]:
        """Return per-window aircraft confidence across the clip's audio.

        One value per analysis window (analogous to per-frame video confidences),
        so the same persistence-based decision logic applies to both modalities.
        """
        ...
