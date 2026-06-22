"""End-to-end pipeline test with a stub detector (no model, no real footage).

Verifies that per-frame detections are aggregated into the right per-clip verdict
and that an evidence image is written when aircraft is present. Artifacts go to a
temp data dir so the test leaves nothing behind.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from observer.config import Settings
from observer.pipeline.detector.base import Detection


def _make_video(path: Path, n_frames: int = 30, fps: int = 10) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(n_frames):
        w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.release()
    return path


class _StubDetector:
    """Emits a detection at ``conf`` for the first ``n_hits`` frames it sees."""

    def __init__(self, n_hits: int, conf: float) -> None:
        self.n_hits = n_hits
        self.conf = conf
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.calls <= self.n_hits:
            return [Detection(xyxy=(10, 10, 30, 30), label="aircraft", confidence=self.conf)]
        return []

    def classify_type(self, frame):
        return "helicopter", 0.8


def _settings(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    return s


def test_persistent_hits_flag_aircraft(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    from observer.storage import files as files_mod

    monkeypatch.setattr(files_mod, "settings", settings)
    from observer.pipeline.processor import process_video

    src = _make_video(tmp_path / "clip.mp4")
    result = process_video(src, settings, _StubDetector(n_hits=5, conf=0.6), media_key="clip")

    assert result.has_aircraft is True
    assert result.aircraft_type == "helicopter"
    assert result.evidence_path is not None and Path(result.evidence_path).exists()


def test_no_detections_no_aircraft(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    from observer.storage import files as files_mod

    monkeypatch.setattr(files_mod, "settings", settings)
    from observer.pipeline.processor import process_video

    src = _make_video(tmp_path / "empty.mp4")
    result = process_video(src, settings, _StubDetector(n_hits=0, conf=0.0), media_key="empty")

    assert result.has_aircraft is False
    assert result.evidence_path is None
