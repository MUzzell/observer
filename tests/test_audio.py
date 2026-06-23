"""Tests for the audio path: sidecar discovery and visual/audio fusion.

Uses stub detectors and a synthesized silent WAV, so no model or real audio is
needed — this exercises the plumbing and the fusion logic, not model accuracy.
"""

from __future__ import annotations

import wave
from pathlib import Path

import cv2
import numpy as np

from observer.config import Settings
from observer.pipeline.detector.base import Detection


def _make_video(path: Path, n: int = 20, fps: int = 10) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(n):
        w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.release()
    return path


def _make_wav(path: Path, secs: int = 2, sr: int = 16000) -> Path:
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * sr * secs)
    return path


class _VisualStub:
    """Visual detector that always returns nothing."""

    def detect(self, frame):
        return []

    def classify_type(self, frame):
        return None, 0.0


class _AudioStub:
    def __init__(self, confs):
        self.confs = confs
        self.scanned = False

    def scan(self, wav_path):
        self.scanned = True
        return self.confs


def _prep(tmp_path, monkeypatch, settings):
    settings.ensure_dirs()
    from observer.storage import files as files_mod

    monkeypatch.setattr(files_mod, "settings", settings)
    _make_video(tmp_path / "clip.mp4")
    _make_wav(tmp_path / "clip.wav")
    from observer.pipeline.processor import process_video

    return process_video


def test_audio_mode_decides_from_audio(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, detection_mode="audio")
    process_video = _prep(tmp_path, monkeypatch, settings)
    audio = _AudioStub([0.5, 0.4, 0.6, 0.7])  # 4 hits >= 3 -> aircraft
    r = process_video(tmp_path / "clip.mp4", settings, _VisualStub(),
                      media_key="clip", audio_detector=audio)
    assert audio.scanned is True
    assert r.audio_has_aircraft is True
    assert r.has_aircraft is True          # audio mode -> audio drives verdict
    assert r.confidence == 0.7


def test_fusion_is_or_of_modalities(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, detection_mode="fusion")
    process_video = _prep(tmp_path, monkeypatch, settings)
    # visual sees nothing, audio fires -> fused verdict is aircraft
    r = process_video(tmp_path / "clip.mp4", settings, _VisualStub(),
                      media_key="clip", audio_detector=_AudioStub([0.5, 0.5, 0.5]))
    assert r.has_aircraft is True
    assert r.audio_has_aircraft is True


def test_visual_mode_ignores_audio(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path)  # default detection_mode="visual"
    process_video = _prep(tmp_path, monkeypatch, settings)
    audio = _AudioStub([0.9] * 5)
    r = process_video(tmp_path / "clip.mp4", settings, _VisualStub(),
                      media_key="clip", audio_detector=audio)
    assert audio.scanned is False          # audio not even scanned in visual mode
    assert r.has_aircraft is False
    assert r.audio_has_aircraft is False
