"""End-to-end pipeline test on a synthesized clip, with a stub detector.

Verifies the full motion -> tracking -> trajectory path produces a takeoff event
without needing real footage or a downloaded model. Artifacts are written under a
temporary data dir so the test leaves nothing behind.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from observer.config import Settings
from observer.pipeline.detector.base import Detection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_sample  # noqa: E402


class _StubDetector:
    """Returns no detections; the synthetic dot is judged on trajectory alone."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []


def _settings_for(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path, mog2_warmup_frames=2)
    s.ensure_dirs()
    return s


@pytest.mark.parametrize(
    "kind,expected", [("helicopter", "helicopter"), ("airplane", "airplane")]
)
def test_takeoff_detected(tmp_path, monkeypatch, kind, expected):
    # files.py binds module-level settings at import; point it at the temp dir.
    settings = _settings_for(tmp_path)
    from observer.storage import files as files_mod

    monkeypatch.setattr(files_mod, "settings", settings)

    from observer.pipeline.processor import process_video

    src = make_sample.render(kind, tmp_path)
    result = process_video(src, settings, _StubDetector())

    takeoffs = [e for e in result.events if e.classification.is_takeoff]
    assert takeoffs, f"expected a takeoff event for {kind}"
    assert takeoffs[0].classification.type.value == expected
    # An event clip should have been written.
    assert takeoffs[0].clip_path is not None and Path(takeoffs[0].clip_path).exists()


def test_static_clip_has_no_takeoff(tmp_path, monkeypatch):
    settings = _settings_for(tmp_path)
    from observer.storage import files as files_mod

    monkeypatch.setattr(files_mod, "settings", settings)
    from observer.pipeline.processor import process_video

    src = make_sample.render("none", tmp_path)
    result = process_video(src, settings, _StubDetector())
    assert not any(e.classification.is_takeoff for e in result.events)
