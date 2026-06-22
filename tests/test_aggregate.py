"""Unit tests for the per-clip aircraft decision rule."""

from __future__ import annotations

from observer.config import Settings
from observer.pipeline.aggregate import decide

S = Settings()  # min_hit_frames=3, present_conf=0.30, strong_conf=0.55


def test_no_detections_means_no_aircraft():
    d = decide([], S)
    assert d.has_aircraft is False
    assert d.num_hits == 0 and d.confidence == 0.0


def test_lone_weak_blip_rejected():
    # A single weak frame (e.g. a bird near the treeline) must not count.
    d = decide([0.16], S)
    assert d.has_aircraft is False


def test_persistent_detections_flag_aircraft():
    # Helicopter detected across many frames -> aircraft present.
    d = decide([0.32, 0.41, 0.7, 0.55, 0.6], S)
    assert d.has_aircraft is True
    assert d.num_hits == 5
    assert d.confidence == 0.7


def test_brief_but_strong_detection_flags_aircraft():
    # Two frames only, but one is very confident -> covers a quick pass.
    d = decide([0.2, 0.62], S)
    assert d.has_aircraft is True  # strong_conf rule
    assert d.num_hits == 1


def test_few_mid_detections_below_threshold():
    # Two mid hits, none strong -> not enough persistence.
    d = decide([0.33, 0.40], S)
    assert d.has_aircraft is False
