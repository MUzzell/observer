"""Unit tests for the per-clip decision rule."""

from __future__ import annotations

from observer.pipeline.aggregate import decide

# present_conf=0.30, min_hit_frames=3, strong_conf=0.55
PRESENT, HITS, STRONG = 0.30, 3, 0.55


def d(confs):
    return decide(confs, PRESENT, HITS, STRONG)


def test_no_detections_means_no_aircraft():
    r = d([])
    assert r.has_aircraft is False
    assert r.num_hits == 0 and r.confidence == 0.0


def test_lone_weak_blip_rejected():
    assert d([0.16]).has_aircraft is False


def test_persistent_detections_flag_aircraft():
    r = d([0.32, 0.41, 0.7, 0.55, 0.6])
    assert r.has_aircraft is True
    assert r.num_hits == 5
    assert r.confidence == 0.7


def test_brief_but_strong_detection_flags_aircraft():
    r = d([0.2, 0.62])
    assert r.has_aircraft is True  # strong_conf rule
    assert r.num_hits == 1


def test_few_mid_detections_below_threshold():
    assert d([0.33, 0.40]).has_aircraft is False
