"""Unit tests for the takeoff / aircraft-type classifier.

These build synthetic tracks (no video or model needed) and assert the
heuristic's decisions, which is exactly why ``trajectory.classify`` is pure.
"""

from __future__ import annotations

from observer.config import Settings
from observer.pipeline.tracking import TrackPoint
from observer.storage.db import AircraftType
from observer.pipeline.trajectory import classify

FRAME_W, FRAME_H = 960, 540
SETTINGS = Settings()


def _track(points_xy: list[tuple[float, float]]) -> list[TrackPoint]:
    pts = []
    for i, (x, y) in enumerate(points_xy):
        pts.append(
            TrackPoint(frame_index=i, t_seconds=i * 0.125, cx=x, cy=y, w=20, h=20, area=400)
        )
    return pts


def test_helicopter_vertical_takeoff():
    # Near-vertical ascent from the ground band toward the top of frame.
    ys = [420, 380, 330, 270, 210, 150, 90]
    pts = _track([(480 + (3 if i % 2 else -3), y) for i, y in enumerate(ys)])
    result = classify(pts, FRAME_W, FRAME_H, SETTINGS)
    assert result.is_takeoff is True
    assert result.type == AircraftType.helicopter
    assert result.confidence > 0.4


def test_airplane_shallow_fast_takeoff():
    # Fast horizontal sweep with a shallow climb.
    pts = _track(
        [(80, 420), (220, 405), (360, 388), (500, 372), (640, 356), (780, 340), (900, 326)]
    )
    result = classify(pts, FRAME_W, FRAME_H, SETTINGS)
    assert result.is_takeoff is True
    assert result.type == AircraftType.airplane


def test_no_takeoff_when_static():
    # Barely drifts, never ascends meaningfully.
    pts = _track([(400, 420), (402, 419), (404, 419), (406, 418), (408, 418), (410, 418)])
    result = classify(pts, FRAME_W, FRAME_H, SETTINGS)
    assert result.is_takeoff is False
    assert result.type == AircraftType.unknown


def test_descending_is_not_takeoff():
    # A landing-like descent must not be flagged as a takeoff.
    ys = [120, 180, 250, 320, 380, 430]
    pts = _track([(480, y) for y in ys])
    result = classify(pts, FRAME_W, FRAME_H, SETTINGS)
    assert result.is_takeoff is False


def test_erratic_bird_rejected_by_straightness():
    # Net-upward but wandering left/right every step (flapping bird). High
    # vertical progress yet a long, jagged path -> low straightness -> rejected.
    pts = _track(
        [(480 + (90 if i % 2 else -90), 430 - i * 22) for i in range(12)]
    )
    result = classify(pts, FRAME_W, FRAME_H, SETTINGS)
    assert result.metrics["straightness"] < SETTINGS.min_straightness
    assert result.is_takeoff is False


def test_too_short_track():
    result = classify(_track([(100, 100)]), FRAME_W, FRAME_H, SETTINGS)
    assert result.is_takeoff is False
    assert result.type == AircraftType.unknown
