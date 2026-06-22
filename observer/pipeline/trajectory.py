"""Decide whether a track is a takeoff, and classify airplane vs helicopter.

This is the heart of the heuristic. It is intentionally pure — it takes a list of
centroid points plus frame dimensions and threshold values — so it can be unit
tested against synthetic trajectories without any video or model dependencies.

Image coordinates: y increases downward, so upward (real-world) motion means a
*decrease* in y. We call ``rise = y_start - y_end`` (positive == ascended).

Behavioral discriminator:
  - Airplane: horizontal-dominant, shallow climb angle, high ground speed.
  - Helicopter: vertical-dominant ascent, low horizontal speed, may hover.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from observer.config import Settings
from observer.pipeline.tracking import TrackPoint
from observer.storage.db import AircraftType


@dataclass
class Classification:
    is_takeoff: bool
    type: AircraftType
    confidence: float
    metrics: dict


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def classify(
    points: list[TrackPoint],
    frame_w: int,
    frame_h: int,
    settings: Settings,
    aircraft_confidence: float = 0.0,
) -> Classification:
    if len(points) < 2 or frame_w <= 0 or frame_h <= 0:
        return Classification(False, AircraftType.unknown, 0.0, {"reason": "too_short"})

    start, end = points[0], points[-1]
    rise = start.cy - end.cy                 # +ve == moved up
    rise_frac = rise / frame_h
    horiz = abs(end.cx - start.cx)
    horiz_frac = horiz / frame_w
    diag = math.hypot(frame_w, frame_h)
    net_disp_frac = math.hypot(end.cx - start.cx, end.cy - start.cy) / diag

    # Climb angle of the net displacement vector, in degrees above horizontal.
    climb_angle = math.degrees(math.atan2(max(rise, 0.0), horiz + 1e-6))

    # Per-step speed (fraction of frame height per step), hover fraction, and the
    # total traversed path length (for the straightness metric below).
    speeds: list[float] = []
    up_steps = 0
    path_len_px = 0.0
    for a, b in zip(points, points[1:]):
        seg = math.hypot(b.cx - a.cx, b.cy - a.cy)
        path_len_px += seg
        speeds.append(seg / frame_h)
        if b.cy < a.cy:  # moved up this step
            up_steps += 1
    hover_frac = sum(s < settings.hover_speed_frac for s in speeds) / len(speeds)
    ascending_frac = up_steps / len(speeds)
    net_disp_px = math.hypot(end.cx - start.cx, end.cy - start.cy)
    straightness = net_disp_px / path_len_px if path_len_px > 1e-6 else 0.0

    metrics = {
        "rise_frac": round(rise_frac, 4),
        "horiz_frac": round(horiz_frac, 4),
        "net_disp_frac": round(net_disp_frac, 4),
        "climb_angle_deg": round(climb_angle, 2),
        "hover_frac": round(hover_frac, 3),
        "ascending_frac": round(ascending_frac, 3),
        "straightness": round(straightness, 3),
        "aircraft_confidence": round(aircraft_confidence, 3),
    }

    # --- Takeoff decision ------------------------------------------------
    # Straightness gate rejects erratic bird motion. A real takeoff climb (plane
    # or helicopter) follows a smooth, near-straight path; birds flit and wander.
    is_takeoff = (
        rise_frac >= settings.takeoff_min_rise_frac
        and net_disp_frac >= settings.takeoff_min_displacement_frac
        and ascending_frac >= 0.5
        and straightness >= settings.min_straightness
    )

    # --- Type classification --------------------------------------------
    vertical_dominant = climb_angle >= settings.helicopter_min_climb_angle_deg
    horizontal_dominant = (
        climb_angle <= settings.airplane_max_climb_angle_deg
        and horiz_frac >= settings.takeoff_min_displacement_frac
    )
    has_hover = hover_frac >= 0.25

    if vertical_dominant or has_hover:
        atype = AircraftType.helicopter
    elif horizontal_dominant:
        atype = AircraftType.airplane
    else:
        # Ambiguous mid-angle: lean toward whichever threshold is nearer.
        mid = (
            settings.airplane_max_climb_angle_deg
            + settings.helicopter_min_climb_angle_deg
        ) / 2
        atype = (
            AircraftType.helicopter if climb_angle >= mid else AircraftType.airplane
        )

    if not is_takeoff:
        atype = AircraftType.unknown

    # --- Confidence ------------------------------------------------------
    # Blend trajectory strength with the detector's aircraft-class confidence.
    rise_score = _clamp01(rise_frac / (settings.takeoff_min_rise_frac * 2))
    disp_score = _clamp01(net_disp_frac / (settings.takeoff_min_displacement_frac * 2))
    traj_score = (rise_score + disp_score + ascending_frac + straightness) / 4
    confidence = _clamp01(0.7 * traj_score + 0.3 * aircraft_confidence)
    if not is_takeoff:
        confidence *= 0.4

    return Classification(
        is_takeoff=is_takeoff,
        type=atype,
        confidence=round(confidence, 3),
        metrics=metrics,
    )
