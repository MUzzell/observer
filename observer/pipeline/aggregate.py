"""Per-clip decision: turn per-frame detection confidences into yes/no.

Pure and dependency-free so the decision rule is unit-testable without video or
a model. Aircraft is present if it is detected on enough frames (persistence,
which rejects a lone false positive on a bird) OR a single detection is very
strong (covers brief but unambiguous passes).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    has_aircraft: bool
    confidence: float  # peak across windows
    num_hits: int      # windows at/above present_conf


def decide(
    confidences: list[float],
    present_conf: float,
    min_hit_frames: int,
    strong_conf: float,
) -> Decision:
    """Turn per-window confidences into a yes/no. Used for both video frames and
    audio windows, each passing its own thresholds."""
    peak = max(confidences, default=0.0)
    num_hits = sum(c >= present_conf for c in confidences)
    has = num_hits >= min_hit_frames or peak >= strong_conf
    return Decision(has_aircraft=has, confidence=peak, num_hits=num_hits)
