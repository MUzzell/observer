"""Per-clip decision: turn per-frame detection confidences into yes/no.

Pure and dependency-free so the decision rule is unit-testable without video or
a model. Aircraft is present if it is detected on enough frames (persistence,
which rejects a lone false positive on a bird) OR a single detection is very
strong (covers brief but unambiguous passes).
"""

from __future__ import annotations

from dataclasses import dataclass

from observer.config import Settings


@dataclass
class Decision:
    has_aircraft: bool
    confidence: float  # peak across frames
    num_hits: int      # frames at/above present_conf


def decide(frame_confidences: list[float], settings: Settings) -> Decision:
    peak = max(frame_confidences, default=0.0)
    num_hits = sum(c >= settings.present_conf for c in frame_confidences)
    has = num_hits >= settings.min_hit_frames or peak >= settings.strong_conf
    return Decision(has_aircraft=has, confidence=peak, num_hits=num_hits)
