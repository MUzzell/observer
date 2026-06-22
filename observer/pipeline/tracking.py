"""Associate per-frame motion candidates into tracks (centroid tracker).

Aircraft appear as small blobs that can move a long way between sampled frames, so
IoU-based association (e.g. ByteTrack) fails — consecutive boxes barely overlap.
A nearest-centroid tracker with a generous gating distance is both more robust for
this use case and dependency-free. Each track is a time-ordered list of centroid
points that the trajectory classifier consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from observer.pipeline.motion import MotionCandidate


@dataclass
class TrackPoint:
    frame_index: int
    t_seconds: float
    cx: float
    cy: float
    w: float
    h: float
    area: float


@dataclass
class Track:
    track_id: int
    points: list[TrackPoint] = field(default_factory=list)
    # Mean aircraft-class confidence from the detector over sampled crops.
    aircraft_confidence: float = 0.0
    _last_frame: int = 0

    @property
    def num_frames(self) -> int:
        return len(self.points)

    @property
    def centroid(self) -> tuple[float, float]:
        last = self.points[-1]
        return last.cx, last.cy


class TrackBuilder:
    def __init__(self, max_match_distance_frac: float, max_age_frames: int) -> None:
        self._max_match_frac = max_match_distance_frac
        self._max_age = max_age_frames
        self._tracks: dict[int, Track] = {}
        self._active: dict[int, Track] = {}
        self._next_id = 0

    def update(
        self,
        frame_index: int,
        t_seconds: float,
        candidates: list[MotionCandidate],
        frame_w: int,
        frame_h: int,
    ) -> None:
        # Retire tracks that have not been seen recently.
        for tid in [t for t, tr in self._active.items()
                    if frame_index - tr._last_frame > self._max_age]:
            self._active.pop(tid)

        max_dist = self._max_match_frac * math.hypot(frame_w, frame_h)
        cand_centroids = [
            ((c.xyxy[0] + c.xyxy[2]) / 2, (c.xyxy[1] + c.xyxy[3]) / 2) for c in candidates
        ]

        # Greedy nearest-centroid matching of candidates to active tracks.
        pairs: list[tuple[float, int, int]] = []  # (distance, track_id, cand_index)
        for tid, tr in self._active.items():
            tx, ty = tr.centroid
            for ci, (cx, cy) in enumerate(cand_centroids):
                d = math.hypot(cx - tx, cy - ty)
                if d <= max_dist:
                    pairs.append((d, tid, ci))
        pairs.sort(key=lambda p: p[0])

        used_tracks: set[int] = set()
        used_cands: set[int] = set()
        for d, tid, ci in pairs:
            if tid in used_tracks or ci in used_cands:
                continue
            used_tracks.add(tid)
            used_cands.add(ci)
            self._append(self._active[tid], frame_index, t_seconds, candidates[ci])

        # Unmatched candidates begin new tracks.
        for ci, cand in enumerate(candidates):
            if ci in used_cands:
                continue
            track = Track(track_id=self._next_id)
            self._next_id += 1
            self._append(track, frame_index, t_seconds, cand)
            self._tracks[track.track_id] = track
            self._active[track.track_id] = track

    @staticmethod
    def _append(
        track: Track, frame_index: int, t_seconds: float, cand: MotionCandidate
    ) -> None:
        x1, y1, x2, y2 = cand.xyxy
        track.points.append(
            TrackPoint(
                frame_index=frame_index,
                t_seconds=t_seconds,
                cx=(x1 + x2) / 2.0,
                cy=(y1 + y2) / 2.0,
                w=x2 - x1,
                h=y2 - y1,
                area=cand.area,
            )
        )
        track._last_frame = frame_index

    def finish(self, min_frames: int) -> list[Track]:
        return [t for t in self._tracks.values() if t.num_frames >= min_frames]
