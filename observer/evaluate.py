"""Score the detector against human labels and tune the decision thresholds.

Pulls clips that carry a human label (`aircraft`/`none`), runs the detector to
collect per-frame confidences (cached so re-runs are instant), then:
  - reports the confusion matrix / precision / recall at the current settings,
  - lists the mismatched clips so they can be eyeballed,
  - sweeps `present_conf` x `min_hit_frames` to recommend the best operating point.

Per-frame confidences are cached to ``data/eval_cache.json`` keyed by source path
and file size, so only new/changed clips are (re)scanned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlmodel import select

from observer.config import Settings, get_settings
from observer.pipeline.detector import build_detector
from observer.pipeline.processor import scan_clip
from observer.storage.db import Video, get_session, init_db


@dataclass
class Sample:
    name: str
    is_aircraft: bool       # ground truth
    confidences: list[float]


@dataclass
class Metrics:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / n if n else 0.0


def _predict(confs: list[float], present: float, hits: int, strong: float) -> bool:
    peak = max(confs, default=0.0)
    n = sum(c >= present for c in confs)
    return n >= hits or peak >= strong


def metrics_at(samples: list[Sample], present: float, hits: int, strong: float) -> Metrics:
    m = Metrics(0, 0, 0, 0)
    for s in samples:
        pred = _predict(s.confidences, present, hits, strong)
        if s.is_aircraft and pred:
            m.tp += 1
        elif s.is_aircraft and not pred:
            m.fn += 1
        elif not s.is_aircraft and pred:
            m.fp += 1
        else:
            m.tn += 1
    return m


def _resolve_path(video: Video, clip_dir: Path, recursive: bool) -> Optional[Path]:
    if video.source_path and Path(video.source_path).is_file():
        return Path(video.source_path)
    direct = clip_dir / Path(video.filename).name
    if direct.is_file():
        return direct
    if recursive:
        return next(iter(clip_dir.rglob(Path(video.filename).name)), None)
    return None


def collect(settings: Settings, clip_dir: Path, recursive: bool,
            use_cache: bool = True) -> list[Sample]:
    cache_path = settings.data_dir / "eval_cache.json"
    cache: dict = {}
    if use_cache and cache_path.exists():
        cache = json.loads(cache_path.read_text())

    with get_session() as session:
        labelled = list(
            session.exec(
                select(Video).where(Video.human_label.in_(["aircraft", "none"]))
            ).all()
        )

    detector = None
    samples: list[Sample] = []
    missing = 0
    for v in labelled:
        path = _resolve_path(v, clip_dir, recursive)
        if path is None:
            missing += 1
            continue
        key = str(path.resolve())
        size = path.stat().st_size
        entry = cache.get(key)
        if entry and entry.get("size") == size:
            confs = entry["confs"]
        else:
            if detector is None:
                print(f"loading detector ({settings.detector_backend}) …")
                detector = build_detector(settings)
            print(f"scanning {path.name} …")
            confs = scan_clip(path, settings, detector).confidences
            cache[key] = {"size": size, "confs": confs}
        samples.append(Sample(Path(v.filename).name, v.human_label == "aircraft", confs))

    if use_cache:
        cache_path.write_text(json.dumps(cache))
    if missing:
        print(f"({missing} labelled clips not found under {clip_dir}, skipped)")
    return samples


def _print_metrics(title: str, m: Metrics) -> None:
    print(f"\n{title}")
    print(f"  precision {m.precision:.2f}  recall {m.recall:.2f}  "
          f"F1 {m.f1:.2f}  accuracy {m.accuracy:.2f}")
    print(f"  TP {m.tp}  FP {m.fp}  FN {m.fn}  TN {m.tn}")


def evaluate(clip_dir: Path, recursive: bool = False, sweep: bool = False,
             use_cache: bool = True) -> None:
    settings = get_settings()
    init_db()
    samples = collect(settings, clip_dir, recursive, use_cache)
    n_air = sum(s.is_aircraft for s in samples)
    print(f"\n{len(samples)} labelled clips scored "
          f"({n_air} aircraft, {len(samples) - n_air} none)")
    if not samples:
        print("Nothing to evaluate — import some labels first (observer import-labels).")
        return

    cur = metrics_at(samples, settings.present_conf, settings.min_hit_frames,
                     settings.strong_conf)
    _print_metrics(
        f"Current settings (present_conf={settings.present_conf}, "
        f"min_hit_frames={settings.min_hit_frames}, strong_conf={settings.strong_conf}):",
        cur,
    )

    # Mismatches at current settings, for eyeballing.
    misses = []
    for s in samples:
        pred = _predict(s.confidences, settings.present_conf,
                        settings.min_hit_frames, settings.strong_conf)
        if pred != s.is_aircraft:
            kind = "false positive" if pred else "false negative"
            peak = max(s.confidences, default=0.0)
            misses.append((kind, s.name, peak))
    if misses:
        print("\nMismatches:")
        for kind, name, peak in sorted(misses):
            print(f"  {kind:15} {name}  (peak conf {peak:.2f})")

    if not sweep:
        print("\nRun with --sweep to search for better thresholds.")
        return

    present_grid = [round(0.15 + 0.05 * i, 2) for i in range(10)]  # 0.15..0.60
    hits_grid = [1, 2, 3, 4, 5]
    results = []
    for present in present_grid:
        for hits in hits_grid:
            m = metrics_at(samples, present, hits, settings.strong_conf)
            results.append((present, hits, m))
    results.sort(key=lambda r: (r[2].f1, r[2].recall), reverse=True)

    print("\nTop threshold combinations by F1:")
    print(f"  {'present':>8} {'hits':>5} {'prec':>6} {'rec':>6} {'F1':>6} {'acc':>6}")
    for present, hits, m in results[:8]:
        print(f"  {present:>8} {hits:>5} {m.precision:>6.2f} {m.recall:>6.2f} "
              f"{m.f1:>6.2f} {m.accuracy:>6.2f}")

    best_present, best_hits, _ = results[0]
    print("\nRecommended — apply with:")
    print(f"  export OBSERVER_PRESENT_CONF={best_present}")
    print(f"  export OBSERVER_MIN_HIT_FRAMES={best_hits}")
