"""Score the detector against human labels and tune the decision thresholds.

Works for either modality:
  - ``visual`` (default): runs the video detector over labelled clips.
  - ``audio``: runs the audio detector over each clip's sidecar ``.wav``.

Pulls clips that carry a human label (`aircraft`/`none`), collects per-window
confidences (cached so re-runs are instant), then reports the confusion matrix /
precision / recall at the current settings, lists mismatches, and (with
``--sweep``) recommends thresholds. The scoring/metrics are modality-agnostic —
only the scanner, thresholds, and recommended env vars differ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from sqlmodel import select

from observer.config import Settings, get_settings
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


def _make_scanner(settings: Settings, modality: str) -> Callable[[Path], list[float]]:
    """Return a function path -> per-window confidences for the chosen modality."""
    if modality == "audio":
        from observer.pipeline.audio import build_audio_detector

        print(f"loading audio detector ({settings.audio_backend}) …")
        det = build_audio_detector(settings)
        return lambda p: det.scan(p)

    from observer.pipeline.detector import build_detector
    from observer.pipeline.processor import scan_clip

    print(f"loading detector ({settings.detector_backend}) …")
    det = build_detector(settings)
    return lambda p: scan_clip(p, settings, det).confidences


def _thresholds(settings: Settings, modality: str) -> tuple[float, int, float]:
    if modality == "audio":
        return (settings.audio_present_conf, settings.audio_min_hit_frames,
                settings.audio_strong_conf)
    return settings.present_conf, settings.min_hit_frames, settings.strong_conf


def _present_grid(modality: str) -> list[float]:
    # Audio (AudioSet) probabilities run lower, so sweep a lower band.
    base = 0.05 if modality == "audio" else 0.15
    return [round(base + 0.05 * i, 2) for i in range(10)]


def collect(settings: Settings, clip_dir: Path, recursive: bool, modality: str,
            use_cache: bool = True) -> list[Sample]:
    cache_path = settings.data_dir / f"eval_cache_{modality}.json"
    cache: dict = {}
    if use_cache and cache_path.exists():
        cache = json.loads(cache_path.read_text())

    with get_session() as session:
        labelled = list(
            session.exec(
                select(Video).where(Video.human_label.in_(["aircraft", "none"]))
            ).all()
        )

    scan: Optional[Callable[[Path], list[float]]] = None
    samples: list[Sample] = []
    missing = no_audio = 0
    for v in labelled:
        path = _resolve_path(v, clip_dir, recursive)
        if path is None:
            missing += 1
            continue
        if modality == "audio":
            scan_path = path.with_suffix(".wav")
            if not scan_path.exists():
                no_audio += 1
                continue
        else:
            scan_path = path

        key = str(scan_path.resolve())
        size = scan_path.stat().st_size
        entry = cache.get(key)
        if entry and entry.get("size") == size:
            confs = entry["confs"]
        else:
            if scan is None:
                scan = _make_scanner(settings, modality)
            print(f"scanning {scan_path.name} …")
            confs = scan(scan_path)
            cache[key] = {"size": size, "confs": confs}
        samples.append(Sample(Path(v.filename).name, v.human_label == "aircraft", confs))

    if use_cache:
        cache_path.write_text(json.dumps(cache))
    if missing:
        print(f"({missing} labelled clips not found under {clip_dir}, skipped)")
    if no_audio:
        print(f"({no_audio} clips have no sidecar .wav, skipped)")
    return samples


def _print_metrics(title: str, m: Metrics) -> None:
    print(f"\n{title}")
    print(f"  precision {m.precision:.2f}  recall {m.recall:.2f}  "
          f"F1 {m.f1:.2f}  accuracy {m.accuracy:.2f}")
    print(f"  TP {m.tp}  FP {m.fp}  FN {m.fn}  TN {m.tn}")


def evaluate(clip_dir: Path, recursive: bool = False, sweep: bool = False,
             use_cache: bool = True, modality: str = "visual") -> None:
    settings = get_settings()
    init_db()
    samples = collect(settings, clip_dir, recursive, modality, use_cache)
    n_air = sum(s.is_aircraft for s in samples)
    print(f"\n{len(samples)} labelled clips scored [{modality}] "
          f"({n_air} aircraft, {len(samples) - n_air} none)")
    if not samples:
        msg = "import some labels first (observer import-labels)"
        if modality == "audio":
            msg = "no clips have sidecar .wav files yet"
        print(f"Nothing to evaluate — {msg}.")
        return

    present, hits, strong = _thresholds(settings, modality)
    cur = metrics_at(samples, present, hits, strong)
    _print_metrics(
        f"Current {modality} settings (present={present}, hits={hits}, strong={strong}):",
        cur,
    )

    # Mismatches at current settings, for eyeballing.
    misses = []
    for s in samples:
        if _predict(s.confidences, present, hits, strong) != s.is_aircraft:
            kind = "false positive" if s.is_aircraft is False else "false negative"
            misses.append((kind, s.name, max(s.confidences, default=0.0)))
    if misses:
        print("\nMismatches:")
        for kind, name, peak in sorted(misses):
            print(f"  {kind:15} {name}  (peak conf {peak:.2f})")

    if not sweep:
        print("\nRun with --sweep to search for better thresholds.")
        return

    results = []
    for p in _present_grid(modality):
        for h in (1, 2, 3, 4, 5):
            results.append((p, h, metrics_at(samples, p, h, strong)))
    results.sort(key=lambda r: (r[2].f1, r[2].recall), reverse=True)

    print("\nTop threshold combinations by F1:")
    print(f"  {'present':>8} {'hits':>5} {'prec':>6} {'rec':>6} {'F1':>6} {'acc':>6}")
    for p, h, m in results[:8]:
        print(f"  {p:>8} {h:>5} {m.precision:>6.2f} {m.recall:>6.2f} "
              f"{m.f1:>6.2f} {m.accuracy:>6.2f}")

    best_present, best_hits, _ = results[0]
    prefix = "OBSERVER_AUDIO" if modality == "audio" else "OBSERVER"
    print("\nRecommended — apply with:")
    print(f"  export {prefix}_PRESENT_CONF={best_present}")
    print(f"  export {prefix}_MIN_HIT_FRAMES={best_hits}")
