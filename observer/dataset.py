"""Build a YOLO-format detection dataset by distilling the YOLO-World detector.

Uses the working open-vocab detector as a *teacher*: on clips a human labelled
``aircraft`` it records the detector's confident boxes as ground-truth labels; on
``none`` clips it saves background frames (no labels) so the student model learns
to ignore birds/clutter. The resulting dataset trains a small YOLOv8n that can be
compiled to Hailo. See README "Hailo deployment".

Output layout (Ultralytics-compatible):
    out/
      images/{train,val}/<clip>_<idx>.jpg
      labels/{train,val}/<clip>_<idx>.txt    # "0 xc yc w h" normalized, per box
      data.yaml
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2

from observer.config import Settings, get_settings
from observer.pipeline import decode
from observer.pipeline.detector import build_detector
from observer.storage.db import Video, get_session, init_db
from sqlmodel import select


def _resolve(video: Video, clip_dir: Path, recursive: bool) -> Optional[Path]:
    if video.source_path and Path(video.source_path).is_file():
        return Path(video.source_path)
    direct = clip_dir / Path(video.filename).name
    if direct.is_file():
        return direct
    if recursive:
        return next(iter(clip_dir.rglob(Path(video.filename).name)), None)
    return None


def _yolo_line(box: tuple, w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def _write_example(out: Path, split: str, name: str, frame, lines: list[str]) -> None:
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), frame)
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))


def build_dataset(
    out_dir: Path,
    clip_dir: Path,
    recursive: bool = False,
    frames_per_clip: int = 20,
    neg_frames_per_clip: int = 10,
    conf: float = 0.25,
    val_split: float = 0.2,
    settings: Settings | None = None,
) -> None:
    settings = settings or get_settings()
    init_db()
    detector = build_detector(settings)

    with get_session() as session:
        clips = [
            (v, _resolve(v, clip_dir, recursive))
            for v in session.exec(
                select(Video).where(Video.human_label.in_(["aircraft", "none"]))
            ).all()
        ]
    clips = [(v, p) for v, p in clips if p is not None]
    if not clips:
        print("No labelled clips with locatable files. Run import-labels first.")
        return

    # Split by clip (not frame) to avoid train/val leakage.
    rng = random.Random(1234)
    rng.shuffle(clips)
    n_val = max(1, int(len(clips) * val_split)) if len(clips) > 1 else 0
    val_clips = {id(v) for v, _ in clips[:n_val]}

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_img = n_box = n_bg = 0
    for video, path in clips:
        split = "val" if id(video) in val_clips else "train"
        stem = Path(video.filename).stem
        is_aircraft = video.human_label == "aircraft"
        print(f"[{split}] {Path(path).name} ({video.human_label}) …")

        positives: list[tuple] = []  # (max_conf, frame, lines)
        negatives: list = []         # frames with no detections
        for sf in decode.iter_frames(path, settings.detect_sample_fps, max_width=0):
            h, w = sf.image.shape[:2]
            dets = [d for d in detector.detect(sf.image) if d.confidence >= conf]
            if is_aircraft and dets:
                lines = [_yolo_line(d.xyxy, w, h) for d in dets]
                positives.append((max(d.confidence for d in dets), sf.image, lines))
            elif not dets:
                negatives.append(sf.image)

        if is_aircraft:
            positives.sort(key=lambda t: t[0], reverse=True)
            for i, (_, frame, lines) in enumerate(positives[:frames_per_clip]):
                _write_example(out_dir, split, f"{stem}_{i}", frame, lines)
                n_img += 1
                n_box += len(lines)
        # Background frames (from both none-clips and aircraft-clip empty frames)
        # teach the model what's NOT an aircraft.
        rng.shuffle(negatives)
        for i, frame in enumerate(negatives[:neg_frames_per_clip]):
            _write_example(out_dir, split, f"{stem}_bg{i}", frame, [])
            n_img += 1
            n_bg += 1

    data_yaml = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: aircraft\n"
    )
    (out_dir / "data.yaml").write_text(data_yaml)

    print(f"\nDataset written to {out_dir}")
    print(f"  {n_img} images · {n_box} aircraft boxes · {n_bg} background frames")
    print(f"  data.yaml ready — train with:\n"
          f"    yolo detect train model=yolov8n.pt data={out_dir}/data.yaml "
          f"imgsz=1280 epochs=100")
