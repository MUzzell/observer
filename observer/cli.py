"""Command-line entry point.

  observer serve                 # run the dashboard + ingestion worker
  observer process <clip.mp4>    # process a single clip and print detected events
  observer batch <dir>           # bulk-process a directory of clips in parallel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from observer.config import get_settings


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "observer.web.app:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
    )


def _process(args: argparse.Namespace) -> None:
    from observer.pipeline.detector import build_detector
    from observer.pipeline.processor import process_video
    from observer.storage import files
    from observer.storage.db import init_db

    settings = get_settings()
    init_db()
    detector = build_detector(settings)
    p = Path(args.clip)
    result = process_video(p, settings, detector, media_key=files.media_key(p))
    out = {
        "clip": p.name,
        "duration_s": round(result.duration_s, 2),
        "has_aircraft": result.has_aircraft,
        "confidence": round(result.confidence, 3),
        "aircraft_type": result.aircraft_type,
        "frames_hit": f"{result.num_hits}/{result.num_frames}",
        "evidence": str(result.evidence_path) if result.evidence_path else None,
    }
    print(json.dumps(out, indent=2))


# --- Batch processing -----------------------------------------------------
# Worker-process globals: the detector is built once per process and reused
# across the many clips that process routes to it.
_WORKER: dict = {}


def _worker_init(backend: str) -> None:
    from observer.config import Settings
    from observer.pipeline.detector import build_detector

    settings = Settings(detector_backend=backend)
    _WORKER["settings"] = settings
    _WORKER["detector"] = build_detector(settings)


def _worker_task(path_str: str) -> dict:
    from observer.pipeline.processor import process_video
    from observer.storage import files

    p = Path(path_str)
    try:
        res = process_video(
            p, _WORKER["settings"], _WORKER["detector"], media_key=files.media_key(p)
        )
        return {
            "path": path_str,
            "ok": True,
            "duration": res.duration_s,
            "has_aircraft": res.has_aircraft,
            "confidence": res.confidence,
            "num_hits": res.num_hits,
            "num_frames": res.num_frames,
            "aircraft_type": res.aircraft_type,
            "best_time": res.best_time_s,
            "evidence": str(res.evidence_path) if res.evidence_path else None,
        }
    except Exception as exc:  # never let one bad clip kill the run
        return {"path": path_str, "ok": False, "error": repr(exc)}


def _batch(args: argparse.Namespace) -> None:
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from datetime import datetime, timezone

    from sqlmodel import select

    from observer.storage import files
    from observer.storage.db import Video, VideoStatus, get_session, init_db

    settings = get_settings()
    init_db()

    root = Path(args.directory)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    exts = settings.video_extensions
    pattern = "**/*" if args.recursive else "*"
    clips = sorted(
        p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in exts
    )
    if args.limit:
        clips = clips[: args.limit]

    # Resumability: skip clips already completed (keyed by absolute path).
    done: set[str] = set()
    if not args.reprocess:
        with get_session() as session:
            done = set(
                session.exec(
                    select(Video.filename).where(Video.status == VideoStatus.done)
                ).all()
            )
    todo = [p for p in clips if str(p.resolve()) not in done]
    print(
        f"{len(clips)} clips found · {len(clips) - len(todo)} already done · "
        f"{len(todo)} to process · backend={args.backend}"
    )
    if not todo:
        return

    workers = args.workers or os.cpu_count() or 4
    aircraft = errors = completed = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(args.backend,)
    ) as ex:
        futures = {ex.submit(_worker_task, str(p)): p for p in todo}
        for fut in as_completed(futures):
            r = fut.result()
            completed += 1
            abspath = str(Path(r["path"]).resolve())
            name = Path(r["path"]).name
            with get_session() as session:
                if not r["ok"]:
                    errors += 1
                    session.add(
                        Video(filename=abspath, status=VideoStatus.error, error=r["error"])
                    )
                    session.commit()
                    print(f"[{completed}/{len(todo)}] ERROR {name}: {r['error']}")
                    continue
                session.add(
                    Video(
                        filename=abspath,
                        status=VideoStatus.done,
                        progress=1.0,
                        duration_s=r["duration"],
                        has_aircraft=r["has_aircraft"],
                        confidence=r["confidence"],
                        num_hits=r["num_hits"],
                        num_frames=r["num_frames"],
                        aircraft_type=r["aircraft_type"],
                        best_time_s=r["best_time"],
                        evidence_path=files.relative_media(r["evidence"]),
                        processed_at=datetime.now(timezone.utc),
                    )
                )
                session.commit()
            if r["has_aircraft"]:
                aircraft += 1
                t = f" ({r['aircraft_type']})" if r["aircraft_type"] else ""
                print(f"[{completed}/{len(todo)}] {name}: AIRCRAFT{t} {r['confidence']:.2f}")
            elif completed % 50 == 0:
                print(f"[{completed}/{len(todo)}] …")

    print(f"\nDone. {completed} processed · {aircraft} with aircraft · {errors} errors.")
    print("Review with:  observer serve   →   http://localhost:8000")


DEFAULT_CLIP_DIR = "/run/media/muzzell/KINGSTON/observer/"


def _find_clip(clip_dir: Path, name: str, recursive: bool) -> Path | None:
    direct = clip_dir / name
    if direct.is_file():
        return direct
    if recursive:
        return next(iter(clip_dir.rglob(name)), None)
    return None


def _extract_thumb(src: Path, dest: Path) -> bool:
    import cv2

    cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    cv2.imwrite(str(dest), frame)
    return True


def _import_labels(args: argparse.Namespace) -> None:
    import csv as csvmod

    from sqlmodel import select

    from observer.pipeline import decode
    from observer.storage import files
    from observer.storage.db import Video, VideoStatus, get_session, init_db

    init_db()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    clip_dir = Path(args.dir)

    rows: list[tuple[str, str]] = []
    with csv_path.open(newline="") as f:
        for r in csvmod.reader(f):
            if len(r) >= 2 and r[0] != "filename":
                rows.append((r[0], r[1]))

    n_new = n_updated = n_thumb = n_missing = 0
    with get_session() as session:
        # Match existing rows by basename (rows may have been created by the
        # watcher or batch run under different path forms).
        existing = {Path(v.filename).name: v for v in session.exec(select(Video)).all()}
        for name, label in rows:
            video = existing.get(name)
            if video is None:
                video = Video(filename=name, status=VideoStatus.done)
                n_new += 1
            else:
                n_updated += 1
            video.human_label = label
            if video.status == VideoStatus.pending:
                video.status = VideoStatus.done

            src = _find_clip(clip_dir, name, args.recursive)
            if src is None:
                n_missing += 1
            else:
                video.source_path = str(src.resolve())
                if video.duration_s is None:
                    try:
                        video.duration_s = decode.probe(src).duration_s
                    except Exception:
                        pass
                # Give imported clips a thumbnail so the site has something to
                # show; never clobber a detector evidence image.
                if (not args.no_thumbs and not video.evidence_path
                        and label != "unreadable"):
                    thumb = files.label_thumb_path(files.media_key(src))
                    if _extract_thumb(src, thumb):
                        video.evidence_path = files.relative_media(thumb)
                        n_thumb += 1

            session.add(video)
            existing[name] = video
        session.commit()

    print(
        f"Imported {len(rows)} labels: {n_new} new, {n_updated} updated, "
        f"{n_thumb} thumbnails, {n_missing} clips not found under {clip_dir}"
    )
    print("View with:  observer serve   →   http://localhost:8000")


def main() -> None:
    parser = argparse.ArgumentParser(prog="observer")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the web dashboard and worker")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_serve)

    proc = sub.add_parser("process", help="process a single clip")
    proc.add_argument("clip")
    proc.set_defaults(func=_process)

    batch = sub.add_parser("batch", help="bulk-process a directory of clips in parallel")
    batch.add_argument("directory")
    batch.add_argument(
        "--backend",
        default="yoloworld",
        choices=["yoloworld", "none"],
        help="detector backend (default: yoloworld)",
    )
    batch.add_argument("--workers", type=int, default=None, help="parallel processes (default: CPU count)")
    batch.add_argument("--limit", type=int, default=None, help="process at most N clips")
    batch.add_argument("--recursive", action="store_true", help="search subdirectories")
    batch.add_argument("--reprocess", action="store_true", help="re-run clips already marked done")
    batch.set_defaults(func=_batch)

    imp = sub.add_parser("import-labels", help="import a manual-labels CSV into the DB/site")
    imp.add_argument("csv", help="labels.csv from tools/label_clips.py")
    imp.add_argument("--dir", default=DEFAULT_CLIP_DIR,
                     help="directory holding the clips (for thumbnails/playback)")
    imp.add_argument("--recursive", action="store_true", help="search subfolders for clips")
    imp.add_argument("--no-thumbs", action="store_true", help="skip thumbnail extraction")
    imp.set_defaults(func=_import_labels)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
