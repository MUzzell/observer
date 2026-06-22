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
    from observer.storage.db import init_db

    settings = get_settings()
    init_db()
    detector = build_detector(settings)
    result = process_video(Path(args.clip), settings, detector)
    out = {
        "duration_s": round(result.duration_s, 2),
        "events": [
            {
                "type": e.classification.type.value,
                "is_takeoff": e.classification.is_takeoff,
                "confidence": e.classification.confidence,
                "window": [round(e.start_time_s, 2), round(e.end_time_s, 2)],
                "metrics": e.classification.metrics,
                "clip": str(e.clip_path) if e.clip_path else None,
            }
            for e in result.events
        ],
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
            p,
            _WORKER["settings"],
            _WORKER["detector"],
            media_key=files.media_key(p),
        )
        events = [
            {
                "type": e.classification.type.value,
                "is_takeoff": e.classification.is_takeoff,
                "confidence": e.classification.confidence,
                "start": e.start_time_s,
                "end": e.end_time_s,
                "thumb": str(e.thumb_path) if e.thumb_path else None,
                "clip": str(e.clip_path) if e.clip_path else None,
                "annotated": str(e.annotated_path) if e.annotated_path else None,
                "metrics": e.classification.metrics,
            }
            for e in res.events
        ]
        return {"path": path_str, "ok": True, "duration": res.duration_s, "events": events}
    except Exception as exc:  # never let one bad clip kill the run
        return {"path": path_str, "ok": False, "error": repr(exc)}


def _batch(args: argparse.Namespace) -> None:
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from datetime import datetime, timezone

    from sqlmodel import select

    from observer.storage import files
    from observer.storage.db import (
        AircraftType,
        Event,
        Video,
        VideoStatus,
        get_session,
        init_db,
    )

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
    total_events = total_takeoffs = errors = completed = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(args.backend,)
    ) as ex:
        futures = {ex.submit(_worker_task, str(p)): p for p in todo}
        for fut in as_completed(futures):
            r = fut.result()
            completed += 1
            abspath = str(Path(r["path"]).resolve())
            with get_session() as session:
                if not r["ok"]:
                    errors += 1
                    session.add(
                        Video(filename=abspath, status=VideoStatus.error, error=r["error"])
                    )
                    session.commit()
                    print(f"[{completed}/{len(todo)}] ERROR {Path(r['path']).name}: {r['error']}")
                    continue
                video = Video(
                    filename=abspath,
                    status=VideoStatus.done,
                    progress=1.0,
                    duration_s=r["duration"],
                    processed_at=datetime.now(timezone.utc),
                )
                session.add(video)
                session.commit()
                session.refresh(video)
                takeoffs = 0
                for e in r["events"]:
                    session.add(
                        Event(
                            video_id=video.id,
                            type=AircraftType(e["type"]),
                            is_takeoff=e["is_takeoff"],
                            confidence=e["confidence"],
                            start_time_s=e["start"],
                            end_time_s=e["end"],
                            thumb_path=files.relative_media(e["thumb"]),
                            clip_path=files.relative_media(e["clip"]),
                            annotated_path=files.relative_media(e["annotated"]),
                            trajectory_json=json.dumps(e["metrics"]),
                        )
                    )
                    total_events += 1
                    if e["is_takeoff"]:
                        takeoffs += 1
                        total_takeoffs += 1
                session.commit()
            if takeoffs:
                kinds = ",".join(sorted({e["type"] for e in r["events"] if e["is_takeoff"]}))
                print(f"[{completed}/{len(todo)}] {Path(r['path']).name}: {takeoffs} takeoff(s) [{kinds}]")
            elif completed % 50 == 0:
                print(f"[{completed}/{len(todo)}] …")

    print(
        f"\nDone. {completed} processed · {total_takeoffs} takeoffs across "
        f"{total_events} events · {errors} errors."
    )
    print("Review with:  observer serve   →   http://localhost:8000")


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
        default="none",
        choices=["none", "ultralytics", "hailo"],
        help="detector backend (default: none = fast, trajectory-only, no torch)",
    )
    batch.add_argument("--workers", type=int, default=None, help="parallel processes (default: CPU count)")
    batch.add_argument("--limit", type=int, default=None, help="process at most N clips")
    batch.add_argument("--recursive", action="store_true", help="search subdirectories")
    batch.add_argument("--reprocess", action="store_true", help="re-run clips already marked done")
    batch.set_defaults(func=_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
