"""Background worker: ingest -> process -> persist -> publish.

Runs inside the FastAPI process. The watchdog watcher (in its own threads) hands
ready clips to an asyncio queue; a consumer coroutine processes them one at a
time, offloading the CPU-bound pipeline to a thread-pool executor so the event
loop (and SSE streaming) stays responsive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from observer.config import Settings, get_settings
from observer.ingest.watcher import IngestWatcher
from observer.pipeline.detector import build_detector
from observer.pipeline.processor import ProcessingResult, process_video
from observer.storage import files
from observer.storage.db import Event, Video, VideoStatus, get_session, init_db
from observer.web.events_bus import EventBus

log = logging.getLogger("observer.worker")


class WorkerService:
    def __init__(self, bus: EventBus, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._bus = bus
        self._queue: asyncio.Queue[Path] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._watcher: IngestWatcher | None = None
        self._consumer_task: asyncio.Task | None = None
        self._detector = None

    async def start(self) -> None:
        init_db()
        self._loop = asyncio.get_running_loop()
        self._bus.bind_loop(self._loop)
        self._detector = build_detector(self._settings)
        self._watcher = IngestWatcher(self._settings, self._on_ready)
        self._watcher.start()
        self._consumer_task = asyncio.create_task(self._consume())
        log.info("worker started (backend=%s)", self._settings.detector_backend)

    async def stop(self) -> None:
        if self._watcher:
            self._watcher.stop()
        if self._consumer_task:
            self._consumer_task.cancel()

    def _on_ready(self, path: Path) -> None:
        # Called from the watcher's settle thread.
        if self._loop:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, path)

    async def _consume(self) -> None:
        while True:
            path = await self._queue.get()
            try:
                await self._process_one(path)
            except Exception:  # keep the worker alive on a bad clip
                log.exception("processing failed for %s", path)
            finally:
                self._queue.task_done()

    async def _process_one(self, path: Path) -> None:
        # Register the video and move it out of incoming/.
        with get_session() as session:
            video = Video(filename=path.name, status=VideoStatus.processing)
            session.add(video)
            session.commit()
            session.refresh(video)
            video_id = video.id
        working_path = files.move_to_processing(path)
        await self._bus.publish(
            {"type": "video_received", "video_id": video_id, "filename": path.name}
        )

        last_emit = 0.0

        def on_progress(p: float) -> None:
            nonlocal last_emit
            now = time.monotonic()
            if now - last_emit < 0.5 and p < 1.0:
                return
            last_emit = now
            self._bus.publish_threadsafe(
                {"type": "progress", "video_id": video_id, "progress": round(p, 3)}
            )

        try:
            result: ProcessingResult = await self._loop.run_in_executor(
                None,
                lambda: process_video(
                    working_path, self._settings, self._detector, on_progress
                ),
            )
        except Exception as exc:
            self._mark_error(video_id, str(exc))
            await self._bus.publish(
                {"type": "error", "video_id": video_id, "error": str(exc)}
            )
            return

        await self._persist_result(video_id, path.name, working_path, result)

    async def _persist_result(
        self, video_id: int, filename: str, working_path: Path, result: ProcessingResult
    ) -> None:
        with get_session() as session:
            video = session.get(Video, video_id)
            for evt in result.events:
                row = Event(
                    video_id=video_id,
                    type=evt.classification.type,
                    is_takeoff=evt.classification.is_takeoff,
                    confidence=evt.classification.confidence,
                    start_time_s=evt.start_time_s,
                    end_time_s=evt.end_time_s,
                    thumb_path=files.relative_media(evt.thumb_path),
                    clip_path=files.relative_media(evt.clip_path),
                    annotated_path=files.relative_media(evt.annotated_path),
                    trajectory_json=json.dumps(evt.classification.metrics),
                )
                session.add(row)
                session.commit()
                session.refresh(row)
                await self._bus.publish(
                    {
                        "type": "event_detected",
                        "video_id": video_id,
                        "event_id": row.id,
                        "aircraft": row.type.value,
                        "confidence": row.confidence,
                    }
                )
            if video:
                video.status = VideoStatus.done
                video.progress = 1.0
                video.duration_s = result.duration_s
                video.processed_at = datetime.now(timezone.utc)
                session.add(video)
                session.commit()

        files.move_to_processed(working_path)
        await self._bus.publish(
            {
                "type": "done",
                "video_id": video_id,
                "filename": filename,
                "event_count": len(result.events),
            }
        )

    def _mark_error(self, video_id: int, error: str) -> None:
        with get_session() as session:
            video = session.get(Video, video_id)
            if video:
                video.status = VideoStatus.error
                video.error = error
                session.add(video)
                session.commit()
