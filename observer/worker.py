"""Background worker: ingest -> detect aircraft -> persist verdict -> publish.

Runs inside the FastAPI process. The watchdog watcher (own threads) hands ready
clips to an asyncio queue; a consumer coroutine processes them one at a time,
offloading the CPU-bound detector to a thread-pool executor so the event loop
(and SSE streaming) stays responsive.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from observer.config import Settings, get_settings
from observer.ingest.watcher import IngestWatcher
from observer.pipeline.detector import build_detector
from observer.pipeline.processor import ClipResult, process_video
from observer.storage import files
from observer.storage.db import Video, VideoStatus, get_session, init_db
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
        if self._loop:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, path)

    async def _consume(self) -> None:
        while True:
            path = await self._queue.get()
            try:
                await self._process_one(path)
            except Exception:
                log.exception("processing failed for %s", path)
            finally:
                self._queue.task_done()

    async def _process_one(self, path: Path) -> None:
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
            result: ClipResult = await self._loop.run_in_executor(
                None,
                lambda: process_video(
                    working_path,
                    self._settings,
                    self._detector,
                    on_progress,
                    media_key=files.media_key(working_path),
                ),
            )
        except Exception as exc:
            self._mark_error(video_id, str(exc))
            await self._bus.publish(
                {"type": "error", "video_id": video_id, "error": str(exc)}
            )
            return

        await self._persist(video_id, path.name, working_path, result)

    async def _persist(
        self, video_id: int, filename: str, working_path: Path, result: ClipResult
    ) -> None:
        with get_session() as session:
            video = session.get(Video, video_id)
            if video:
                video.status = VideoStatus.done
                video.progress = 1.0
                video.duration_s = result.duration_s
                video.has_aircraft = result.has_aircraft
                video.confidence = result.confidence
                video.num_hits = result.num_hits
                video.num_frames = result.num_frames
                video.aircraft_type = result.aircraft_type
                video.best_time_s = result.best_time_s
                video.evidence_path = files.relative_media(result.evidence_path)
                video.processed_at = datetime.now(timezone.utc)
                session.add(video)
                session.commit()

        files.move_to_processed(working_path)
        await self._bus.publish(
            {
                "type": "done",
                "video_id": video_id,
                "filename": filename,
                "has_aircraft": result.has_aircraft,
                "aircraft_type": result.aircraft_type,
                "confidence": round(result.confidence, 3),
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
