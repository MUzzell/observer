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
from observer.naming import parse_capture_time
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
        # A queued ``Path`` is a freshly-arrived clip; a queued ``int`` is an
        # existing clip id to reprocess in place (reusing its DB row).
        self._queue: asyncio.Queue[Path | int] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._watcher: IngestWatcher | None = None
        self._consumer_task: asyncio.Task | None = None
        self._detector = None
        self._audio_detector = None

    async def start(self) -> None:
        init_db()
        self._loop = asyncio.get_running_loop()
        self._bus.bind_loop(self._loop)
        self._detector = build_detector(self._settings)
        if self._settings.detection_mode in ("audio", "fusion"):
            from observer.pipeline.audio import build_audio_detector

            self._audio_detector = build_audio_detector(self._settings)
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

    def enqueue_reprocess(self, video_id: int) -> None:
        """Queue an existing clip (by id) to be processed again. Safe to call
        from any thread (the web layer runs sync endpoints off the loop)."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, video_id)

    async def _consume(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if isinstance(item, Path):
                    await self._process_one(item)
                else:
                    await self._reprocess(item)
            except Exception:
                log.exception("processing failed for %s", item)
            finally:
                self._queue.task_done()

    async def _process_one(self, path: Path) -> None:
        with get_session() as session:
            video = Video(
                filename=path.name,
                status=VideoStatus.processing,
                captured_at=parse_capture_time(path.name),
            )
            session.add(video)
            session.commit()
            session.refresh(video)
            video_id = video.id
        working_path = files.move_to_processing(path)
        await self._bus.publish(
            {"type": "video_received", "video_id": video_id, "filename": path.name}
        )
        await self._run(video_id, path.name, working_path)

    async def _reprocess(self, video_id: int) -> None:
        """Re-run the pipeline for an existing clip, whatever state it's in."""
        with get_session() as session:
            video = session.get(Video, video_id)
            if video is None:
                return
            filename = video.filename
            source_path = video.source_path
            # Reset to a clean processing state; keep the human label (ground
            # truth) and source_path, drop the detector verdict.
            video.status = VideoStatus.processing
            video.progress = 0.0
            video.error = None
            video.processed_at = None
            video.has_aircraft = False
            video.confidence = 0.0
            video.num_hits = 0
            video.num_frames = 0
            video.aircraft_type = None
            video.best_time_s = 0.0
            video.audio_has_aircraft = False
            video.audio_confidence = 0.0
            session.add(video)
            session.commit()

        path = files.locate_clip(filename, source_path)
        if path is None:
            self._mark_error(video_id, "clip file not found for reprocessing")
            await self._bus.publish(
                {"type": "error", "video_id": video_id,
                 "error": "clip file not found for reprocessing"}
            )
            return

        # Pull a pipeline clip back into processing/; leave an imported source
        # clip where it lives (and don't move it to processed/ afterwards).
        in_pipeline = path.parent in (
            self._settings.incoming_dir, self._settings.processed_dir,
            self._settings.processing_dir,
        )
        if path.parent in (self._settings.incoming_dir, self._settings.processed_dir):
            working_path = files.move_to_processing(path)
        else:
            working_path = path
        await self._bus.publish(
            {"type": "video_received", "video_id": video_id, "filename": filename}
        )
        await self._run(video_id, filename, working_path, finalize_move=in_pipeline)

    async def _run(
        self, video_id: int, filename: str, working_path: Path,
        finalize_move: bool = True,
    ) -> None:
        """Run the detector on ``working_path`` and persist the verdict to the
        existing ``video_id`` row."""
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

        log.info("processing video %s (id=%s) at %s", filename, video_id, working_path)
        try:
            result: ClipResult = await self._loop.run_in_executor(
                None,
                lambda: process_video(
                    working_path,
                    self._settings,
                    self._detector,
                    on_progress,
                    media_key=files.media_key(working_path),
                    audio_detector=self._audio_detector,
                ),
            )
        except Exception as exc:
            # Log the full traceback (this is where a Hailo/detector failure
            # surfaces) — storing only str(exc) in the DB hides the cause.
            log.exception(
                "detector failed for video %s (id=%s, backend=%s)",
                filename, video_id, self._settings.detector_backend,
            )
            self._mark_error(video_id, str(exc))
            await self._bus.publish(
                {"type": "error", "video_id": video_id, "error": str(exc)}
            )
            return

        log.info(
            "processed video %s (id=%s): has_aircraft=%s conf=%.3f hits=%d/%d",
            filename, video_id, result.has_aircraft, result.confidence,
            result.num_hits, result.num_frames,
        )
        await self._persist(video_id, filename, working_path, result, finalize_move)

    async def _persist(
        self, video_id: int, filename: str, working_path: Path, result: ClipResult,
        finalize_move: bool = True,
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
                video.audio_has_aircraft = result.audio_has_aircraft
                video.audio_confidence = result.audio_confidence
                video.processed_at = datetime.now(timezone.utc)
                session.add(video)
                session.commit()

        if finalize_move:
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
