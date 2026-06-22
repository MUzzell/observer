"""Watch the incoming directory and emit clips once they have finished landing.

A clip synced over the network arrives incrementally, so we never enqueue a file
the instant we see it. Instead we wait until its size has been stable for
``watch_settle_seconds`` before calling ``on_ready``. Files already present when
the watcher starts are picked up via an initial scan.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from observer.config import Settings


class _Handler(FileSystemEventHandler):
    def __init__(self, on_change: Callable[[Path], None]) -> None:
        self._on_change = on_change

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._on_change(Path(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._on_change(Path(event.src_path))


class IngestWatcher:
    def __init__(self, settings: Settings, on_ready: Callable[[Path], None]) -> None:
        self._settings = settings
        self._on_ready = on_ready
        self._observer = Observer()
        # path -> (last_size, last_seen_monotonic)
        self._pending: dict[Path, tuple[int, float]] = {}
        self._emitted: set[Path] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._settle_thread = threading.Thread(target=self._settle_loop, daemon=True)

    def _is_video(self, path: Path) -> bool:
        return path.suffix.lower() in self._settings.video_extensions

    def _note(self, path: Path) -> None:
        if not self._is_video(path) or path in self._emitted:
            return
        with self._lock:
            self._pending[path] = (0, time.monotonic())

    def _settle_loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            ready: list[Path] = []
            with self._lock:
                for path, (last_size, _) in list(self._pending.items()):
                    if not path.exists():
                        self._pending.pop(path, None)
                        continue
                    size = path.stat().st_size
                    if size != last_size:
                        self._pending[path] = (size, now)  # changed; reset timer
                    elif now - self._pending[path][1] >= self._settings.watch_settle_seconds:
                        ready.append(path)
                for path in ready:
                    self._pending.pop(path, None)
                    self._emitted.add(path)
            for path in ready:
                self._on_ready(path)
            self._stop.wait(0.5)

    def _initial_scan(self) -> None:
        for path in sorted(self._settings.incoming_dir.glob("*")):
            if path.is_file():
                self._note(path)

    def start(self) -> None:
        self._settings.ensure_dirs()
        self._initial_scan()
        self._observer.schedule(
            _Handler(self._note), str(self._settings.incoming_dir), recursive=False
        )
        self._observer.start()
        self._settle_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._observer.stop()
        self._observer.join(timeout=5)
