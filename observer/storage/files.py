"""Media file lifecycle helpers.

Clips move through ``incoming/`` -> ``processing/`` -> ``processed/`` so the
watcher never re-picks a file that is mid-process, and so a crash leaves a clip
in ``processing/`` where it can be requeued. Derived artifacts (event clips,
thumbnails, annotated previews) live in their own directories keyed by event.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from observer.config import get_settings

settings = get_settings()


def media_key(path: Path) -> str:
    """A filesystem-safe, collision-resistant key for a source clip.

    Combines the stem with a short hash of the absolute path so clips that share
    a basename across different subfolders don't overwrite each other's artifacts.
    """
    digest = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:8]
    return f"{path.stem}_{digest}"


def move_to_processing(path: Path) -> Path:
    dest = settings.processing_dir / path.name
    shutil.move(str(path), str(dest))
    return dest


def move_to_processed(path: Path) -> Path:
    dest = settings.processed_dir / path.name
    shutil.move(str(path), str(dest))
    return dest


def event_clip_path(key: str, event_index: int) -> Path:
    return settings.clips_dir / f"{key}_evt{event_index}.mp4"


def event_thumb_path(key: str, event_index: int) -> Path:
    return settings.thumbs_dir / f"{key}_evt{event_index}.jpg"


def event_annotated_path(key: str, event_index: int) -> Path:
    return settings.thumbs_dir / f"{key}_evt{event_index}_annotated.jpg"


def relative_media(path: Path | str | None) -> str | None:
    """Return a path relative to the data dir for serving over ``/media``."""
    if path is None:
        return None
    return str(Path(path).relative_to(settings.data_dir))
