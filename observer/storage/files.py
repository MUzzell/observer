"""Media file lifecycle helpers.

Clips move ``incoming/`` -> ``processing/`` -> ``processed/`` so the watcher
never re-picks a file mid-process and a crash leaves the clip recoverable. The
evidence image (best annotated frame) is keyed per clip.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from observer.config import get_settings

settings = get_settings()


def media_key(path: Path) -> str:
    """A filesystem-safe, collision-resistant key for a source clip (stem + short
    hash of the absolute path, so duplicate basenames don't clash)."""
    digest = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:8]
    return f"{path.stem}_{digest}"


def _move_with_sidecar(path: Path, dest_dir: Path) -> Path:
    """Move the clip and, if present, its matching ``.wav`` audio sidecar."""
    dest = dest_dir / path.name
    shutil.move(str(path), str(dest))
    wav = path.with_suffix(".wav")
    if wav.exists():
        shutil.move(str(wav), str(dest_dir / wav.name))
    return dest


def move_to_processing(path: Path) -> Path:
    return _move_with_sidecar(path, settings.processing_dir)


def move_to_processed(path: Path) -> Path:
    return _move_with_sidecar(path, settings.processed_dir)


def evidence_path(key: str) -> Path:
    return settings.thumbs_dir / f"{key}_evidence.jpg"


def label_thumb_path(key: str) -> Path:
    """Thumbnail for a manually-labelled clip (no detector evidence image)."""
    return settings.thumbs_dir / f"{key}_label.jpg"


def relative_media(path: Path | str | None) -> str | None:
    """Path relative to the data dir, for serving over ``/media``."""
    if path is None:
        return None
    return str(Path(path).relative_to(settings.data_dir))


def delete_clip_media(
    filename: str,
    source_path: str | None = None,
    evidence_path: str | None = None,
) -> list[Path]:
    """Wipe every on-disk artefact for a clip and return what was removed.

    Covers the video in any lifecycle dir (incoming/processing/processed) plus
    an imported source copy, each clip's ``.wav`` audio sidecar, and the
    evidence/thumbnail still. Missing files are skipped silently. Used by the
    dashboard 'delete' action to discard irrelevant captures wholesale.
    """
    candidates: list[Path] = [
        d / filename
        for d in (settings.incoming_dir, settings.processing_dir, settings.processed_dir)
    ]
    if source_path:
        candidates.append(Path(source_path))
    # Audio sidecars sit next to each video copy.
    candidates += [c.with_suffix(".wav") for c in list(candidates)]
    if evidence_path:
        # evidence_path is stored relative to the data dir (served via /media).
        still = (settings.data_dir / evidence_path).resolve()
        if str(still).startswith(str(settings.data_dir.resolve())):
            candidates.append(still)

    removed: list[Path] = []
    for p in candidates:
        try:
            if p.is_file():
                p.unlink()
                removed.append(p)
        except OSError:
            pass
    return removed


def locate_clip(filename: str, source_path: str | None = None) -> Path | None:
    """Find a clip's current on-disk location regardless of lifecycle stage.

    A clip moves incoming/ -> processing/ -> processed/ as it's handled, so its
    DB row can live in any of those at a given moment (an imported clip lives at
    its original ``source_path``). Returns the first candidate that exists, in
    order of likelihood, else ``None``.
    """
    candidates: list[Path] = []
    if source_path:
        candidates.append(Path(source_path))
    for d in (settings.processed_dir, settings.processing_dir, settings.incoming_dir):
        candidates.append(d / filename)
    for p in candidates:
        if p.is_file():
            return p
    return None
