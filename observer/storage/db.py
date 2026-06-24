"""Database model and session management (SQLModel + SQLite).

The unit of record is the clip (``Video``): each carries the per-clip verdict —
does it contain an aircraft — plus an evidence image.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from observer.config import get_settings
from observer.naming import parse_capture_time


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class VideoStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    error = "error"


class Video(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(index=True)
    received_at: datetime = Field(default_factory=_utcnow)
    # Capture time parsed from the filename (camera Pi stamps it); None if absent.
    captured_at: Optional[datetime] = Field(default=None, index=True)
    duration_s: Optional[float] = None
    status: VideoStatus = Field(default=VideoStatus.pending, index=True)
    progress: float = 0.0  # 0..1

    # Verdict
    has_aircraft: bool = Field(default=False, index=True)
    confidence: float = 0.0           # peak detection confidence
    num_hits: int = 0                 # frames with a confident detection
    num_frames: int = 0               # frames sampled
    aircraft_type: Optional[str] = None  # "helicopter"/"airplane" hint, optional
    best_time_s: float = 0.0          # timestamp of the evidence frame
    evidence_path: Optional[str] = None  # annotated best frame, served via /media

    # Audio sub-verdict ("audio"/"fusion" modes); has_aircraft above is the fused result.
    audio_has_aircraft: bool = Field(default=False, index=True)
    audio_confidence: float = 0.0

    # Human ground-truth label imported from the manual labeller (independent of
    # the detector verdict, so the two can be compared). "aircraft"/"none"/
    # "unreadable"/None.
    human_label: Optional[str] = Field(default=None, index=True)
    source_path: Optional[str] = None  # absolute path to the original clip, if known

    error: Optional[str] = None
    processed_at: Optional[datetime] = None


_settings = get_settings()
# check_same_thread=False: the worker runs CPU work in a thread pool while the
# web layer reads from the event loop thread. timeout: wait for write locks.
engine = create_engine(
    _settings.db_url,
    echo=False,
    connect_args={"check_same_thread": False, "timeout": 30},
)


def _add_missing_columns(conn) -> None:
    """Lightweight forward migration: ADD COLUMN for any model field missing from
    an existing table, so schema additions don't require dropping the DB."""
    existing = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(video)")}
    for col in Video.__table__.columns:
        if col.name not in existing:
            coltype = col.type.compile(engine.dialect)
            conn.exec_driver_sql(f"ALTER TABLE video ADD COLUMN {col.name} {coltype}")


def _backfill_captured_at() -> None:
    """Populate captured_at for rows that don't have it yet (e.g. existing rows
    after the column was added). Unparseable names stay NULL."""
    with Session(engine) as session:
        rows = session.exec(select(Video).where(Video.captured_at.is_(None))).all()
        changed = False
        for v in rows:
            dt = parse_capture_time(v.filename)
            if dt is not None:
                v.captured_at = dt
                session.add(v)
                changed = True
        if changed:
            session.commit()


def init_db() -> None:
    _settings.ensure_dirs()
    SQLModel.metadata.create_all(engine)
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        _add_missing_columns(conn)
        conn.commit()
    _backfill_captured_at()


def get_session() -> Session:
    return Session(engine)
