"""Database models and session management (SQLModel + SQLite)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine

from observer.config import get_settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class VideoStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    error = "error"


class AircraftType(str, Enum):
    airplane = "airplane"
    helicopter = "helicopter"
    unknown = "unknown"


class Video(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(index=True)
    received_at: datetime = Field(default_factory=_utcnow)
    duration_s: Optional[float] = None
    status: VideoStatus = Field(default=VideoStatus.pending, index=True)
    progress: float = 0.0  # 0..1
    error: Optional[str] = None
    processed_at: Optional[datetime] = None


class Event(SQLModel, table=True):
    """A detected aircraft track, flagged when it is a takeoff."""

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    type: AircraftType = Field(default=AircraftType.unknown, index=True)
    is_takeoff: bool = Field(default=False, index=True)
    confidence: float = 0.0
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    thumb_path: Optional[str] = None
    clip_path: Optional[str] = None
    annotated_path: Optional[str] = None
    trajectory_json: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)


_settings = get_settings()
# check_same_thread=False: the worker runs CPU work in a thread pool while the
# web layer reads from the event loop thread.
engine = create_engine(
    _settings.db_url,
    echo=False,
    # timeout: wait (seconds) for a write lock instead of erroring immediately,
    # so concurrent batch writers + dashboard readers coexist.
    connect_args={"check_same_thread": False, "timeout": 30},
)


def init_db() -> None:
    _settings.ensure_dirs()
    SQLModel.metadata.create_all(engine)
    # WAL lets readers (dashboard) and a writer (batch) work concurrently.
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        conn.commit()


def get_session() -> Session:
    return Session(engine)
