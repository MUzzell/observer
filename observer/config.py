"""Central configuration.

All tunable thresholds for ingestion, the processing pipeline, and the trajectory
classifier live here so they can be adjusted without touching pipeline code (and
overridden via environment variables prefixed with ``OBSERVER_``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: <repo>/observer/config.py -> parents[1] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OBSERVER_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # --- Storage ----------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"

    # --- Ingestion --------------------------------------------------------
    # A file is only enqueued once its size has been stable for this long, so we
    # never start processing a clip that is still being copied/synced in.
    watch_settle_seconds: float = 2.0
    video_extensions: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".m4v")

    # --- Detector backend -------------------------------------------------
    # "ultralytics" (portable CPU/GPU default) or "hailo" (RPi accelerator).
    detector_backend: str = "ultralytics"
    yolo_weights: str = "yolov8n.pt"
    hailo_hef_path: Path = REPO_ROOT / "models" / "yolov8n.hef"
    # COCO class ids. "airplane" confirms aircraft; "bird" lets the detector
    # actively reject the many birds that otherwise look like distant aircraft.
    airplane_class_id: int = 4
    bird_class_id: int = 14
    detect_conf: float = 0.25
    # If a track's region reads as a bird above this confidence (and beats the
    # airplane score), it is discarded rather than emitted as a takeoff.
    bird_reject_conf: float = 0.35
    # Upscale small/distant object crops to at least this long-edge size before
    # running the detector, so tiny aircraft are still recognizable.
    detect_crop_min_size: int = 320

    # --- Frame sampling ---------------------------------------------------
    sample_fps: float = 8.0
    max_frame_width: int = 960  # downscale wide frames before processing

    # --- Motion (MOG2) ----------------------------------------------------
    mog2_history: int = 200
    mog2_var_threshold: float = 25.0
    mog2_warmup_frames: int = 5  # prime the background model before trusting it
    min_blob_area_frac: float = 0.0002  # of frame area; reject tiny noise
    max_blob_area_frac: float = 0.5     # reject whole-frame lighting changes

    # --- Tracking ---------------------------------------------------------
    min_track_frames: int = 5  # tracks shorter than this are ignored
    # Gating distance for matching a blob to a track, as a fraction of the frame
    # diagonal — generous enough to follow fast aircraft between sampled frames.
    track_match_distance_frac: float = 0.12
    track_max_age_frames: int = 6  # drop a track unseen for this many frames

    # --- Trajectory / takeoff classification ------------------------------
    # Fractions are of frame height/width unless noted.
    takeoff_min_rise_frac: float = 0.12      # net upward motion to count as ascending
    takeoff_min_displacement_frac: float = 0.10
    # Straightness = net_displacement / path_length. Aircraft fly near-straight
    # paths (~1.0); erratic birds score much lower. Key filter against birds.
    min_straightness: float = 0.80
    # Climb angle (degrees from horizontal) separating airplane vs helicopter ascent.
    helicopter_min_climb_angle_deg: float = 55.0
    airplane_max_climb_angle_deg: float = 35.0
    hover_speed_frac: float = 0.01  # per-frame centroid speed below this = hovering

    # --- Web --------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Derived media directories ---------------------------------------
    @property
    def incoming_dir(self) -> Path:
        return self.data_dir / "incoming"

    @property
    def processing_dir(self) -> Path:
        return self.data_dir / "processing"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "observer.db"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    def ensure_dirs(self) -> None:
        for d in (
            self.incoming_dir,
            self.processing_dir,
            self.processed_dir,
            self.clips_dir,
            self.thumbs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
