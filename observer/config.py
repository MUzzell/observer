"""Central configuration.

The pipeline answers one question per clip: **is there an aircraft in it?**
Detection uses an open-vocabulary detector (YOLO-World) prompted with the word
"aircraft", run full-frame at high resolution — the combination verified to
detect the small, distant helicopters in this footage where stock COCO models
score at the noise floor. All knobs are overridable via ``OBSERVER_*`` env vars.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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
    # "yoloworld" (open-vocab, default), "hailo" (compiled HEF on the RPi), or
    # "none" (no-op, for tests).
    detector_backend: str = "yoloworld"
    yoloworld_weights: str = "yolov8x-worldv2.pt"
    # Hailo: path to the compiled HEF and the model's square input size.
    hailo_hef_path: Path = REPO_ROOT / "models" / "aircraft_yolov8n.hef"
    hailo_imgsz: int = 640
    # Prompt used for the yes/no decision. A single clean class word works far
    # better than a competing list (verified on real footage).
    aircraft_prompt: tuple[str, ...] = ("aircraft",)
    # Secondary prompts used only to guess type on the best frame (nice-to-have).
    type_prompts: tuple[str, ...] = ("helicopter", "airplane")
    enable_type_hint: bool = True
    detect_imgsz: int = 1280
    # Low floor: collect all candidate detections; thresholds below decide.
    detect_conf: float = 0.10

    # --- Frame sampling ---------------------------------------------------
    # Frames per second to sample for detection. Speed is a non-issue (~10
    # clips/day), so sample densely enough to catch brief passes.
    detect_sample_fps: float = 3.0

    # --- Per-clip decision (video) ---------------------------------------
    # A frame "hits" when its best aircraft confidence reaches present_conf.
    present_conf: float = 0.30
    # Aircraft present if it hits on this many frames (persistence rejects the
    # occasional one-frame false positive on a bird/treeline) ...
    min_hit_frames: int = 3
    # ... OR a single very strong detection (covers very brief but clear passes).
    strong_conf: float = 0.55

    # --- Audio detection --------------------------------------------------
    # How the per-clip verdict is formed: "visual" (default), "audio", or
    # "fusion" (aircraft if either modality fires).
    detection_mode: str = "visual"
    audio_backend: str = "panns"        # "panns" (AudioSet CNN) or "none"
    audio_sample_rate: int = 32000      # model input rate; clips resampled to this
    audio_window_s: float = 1.0
    audio_hop_s: float = 0.5
    # AudioSet class names that count as "aircraft" (the teacher's vocabulary).
    audio_aircraft_classes: tuple[str, ...] = (
        "Aircraft",
        "Aircraft engine",
        "Helicopter",
        "Fixed-wing aircraft, airplane",
        "Propeller, airscrew",
        "Jet engine",
    )
    # Audio thresholds (AudioSet probabilities run lower than detector scores).
    # Starting points — tune against real recordings with `observer eval`.
    audio_present_conf: float = 0.10
    audio_min_hit_frames: int = 3
    audio_strong_conf: float = 0.30

    # --- Web --------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    # Live camera MJPEG stream, embedded as an <img> source. Empty to hide it.
    camera_stream_url: str = "http://172.16.50.10:8081/"

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
