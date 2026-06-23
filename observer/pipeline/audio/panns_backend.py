"""Audio aircraft detection via PANNs (CNN14 trained on AudioSet).

AudioSet includes the classes we want — "Helicopter", "Fixed-wing aircraft",
"Aircraft", "Aircraft engine", "Propeller", "Jet engine" — so this is the audio
analogue of using YOLO-World for vision: an off-the-shelf model that already
knows the target. The clip is windowed and each window scored by the maximum
probability across the configured aircraft classes, giving a per-window
confidence the standard decision logic aggregates.

Needs the optional audio deps: ``pip install -e ".[audio]"`` (panns_inference,
librosa). Both are imported lazily. The model (~80 MB CNN14) downloads on first
use. NOTE: the aircraft-vs-bird/wind separation must be validated on real
recordings before trusting the thresholds — this code is the plumbing, not a
guarantee.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from observer.config import Settings


class PannsDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._class_idx: list[int] = []

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from panns_inference import AudioTagging, labels  # type: ignore

        self._model = AudioTagging(checkpoint_path=None, device="cpu")
        name_to_idx = {name: i for i, name in enumerate(labels)}
        self._class_idx = [
            name_to_idx[n]
            for n in self._settings.audio_aircraft_classes
            if n in name_to_idx
        ]
        if not self._class_idx:
            raise ValueError(
                "None of audio_aircraft_classes matched AudioSet labels: "
                f"{self._settings.audio_aircraft_classes}"
            )

    def scan(self, wav_path: Path) -> list[float]:
        self._ensure_model()
        import librosa  # type: ignore

        sr = self._settings.audio_sample_rate
        y, _ = librosa.load(str(wav_path), sr=sr, mono=True)

        win = int(self._settings.audio_window_s * sr)
        hop = max(1, int(self._settings.audio_hop_s * sr))
        if len(y) < win:
            y = np.pad(y, (0, win - len(y)))

        confidences: list[float] = []
        for start in range(0, len(y) - win + 1, hop):
            seg = y[start:start + win].astype(np.float32)
            clipwise, _ = self._model.inference(seg[None, :])  # (1, 527)
            confidences.append(float(clipwise[0, self._class_idx].max()))
        return confidences
