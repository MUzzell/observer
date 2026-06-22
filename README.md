# Observer

Answer one question per clip from a remote camera: **is there an aircraft in it?**

Clips (a few seconds, color, daytime, no audio) are dropped into a watched
folder. Each clip is scanned for aircraft and the verdict — aircraft present or
not, with an evidence frame — is shown on a live dashboard.

## How it works

```
camera ──sync──> data/incoming/ ──watcher──> queue ──> processor ──> SQLite + evidence
                                                          │
                              sample frames ─► YOLO-World detector (open-vocab,
                              prompt "aircraft", full-frame @1280) ─► per-clip
                              decision (persistence + peak confidence)
                                                          │
                                          event bus ──SSE──> live dashboard
```

The aircraft in this footage are small, distant helicopters near the treeline.
Stock COCO YOLO can't see them (and has no helicopter class), and small
open-vocab models score at the noise floor. What works — verified on real
clips — is **YOLO-World-X**, prompted with the single word **`"aircraft"`**, run
**full-frame at imgsz=1280**: ~0.66–0.86 confidence on helicopters, while
bird-only clips stay below threshold. No training or labelling required.

A clip is flagged as containing an aircraft when it is detected on enough frames
(persistence rejects a lone false positive on a bird) **or** a single detection
is very strong (a brief but unambiguous pass). See
`observer/pipeline/aggregate.py`.

## Setup

```bash
pip install -e .
```

First run downloads the model weights (`yolov8x-worldv2.pt`, ~350 MB) once.

## Run

```bash
# live: watch a folder, process clips as they arrive, review on the dashboard
observer serve                       # http://localhost:8000
#   ...then drop clips into data/incoming/

# one-off: print the verdict for a single clip
observer process path/to/clip.mp4

# backfill: process an existing directory of clips (resumable, parallel)
observer batch path/to/clips --recursive
```

Drop a clip into `data/incoming/` and watch it appear, process (live progress),
and render as "aircraft" (with an evidence frame) or "no aircraft".

## Tuning

All knobs live in `observer/config.py`, overridable with `OBSERVER_*` env vars:

| Symptom | Knob |
|---|---|
| Aircraft clips being missed | lower `present_conf` / `min_hit_frames`, raise `detect_sample_fps` |
| Birds/clutter falsely flagged | raise `present_conf` / `min_hit_frames` / `strong_conf` |
| Detector too slow | lower `detect_sample_fps` or `detect_imgsz` (costs recall) |
| Different target wording | `aircraft_prompt` (e.g. add `"drone"`) |

## Tests

```bash
pytest
```

## Notes

- **Type hint** (airplane vs helicopter) is a best-effort secondary signal and
  currently unreliable; the yes/no verdict is the supported output.
- **Hailo / RPi**: the open-vocab detector's text encoder isn't Hailo-friendly,
  so the accelerator isn't used for this model. At ~10 clips/day on a fixed
  camera, CPU is comfortably fast enough.
