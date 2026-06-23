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

## Ground-truth labelling & evaluation

Build a labelled set by hand, then measure and tune the detector against it:

```bash
# 1. label clips by hand (needs the GUI build of OpenCV; see the script header)
python tools/label_clips.py                 # A=aircraft D=none -> labels.csv

# 2. import labels into the DB + dashboard (with thumbnails)
observer import-labels labels.csv

# 3. score the detector against your labels, and tune thresholds
observer eval --sweep
```

`eval` reports precision/recall/F1 against your labels, lists the mismatched
clips, and (with `--sweep`) recommends `present_conf` / `min_hit_frames`. Per-clip
scores are cached, so re-running is instant.

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

## Hailo deployment (two-Pi setup)

Target: **camera Pi** (motion capture) → **processor Pi** (Hailo) runs detection.

YOLO-World can't run on Hailo (its text encoder isn't compilable), and stock COCO
models can't see these aircraft. The route that works is **distillation**: use
YOLO-World as a teacher to label your footage, train a small YOLOv8n on it, and
compile *that* to Hailo. YOLOv8n is a first-class Hailo model.

> Steps 1–4 run on an **x86_64 Linux desktop** (the Hailo Dataflow Compiler is
> x86-only). Only step 5 runs on the Pi.

```bash
# 1. distil a dataset from clips you've labelled + imported (see above)
observer build-dataset --out dataset

# 2. train YOLOv8n on your footage (GPU desktop recommended)
yolo detect train model=yolov8n.pt data=dataset/data.yaml imgsz=1280 epochs=100

# 3. export to ONNX
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=1280 opset=11

# 4. compile to HEF (Hailo Model Zoo; --hw-arch hailo8 or hailo8l for your chip)
hailomz compile yolov8n \
    --ckpt runs/detect/train/weights/best.onnx \
    --calib-path dataset/images/train \
    --hw-arch hailo8l
```

```bash
# 5. on the PROCESSOR Pi: drop the HEF in models/ and run with the hailo backend
scp aircraft_yolov8n.hef pi-processor:~/observer/models/
# on the Pi (HailoRT installed):
OBSERVER_DETECTOR_BACKEND=hailo \
OBSERVER_HAILO_HEF_PATH=models/aircraft_yolov8n.hef \
  observer serve
```

The camera Pi forwards motion clips to the processor Pi's `data/incoming/` (e.g.
`rsync`/`scp` on motion-event, or Syncthing). The processor Pi runs `observer
serve`, which ingests, detects via Hailo, and serves the dashboard.

Notes & caveats:
- The Hailo backend (`observer/pipeline/detector/hailo_backend.py`) assumes the
  HEF was compiled **with NMS on-chip**; verify the output layout on-device with
  `hailortcli parse-hef` and adjust `_postprocess` if needed. **This part can't be
  validated off-device.**
- Model quality depends on how much you've labelled — distillation needs a decent
  number of `aircraft` clips (tens+), not a handful. Label, then re-run step 1.
- Tune detection size/recall by retraining at a different `imgsz`.

## Notes

- **Type hint** (airplane vs helicopter) is a best-effort secondary signal and
  currently unreliable; the yes/no verdict is the supported output.
