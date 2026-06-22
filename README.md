# Observer

Detect **airplanes and helicopters taking off** in footage from a remote camera,
and render the results on a live web dashboard.

Clips (2–60s, color, daytime, no audio) are dropped into a watched folder. Each
clip is processed to find moving aircraft, decide whether the motion is a
takeoff, and classify airplane vs. helicopter from the trajectory. Detections are
shown on a dashboard with live processing status.

## How it works

```
camera ──sync──> data/incoming/ ──watcher──> queue ──> processor ──> SQLite + media
                                                          │
                          motion (MOG2) → tracking (centroid) → detector (YOLO)
                          → trajectory classifier (takeoff? airplane/helicopter?)
                                                          │
                                          event bus ──SSE──> live dashboard
```

- **Motion**: a fixed daytime camera has a near-static background, so MOG2
  background subtraction cheaply isolates moving objects.
- **Tracking**: a nearest-centroid tracker stitches per-frame blobs into
  trajectories (robust to small, fast-moving aircraft).
- **Detection**: a pretrained YOLO confirms aircraft and supplies a class hint.
- **Trajectory** (`observer/pipeline/trajectory.py`): sustained upward motion ⇒
  *takeoff*; a steep, low-horizontal-speed climb (with possible hover) ⇒
  *helicopter*; a fast, shallow climb ⇒ *airplane*.

## Detector backends

Inference sits behind a `Detector` protocol (`observer/pipeline/detector/`):

- `ultralytics` (default) — portable CPU/GPU YOLOv8.
- `hailo` — optional, runs a precompiled `.hef` on a Raspberry Pi + Hailo
  accelerator. Select via `OBSERVER_DETECTOR_BACKEND=hailo`.

## Setup

```bash
pip install -e ".[dev]"
```

## Run

```bash
# generate sample clips (if you have no real footage yet)
python scripts/make_sample.py --out data/incoming

# start the dashboard + ingestion worker
observer serve            # http://localhost:8000

# or process a single clip from the CLI
observer process data/incoming/sample_helicopter.mp4
```

Drop a clip into `data/incoming/` and watch it appear, process (with live
progress), and render as a detected event you can play back.

## Processing a large archive

Bulk-process an existing directory of clips in parallel. Originals are read in
place (not moved); results land in the same database/media the dashboard reads,
so you can `observer serve` alongside or after the run to review.

```bash
# fast first pass — trajectory-only, no torch needed, all CPU cores
observer batch /path/to/clips --recursive

# refine with the real detector once you've eyeballed the first pass
observer batch /path/to/clips --recursive --backend ultralytics
```

- `--backend none` (default) skips object detection entirely — fastest, and the
  takeoff/type decision is trajectory-based anyway. `ultralytics` adds the YOLO
  aircraft-confidence signal; `hailo` uses the RPi accelerator.
- The run is **resumable**: clips already completed are skipped on re-run (pass
  `--reprocess` to force). Use `--limit N` to trial a subset, `--workers N` to
  cap parallelism.

## Tests

```bash
pytest
```

## Configuration

All thresholds live in `observer/config.py` and can be overridden with
`OBSERVER_`-prefixed environment variables (e.g. `OBSERVER_SAMPLE_FPS=10`,
`OBSERVER_DETECTOR_BACKEND=hailo`).
