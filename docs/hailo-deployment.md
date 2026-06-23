# Observer — Hailo Deployment Guide

A step-by-step runbook for running Observer's aircraft detection on a Raspberry Pi
with a Hailo accelerator.

---

## Why this approach

The detector that reliably finds the small, distant helicopters in this footage is
**YOLO-World** (open-vocabulary). It works because you prompt it with the word
"aircraft" — but that prompting needs a CLIP text encoder baked in, which the
Hailo chip **cannot run**. Stock COCO models *do* compile to Hailo but can't see
these aircraft (they score at the noise floor).

The route that works is **distillation**:

1. Use YOLO-World (on a desktop) as a *teacher* to auto-label your footage.
2. Train a small **YOLOv8n** on those labels — YOLOv8n is a first-class Hailo
   model, and trained on *your* scene it can do what stock COCO couldn't.
3. Compile that YOLOv8n to a Hailo `.hef` and run it on the Pi.

```
camera Pi (motion capture) ──clips──► processor Pi (Hailo) ──► dashboard
                                          runs: observer serve
```

---

## Where each step runs

| Steps | Machine | Why |
|------|---------|-----|
| 0–4  | **x86_64 Linux desktop** | The Hailo Dataflow Compiler is x86-only; training wants a GPU |
| 5    | **processor Pi** | HailoRT runs the compiled HEF on the chip |

> The Pi (ARM) runs *inference* (HailoRT), not the *compiler*. Build the HEF on a
> desktop and copy it over.

---

## Prerequisites

**Desktop (x86_64 Linux):**
- This repo installed: `pip install -e .`
- The Hailo Dataflow Compiler + Hailo Model Zoo (`hailomz`) — from the
  [Hailo Developer Zone](https://hailo.ai/developer-zone/) (free account).
- A GPU is strongly recommended for training (CPU works but is slow).

**Processor Pi:**
- HailoRT installed and the chip detected (`hailortcli fw-control identify`).
- This repo installed: `pip install -e .`
- Know your chip: **Hailo-8** or **Hailo-8L** (affects `--hw-arch`).

---

## Step 0 — Label and import your footage (the training data)

The teacher only labels clips you've told it *contain* aircraft, so a hand-labelled
set is the foundation. **Distillation needs tens or more `aircraft` clips**, not a
handful — the more you label, the better the trained model.

```bash
# label clips by hand (needs the GUI build of OpenCV — see the script header)
python tools/label_clips.py            # A = aircraft, D = none  →  labels.csv

# import labels into the DB (records which clips are aircraft/none + thumbnails)
observer import-labels labels.csv
```

---

## Step 1 — Build the training dataset (desktop)

Distil YOLO-World into a YOLO-format detection dataset: confident teacher boxes on
`aircraft` clips become labels; frames from `none` clips become background
(teaching the model to ignore birds/clutter).

```bash
observer build-dataset --out dataset
```

Useful flags:
- `--frames-per-clip 20` — max positive frames kept per aircraft clip
- `--neg-frames-per-clip 10` — background frames kept per clip
- `--conf 0.25` — minimum teacher confidence to accept a box
- `--dir /path/to/clips` — where the clips live (defaults to the KINGSTON path)

Output (`dataset/`):
```
images/{train,val}/<clip>_<idx>.jpg
labels/{train,val}/<clip>_<idx>.txt     # "0 xc yc w h" normalized, per box
data.yaml
```

**Sanity-check before training:** open a few `images/train/*.jpg` and confirm the
corresponding `labels/*.txt` boxes land on the aircraft. Garbage labels here =
garbage model.

---

## Step 2 — Train YOLOv8n (desktop)

```bash
yolo detect train model=yolov8n.pt data=dataset/data.yaml imgsz=1280 epochs=100
```

- `imgsz=1280` keeps the small distant aircraft resolvable. If recall is poor, the
  main lever is keeping resolution high; if it's slow on the Pi later, retrain at a
  smaller `imgsz`.
- Best weights land in `runs/detect/train/weights/best.pt`.
- Review `runs/detect/train/` (PR curve, val predictions) before continuing.

---

## Step 3 — Export to ONNX (desktop)

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=1280 opset=11
# → runs/detect/train/weights/best.onnx
```

---

## Step 4 — Compile to a Hailo HEF (desktop)

Use the Hailo Model Zoo. Set `--hw-arch` to match your chip (`hailo8` or
`hailo8l`). The calibration images (for INT8 quantization) come from your dataset.

```bash
hailomz compile yolov8n \
    --ckpt runs/detect/train/weights/best.onnx \
    --calib-path dataset/images/train \
    --hw-arch hailo8l \
    --classes 1
# → yolov8n.hef   (rename to aircraft_yolov8n.hef)
```

> Compile **with NMS on-chip** (the Model Zoo default for YOLOv8). The Observer
> Hailo backend expects that output format.

---

## Step 5 — Deploy on the processor Pi

```bash
# copy the compiled model to the Pi
scp aircraft_yolov8n.hef  pi-processor:~/observer/models/

# on the Pi: confirm the model's input/output layout
hailortcli parse-hef models/aircraft_yolov8n.hef

# run Observer with the Hailo backend
OBSERVER_DETECTOR_BACKEND=hailo \
OBSERVER_HAILO_HEF_PATH=models/aircraft_yolov8n.hef \
  observer serve            # dashboard at http://<pi>:8000
```

If `parse-hef` shows a different output layout than the NMS-on-chip format, adjust
`_postprocess` in `observer/pipeline/detector/hailo_backend.py` to match. *(This is
the one part that can't be validated off-device.)*

---

## Clip forwarding (camera Pi → processor Pi)

The processor Pi processes whatever lands in `data/incoming/`. Have the camera Pi
push motion clips there. Simplest options:

- **rsync on a timer / motion-event hook:**
  ```bash
  rsync -av --remove-source-files /var/lib/motion/  pi-processor:~/observer/data/incoming/
  ```
- **Syncthing:** share the camera's capture folder into the processor's
  `data/incoming/`.

---

## Tuning the verdict (optional)

Once detection runs, the per-clip aircraft yes/no decision is tuned the same way as
the desktop version — measure against your labels and adjust thresholds:

```bash
observer eval --sweep
export OBSERVER_PRESENT_CONF=0.30 OBSERVER_MIN_HIT_FRAMES=3   # apply recommended values
```

---

## Checklist

- [ ] Labelled a meaningful set of clips (tens+ aircraft) and imported them
- [ ] `observer build-dataset` run; spot-checked that labels land on aircraft
- [ ] YOLOv8n trained; val results look reasonable
- [ ] Exported to ONNX
- [ ] Compiled to HEF with correct `--hw-arch` (8 vs 8L)
- [ ] HEF copied to the Pi; `hailortcli parse-hef` output matches the backend
- [ ] `observer serve` runs on the Pi with the `hailo` backend
- [ ] Camera Pi forwarding clips into `data/incoming/`
- [ ] Thresholds tuned with `observer eval --sweep`

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `build-dataset`: "No labelled clips" | Run `observer import-labels` first; check `--dir` points at the clips |
| Trained model misses aircraft | Too few/poor labels, or `imgsz` too low — label more, keep `imgsz=1280` |
| `hailomz compile` errors on an op | Re-export with `opset=11`; check the op is in the Hailo support list |
| Detections empty / wrong on the Pi | Output layout differs — compare `hailortcli parse-hef` to `_postprocess` |
| Wrong `--hw-arch` | Hailo-8 and Hailo-8L are not interchangeable; recompile for the right one |
| Lots of false positives (birds) | Add more `none`/background clips, raise `OBSERVER_PRESENT_CONF` |
