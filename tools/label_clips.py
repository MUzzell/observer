#!/usr/bin/env python3
"""Rapid keyboard labeller for clips: aircraft vs no-aircraft.

Plays each clip on a loop in an OpenCV window; you press one key to label it and
it jumps straight to the next. Labels are written to a CSV after every decision
(so it's crash-safe) and already-labelled clips are skipped on the next run, so
you can stop and resume anytime.

Keys:
    A   aircraft present
    D   no aircraft
    S   skip (decide later — stays unlabelled)
    Z   undo (go back to the previous clip and relabel)
    R   replay current clip from the start
    SPACE  pause / resume
    + / -  zoom in / out
    . / ,  faster / slower playback
    Q or ESC  quit and save

Requires the GUI build of OpenCV (`pip install opencv-python`), NOT
opencv-python-headless. Run it outside the project's headless venv, e.g.:

    python3 -m venv ~/.venv-label && ~/.venv-label/bin/pip install opencv-python
    ~/.venv-label/bin/python tools/label_clips.py

Usage:
    python tools/label_clips.py [--dir DIR] [--out labels.csv] [--relabel]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit("OpenCV not installed. Run: pip install opencv-python")

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
DEFAULT_DIR = "/run/media/muzzell/KINGSTON/observer/"
LABEL_AIRCRAFT = "aircraft"
LABEL_NONE = "none"
WINDOW = "Observer labeller"


def check_gui() -> None:
    """Fail early with a helpful message if this is the headless OpenCV build."""
    info = cv2.getBuildInformation()
    if "GUI:" in info and "NONE" in info.split("GUI:", 1)[1][:60].upper():
        sys.exit(
            "This OpenCV build has no GUI support (likely opencv-python-headless).\n"
            "Install the GUI build in a separate env:\n"
            "  python3 -m venv ~/.venv-label\n"
            "  ~/.venv-label/bin/pip install opencv-python\n"
            "  ~/.venv-label/bin/python tools/label_clips.py"
        )


def load_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0] != "filename":  # skip header
                    labels[row[0]] = row[1]
    return labels


def save_labels(path: Path, labels: dict[str, str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for name, label in labels.items():
            w.writerow([name, label])
    tmp.replace(path)


def overlay(frame, lines: list[tuple[str, tuple]], scale: float):
    h, w = frame.shape[:2]
    disp = cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_LINEAR)
    # Dark banner at the top for legibility.
    band_h = 26 + 22 * len(lines)
    cv2.rectangle(disp, (0, 0), (disp.shape[1], band_h), (0, 0, 0), -1)
    y = 24
    for text, color in lines:
        cv2.putText(disp, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
                    cv2.LINE_AA)
        y += 22
    return disp


def draw_progress(disp, frac: float) -> None:
    """Draw a playback progress bar along the bottom of the displayed frame."""
    h, w = disp.shape[:2]
    frac = max(0.0, min(1.0, frac))
    bar_h = 6
    cv2.rectangle(disp, (0, h - bar_h), (w, h), (50, 50, 50), -1)
    cv2.rectangle(disp, (0, h - bar_h), (int(w * frac), h), (120, 230, 120), -1)


def label_one(path: Path, index: int, total: int, current: str | None,
              scale: float, speed: float) -> tuple[str, float, float]:
    """Play one clip until a decision key is pressed. Returns (action, scale, speed)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Read the first frame up front: a corrupt/truncated clip (e.g. "moov atom
    # not found") opens but yields no frames — bail instead of looping forever.
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return "unreadable", scale, speed
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cur_idx = 0
    paused = False
    while True:
        if not paused:
            ok, f = cap.read()
            if not ok or f is None:  # end of clip: rewind and loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cur_idx = 0
                ok, f = cap.read()
                if not ok or f is None:  # truly unreadable after rewind
                    cap.release()
                    return "unreadable", scale, speed
            else:
                cur_idx += 1
            frame = f
        cur = f"  [current: {current}]" if current else ""
        disp = overlay(
            frame,
            [
                (f"{index + 1}/{total}  {path.name}{cur}", (255, 255, 255)),
                ("A=aircraft   D=no aircraft   S=skip   Z=undo   R=replay",
                 (120, 230, 120)),
                (f"SPACE=pause   +/-=zoom   ,/.=speed ({speed:.2g}x)   Q=quit",
                 (180, 180, 180)),
            ],
            scale,
        )
        if total_frames > 0:
            draw_progress(disp, cur_idx / total_frames)
        cv2.imshow(WINDOW, disp)
        delay = 60 if paused else max(1, int(1000 / (fps * speed)))
        key = cv2.waitKey(delay) & 0xFF
        if key == 255:
            continue
        ch = chr(key).lower() if key < 128 else ""
        if ch == "a":
            action = "aircraft"
        elif ch == "d":
            action = "none"
        elif ch == "s":
            action = "skip"
        elif ch == "z":
            action = "undo"
        elif ch == "r":
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); cur_idx = 0; paused = False; continue
        elif ch == " ":
            paused = not paused; continue
        elif ch in ("+", "="):
            scale = min(4.0, scale + 0.25); continue
        elif ch in ("-", "_"):
            scale = max(0.5, scale - 0.25); continue
        elif ch in (".", ">"):
            speed = min(8.0, speed + 0.5); continue
        elif ch in (",", "<"):
            speed = max(0.25, speed - 0.5); continue
        elif ch == "q" or key == 27:
            action = "quit"
        else:
            continue
        cap.release()
        return action, scale, speed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR, help="directory of clips")
    ap.add_argument("--out", default="labels.csv", help="output CSV path")
    ap.add_argument("--recursive", action="store_true", help="search subfolders")
    ap.add_argument("--relabel", action="store_true",
                    help="include already-labelled clips")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="display zoom (default: auto, ~1280px wide)")
    ap.add_argument("--speed", type=float, default=2.0,
                    help="playback speed multiplier (default: 2.0)")
    args = ap.parse_args()

    check_gui()

    root = Path(args.dir)
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")
    pattern = "**/*" if args.recursive else "*"
    files = sorted(p for p in root.glob(pattern)
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not files:
        sys.exit(f"No video clips found in {root}")

    out = Path(args.out)
    labels = load_labels(out)

    # Auto display scale from the first clip so small aircraft are visible.
    scale = args.scale
    if scale <= 0:
        cap = cv2.VideoCapture(str(files[0]))
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640
        cap.release()
        scale = max(1.0, round(1280 / w * 4) / 4)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    todo = [f for f in files if args.relabel or f.name not in labels]
    print(f"{len(files)} clips · {len(files) - len(todo)} already labelled · "
          f"{len(todo)} to go")

    # Index into the full list so 'undo' can step back across skipped items.
    order = files if args.relabel else todo
    speed = max(0.25, args.speed)
    i = 0
    while i < len(order):
        f = order[i]
        action, scale, speed = label_one(
            f, i, len(order), labels.get(f.name), scale, speed)
        if action == "quit":
            break
        if action == "undo":
            i = max(0, i - 1)
            labels.pop(order[i].name, None)
            save_labels(out, labels)
            continue
        if action == "skip":
            i += 1
            continue
        if action == "unreadable":
            print(f"  ! skipping unreadable clip (corrupt/truncated): {f.name}")
        labels[f.name] = action  # "aircraft", "none", or "unreadable"
        save_labels(out, labels)
        i += 1

    cv2.destroyAllWindows()
    n_air = sum(v == LABEL_AIRCRAFT for v in labels.values())
    n_none = sum(v == LABEL_NONE for v in labels.values())
    n_bad = sum(v == "unreadable" for v in labels.values())
    print(f"\nSaved {out}: {n_air} aircraft · {n_none} none · "
          f"{n_bad} unreadable · {len(files) - len(labels)} unlabelled")


if __name__ == "__main__":
    main()
