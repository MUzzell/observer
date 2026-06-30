#!/usr/bin/env python3
"""Motion-triggered A/V capture for the camera Pi — a picamera2 replacement for `motion`.

One camera, two streams pulled at once:
    lores (small)  -> cheap frame-difference motion detection on the CPU
    main  (large)  -> H.264, kept in a rolling in-memory buffer, written only on an event

A circular buffer holds the last few seconds of encoded video, so each clip starts
*before* the trigger (pre-roll). picamera2 has no audio path, so audio is captured on a
background thread straight from an ALSA mic and held in a matching rolling buffer — so
the saved audio covers the ENTIRE clip, pre-roll included, not just from the trigger on.

Each event produces two aligned, same-basename files, both starting at the same instant:
    <stamp>.h264   the video (raw H.264 elementary stream)
    <stamp>.wav    the audio (mono S16_LE)
There is no ffmpeg anywhere in the capture path. A single muxed container would require a
muxer (ffmpeg / PyAV / GStreamer); the sidecar .wav instead lines up start-to-start with
the video, and the desktop/processor side can pair them by basename (or mux later if ever
wanted). Note: a raw .h264 has no container timestamps — fine for OpenCV frame extraction,
but not ideal for scrubbing in a player.

Timestamp: an ISO-8601 stamp (yyyy-mm-ddTHH:MM:SS) is burned into every main-stream frame
before encoding, so it appears in the recordings AND the live preview. The "camera offline"
placeholder carries a live clock too. (This is a visual overlay on the pixels, independent of
file/clip names — the recordings are self-evidencing even after the .h264 is muxed/renamed.)

Why this instead of the `motion` daemon: detection runs on the tiny lores stream, so the
recorded resolution (`--main-size`) is decoupled from CPU cost — raise it without melting
the Pi. The classic `motion` daemon detects at full res, which is what wedged the Pi 5.

Setup (on the Pi):
    sudo apt install -y python3-picamera2 python3-alsaaudio python3-opencv
    arecord -l                      # find your mic, e.g. card 1 -> --mic hw:1,0
    python3 tools/pi_capture.py --out /home/pi/clips --mic hw:1,0

picamera2 / alsaaudio are system (apt) packages, so run with the system /usr/bin/python3,
NOT a project venv (unless the venv was created with --system-site-packages).

Tune --threshold against your sky: birds and moving cloud will trip it. That's fine —
the model's `none`/background training teaches it to ignore them downstream.

Scheduling: `--active 08:00-18:00` records only during that daily window (the Pi's local
time — make sure its timezone is set). Outside the window the camera and H.264 encoder are
fully stopped, so the Pi isn't encoding all night; it just polls the clock every 15s and
spins the pipeline back up at the start time. A clip in progress at the cutoff is finalized,
not dropped. Windows that wrap past midnight (e.g. 22:00-06:00) work too.

Live preview: `--stream-port 8090` serves an MJPEG stream off the main camera stream for the
web app to embed — `<img src="http://<pi>:8090/stream.mjpg">` (or open the port's root for a
bare full-page view). It runs in this same process because only one process can own the
camera. The preview is live *only while the camera is on*: with `--active` set, the camera is
stopped outside the record window to spare the Pi, and the stream then shows a black
"CAMERA OFFLINE" placeholder with a live clock. `--stream-port 0` disables it. JPEG-encoding
the main stream costs some CPU on the Pi 5 (no hardware JPEG) — fine for a LAN preview; drop
it if you ever need the headroom.

Recommended companion: enable the hardware watchdog so a freeze self-reboots instead of
locking you out — set `RuntimeWatchdogSec=15` in /etc/systemd/system.conf.

Usage:
    python3 tools/pi_capture.py [--out DIR] [--main-size WxH] [--lores-size WxH]
                                [--fps N] [--pre-roll S] [--post-roll S]
                                [--threshold MSE] [--mic DEVICE | --no-audio]
                                [--active HH:MM-HH:MM] [--stream-port PORT]
"""

from __future__ import annotations

import argparse
import collections
import io
import signal
import sys
import threading
import time
import wave
from datetime import datetime, time as dtime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:                          # hard requirement: timestamps are mandatory
    sys.exit("This tool needs OpenCV for the timestamp overlay — sudo apt install -y python3-opencv")

from picamera2 import Picamera2, MappedArray
from picamera2.encoders import H264Encoder, JpegEncoder
from picamera2.outputs import CircularOutput, FileOutput

_stop = False
IDLE_POLL_S = 15        # how often to re-check the clock while outside the record window


def _handle_stop(signum, frame) -> None:
    """Flip the graceful-stop flag so the main loop can finalize an in-progress clip."""
    global _stop
    _stop = True


def _parse_size(text: str) -> tuple[int, int]:
    w, h = (int(v) for v in text.lower().split("x"))
    return w, h


def _parse_hm(text: str) -> dtime:
    parts = text.split(":")
    return dtime(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


def _parse_window(text: str) -> tuple[dtime, dtime]:
    """Parse 'HH:MM-HH:MM' (or 'H-H') into (start, end) times."""
    start, end = (part.strip() for part in text.split("-"))
    return _parse_hm(start), _parse_hm(end)


def in_window(now: dtime, start: dtime, end: dtime) -> bool:
    """True if `now` is within [start, end). Handles windows that wrap past midnight."""
    if start <= end:
        return start <= now < end
    return now >= start or now < end


class AudioRing:
    """Continuously captures mono S16_LE audio into a rolling buffer.

    While idle it keeps only the last `pre_roll` seconds, matching the video pre-roll.
    Call begin() at the trigger to stop trimming and accumulate through the whole event,
    then write_wav() at clip-close to flush everything from `pre_roll` before the trigger.
    """

    def __init__(self, device: str, pre_roll: float, rate: int = 48000,
                 channels: int = 1, period: int = 1024) -> None:
        import alsaaudio  # local import so video-only mode needs no alsaaudio install

        self.rate, self.channels = rate, channels
        self._pre_frames = int(pre_roll * rate)
        self._buf: collections.deque[tuple[int, bytes]] = collections.deque()
        self._total = 0
        self._recording = False
        self._stop = False
        self._lock = threading.Lock()
        self._pcm = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device=device,
            channels=channels, rate=rate, format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=period,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            n, data = self._pcm.read()
            if n <= 0:                       # overrun / no data this period
                continue
            with self._lock:
                self._buf.append((n, data))
                self._total += n
                if not self._recording:      # idle: keep only ~pre_roll seconds
                    while self._buf and self._total - self._buf[0][0] >= self._pre_frames:
                        frames, _ = self._buf.popleft()
                        self._total -= frames

    def begin(self) -> None:
        """Stop trimming; from now the buffer grows through the event."""
        with self._lock:
            self._recording = True

    def write_wav(self, path: Path) -> None:
        with self._lock:                     # snapshot fast, then write outside the lock
            data = b"".join(chunk for _, chunk in self._buf)
            self._buf.clear()
            self._total = 0
            self._recording = False
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)               # S16_LE = 2 bytes/sample
            wf.setframerate(self.rate)
            wf.writeframes(data)

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=2)


class StreamingOutput(io.BufferedIOBase):
    """Holds the latest JPEG frame; MJPEG readers wait on the condition for the next one."""

    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.condition = threading.Condition()

    def write(self, buf) -> int:
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)


def _make_stream_handler(output: StreamingOutput):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:          # silence per-request console spam
            pass

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                page = (b"<html><body style='margin:0;background:#000'>"
                        b"<img src='/stream.mjpg' style='width:100%;height:auto'></body></html>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()
                try:
                    while True:
                        with output.condition:
                            # timeout so an idle camera (outside the record window) doesn't
                            # block the handler forever — just keep waiting for frames.
                            if not output.condition.wait(timeout=5):
                                continue
                            frame = output.frame
                        self.wfile.write(b"--FRAME\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass                                # client navigated away
            else:
                self.send_error(404)

    return Handler


def start_stream_server(port: int, output: StreamingOutput) -> None:
    """Launch a threaded MJPEG HTTP server in the background (daemon thread)."""
    server = ThreadingHTTPServer(("", port), _make_stream_handler(output))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()


def make_offline_frame(size: tuple[int, int], text: str = "CAMERA OFFLINE") -> bytes | None:
    """Render a black JPEG with centered text + the current ISO timestamp, shown on the preview
    while the camera is off. Regenerated each second so the offline screen still ticks a clock."""
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, w / 900)
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    cv2.putText(img, text, ((w - tw) // 2, h // 2), font, scale, (60, 60, 220), 2, cv2.LINE_AA)
    stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    sscale = max(0.5, w / 1700)
    (sw, _sh), _ = cv2.getTextSize(stamp, font, sscale, 1)
    cv2.putText(img, stamp, ((w - sw) // 2, h // 2 + th + 20), font, sscale, (200, 200, 200), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("/home/pi/clips"), help="output dir for clips")
    p.add_argument("--main-size", type=_parse_size, default=(1280, 720), metavar="WxH",
                   help="recorded clip resolution (default 1280x720)")
    p.add_argument("--lores-size", type=_parse_size, default=(320, 240), metavar="WxH",
                   help="detection resolution; keep width a multiple of 32 (default 320x240)")
    p.add_argument("--fps", type=int, default=30, help="capture frame rate (default 30)")
    p.add_argument("--pre-roll", type=float, default=3.0, help="seconds kept before the trigger (default 3)")
    p.add_argument("--post-roll", type=float, default=3.0, help="keep recording this long after motion stops (default 3)")
    p.add_argument("--threshold", type=float, default=7.0, help="motion sensitivity as luma MSE; higher = less twitchy (default 7)")
    p.add_argument("--mic", default="default", help="ALSA capture device from `arecord -l`, e.g. hw:1,0 (default 'default')")
    p.add_argument("--no-audio", action="store_true", help="record video only (skip the mic)")
    p.add_argument("--active", type=_parse_window, default=None, metavar="HH:MM-HH:MM",
                   help="only record during this daily window in the Pi's local time, e.g. "
                        "08:00-18:00 (default: always). Outside it the camera + encoder stop "
                        "so the Pi isn't encoding all night.")
    p.add_argument("--stream-port", type=int, default=8090, metavar="PORT",
                   help="serve a live MJPEG preview on this port for the web app "
                        "(<img src=http://pi:PORT/stream.mjpg>); 0 disables (default 8090)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    lw, lh = args.lores_size
    args.out.mkdir(parents=True, exist_ok=True)

    audio: AudioRing | None = None
    if not args.no_audio:
        try:
            audio = AudioRing(args.mic, args.pre_roll)
            audio.start()
        except Exception as exc:             # mic missing/busy — degrade to video-only
            print(f"audio disabled ({exc}); recording video only", file=sys.stderr, flush=True)
            audio = None

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": args.main_size},
        lores={"size": args.lores_size, "format": "YUV420"},
        controls={"FrameRate": args.fps},
    ))

    # Burn an ISO-8601 timestamp onto the main stream every frame, BEFORE encoding — so it
    # lands in both the recordings and the live preview (both read the main stream). Drawn
    # white over a black outline for legibility against bright sky. Detection uses lores, so
    # the overlay never affects motion sensing.
    ts_scale = max(0.5, args.main_size[0] / 1280)
    ts_org = (10, max(24, int(34 * ts_scale)))
    ts_th = max(1, round(2 * ts_scale))

    def _timestamp_overlay(request) -> None:
        stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        with MappedArray(request, "main") as m:
            cv2.putText(m.array, stamp, ts_org, cv2.FONT_HERSHEY_SIMPLEX, ts_scale, (0, 0, 0, 255), ts_th + 2, cv2.LINE_AA)
            cv2.putText(m.array, stamp, ts_org, cv2.FONT_HERSHEY_SIMPLEX, ts_scale, (255, 255, 255, 255), ts_th, cv2.LINE_AA)

    picam2.pre_callback = _timestamp_overlay

    stream_output: StreamingOutput | None = None
    pipeline_live = threading.Event()        # set only while the camera is actually streaming
    if args.stream_port:
        stream_output = StreamingOutput()
        start_stream_server(args.stream_port, stream_output)
        if make_offline_frame(args.main_size) is not None:
            def _push_offline() -> None:     # "camera offline" + live clock whenever the camera is down
                while not _stop:
                    if not pipeline_live.is_set():
                        frame = make_offline_frame(args.main_size)
                        if frame is not None:
                            stream_output.write(frame)
                    time.sleep(1.0)
            threading.Thread(target=_push_offline, daemon=True).start()
        print(f"live preview: http://<pi>:{args.stream_port}/stream.mjpg", flush=True)

    streaming = False                        # camera + H.264 encoder running
    recording = False                        # currently writing a clip
    prev = None
    last_motion = 0.0
    clip: Path | None = None
    circular: CircularOutput | None = None

    def start_pipeline() -> None:
        nonlocal streaming, circular, prev
        pipeline_live.set()                               # stop the offline-placeholder pusher
        circular = CircularOutput(buffersize=int(args.pre_roll * args.fps))   # in-memory video ring buffer
        picam2.start_recording(H264Encoder(), circular)   # fresh encoder per session
        if stream_output is not None:                     # live MJPEG preview off the main stream
            picam2.start_encoder(JpegEncoder(), FileOutput(stream_output), name="main")
        streaming, prev = True, None

    def stop_clip() -> None:
        nonlocal recording, clip
        circular.stop()                      # back to ring-buffer-only
        if audio is not None:
            audio.write_wav(clip.with_suffix(".wav"))
        print(f"  saved {clip.stem}.h264" + ("" if audio is None else " + .wav"), flush=True)
        recording = False
        clip = None

    def stop_pipeline() -> None:
        nonlocal streaming
        if recording:
            stop_clip()                      # finalize a clip in progress (e.g. at the cutoff)
        picam2.stop_recording()              # stops the encoder — no CPU burn off-hours
        streaming = False
        pipeline_live.clear()                # resume pushing the "camera offline" placeholder

    window = "always" if args.active is None else f"{args.active[0]:%H:%M}-{args.active[1]:%H:%M}"
    print(f"watching for motion (main={args.main_size}, lores={args.lores_size}, "
          f"audio={'off' if audio is None else args.mic}, record window={window})…", flush=True)
    try:
        while not _stop:
            if args.active is not None and not in_window(datetime.now().time(), *args.active):
                if streaming:
                    print("  outside record window — idling camera", flush=True)
                    stop_pipeline()
                time.sleep(IDLE_POLL_S)       # cheap: camera + encoder are off
                continue
            if not streaming:
                print("  inside record window — capturing", flush=True)
                start_pipeline()

            # lores luma (Y) plane = top lh rows of the YUV420 array — cheap to diff
            y = picam2.capture_array("lores")[:lh, :lw].astype(np.int16)
            if prev is not None:
                mse = float(np.mean((y - prev) ** 2))
                now = time.time()
                if mse > args.threshold:
                    last_motion = now
                    if not recording:
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        clip = args.out / f"{stamp}.h264"
                        circular.fileoutput = str(clip)
                        circular.start()             # flush video pre-roll + keep writing
                        if audio is not None:
                            audio.begin()            # stop trimming; audio now spans the whole clip
                        recording = True
                        print(f"  motion (mse={mse:.0f}) -> {clip.stem}", flush=True)
                elif recording and now - last_motion > args.post_roll:
                    stop_clip()
            prev = y
    finally:
        if streaming:
            stop_pipeline()
        if audio is not None:
            audio.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
