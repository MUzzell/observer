# Observer — Camera Pi (`reconone`) Setup

The barn Pi. Its only jobs: run **Motion** to capture clips, record a **sidecar WAV**
per clip, and forward both to the processor Pi `observer`. It must stay light — no
detection, no encoding circus. All analysis happens downstream.

This guide covers:
1. Adding **audio capture** alongside the existing Motion config.
2. Diagnosing the **lockups / undervoltage** seen when bumping resolution.
3. **Motion config changes** to push resolution up without locking the Pi.

> Audio wiring is documented in full in [`audio-capture.md`](audio-capture.md);
> this file is the operational companion (sit it next to Motion, keep it stable).

---

## 1. Audio capture (sidecar WAV)

Short version of [`audio-capture.md`](audio-capture.md) — Motion can't mux a local
mic, so we record a matching `clip.wav` for every `clip.mp4` using Motion's event
hooks. `arecord` only runs *during* a clip, so idle load is ~zero.

Find the mic, test it:

```sh
arecord -l                                   # note the card,device -> plughw:1,0
arecord -D plughw:1,0 -f S16_LE -c1 -r16000 test.wav    # Ctrl-C to stop
```

Two hook scripts (`/home/pi/audio_start.sh`, `/home/pi/audio_stop.sh`) — see
[`audio-capture.md`](audio-capture.md) for the exact contents — then wire into
`motion.conf`:

```
on_movie_start /home/pi/audio_start.sh %f
on_movie_end   /home/pi/audio_stop.sh  %f
```

```sh
chmod +x /home/pi/audio_*.sh
sudo systemctl restart motion
```

**Outdoors a foam/dead-cat windscreen is mandatory** — wind rumble sits in the same
low band as rotor noise and will bury the signal.

---

## 2. Lockups & undervoltage — diagnose first

A *hard lockup* when you raise resolution is almost never "CPU too busy" — it's
**power (undervoltage)**, **memory (OOM)**, or **thermal**. Higher resolution means
larger frames buffered in RAM *and* more software-encoding work *and* a fatter MJPEG
stream — all of which spike current draw and memory at once.

After a reboot following a lockup, check what actually happened:

```sh
# Power: any non-zero value = undervoltage/throttling occurred
vcgencmd get_throttled
#   0x0          -> clean
#   bit 0  (0x1)        under-voltage NOW
#   bit 16 (0x10000)    under-voltage HAS occurred since boot
#   bit 18 (0x40000)    throttling HAS occurred

# Temperature (throttles ~80–85°C)
vcgencmd measure_temp

# The smoking gun in the logs — OOM kills, under-voltage, throttling
sudo dmesg | grep -iE 'oom|under-volt|throttl|hung task'

# Live memory headroom while a capture runs
free -h
```

Interpretation:

| Finding | Cause | Fix |
|---|---|---|
| `get_throttled` non-zero, "under-voltage" in dmesg | **Undervoltage** — PSU/cable can't supply peak current | Use the official PSU (Pi 5: 27 W USB-C PD; Pi 4: 5 V/3 A), short/thick cable, no hub |
| `oom` / "Out of memory" in dmesg | **OOM** — frame buffers too large | Lower framerate, cut pre/post-capture & buffers (§3), add `gpu_mem`/zram |
| `measure_temp` ≥ 80°C | **Thermal** | Add a heatsink/fan; improve barn airflow |
| Clean, but still froze | Likely transient peak combining all three | Apply the §3 load cuts and re-test incrementally |

**Power is the usual culprit in a barn** — long/thin cables and shared adapters sag
under the extra current a higher resolution demands. Fix power before blaming the
resolution.

---

## 3. Motion config changes — more resolution, no lockup

The goal: raise capture resolution while cutting the *peak* load that triggers the
freeze. Edit the active config (`/etc/motion/motion.conf`, or the per-camera
`camera*.conf`).

### a. Pick an alignment-safe resolution

This is a **raw sensor** (no MJPEG; YUYV/Bayer). Width must be a **multiple of 64**,
height a **multiple of 16** — otherwise you get the green stride-padding bar.
`1280x720` is clean; `800x600` is not (800 isn't ÷64).

```
width 1280
height 720
```

### b. Drop the framerate (biggest single win)

Aircraft cross slowly and detection samples a few fps anyway. Fewer frames = less
encode work, less memory churn, lower current.

```
framerate 8          # 5–8 is plenty; 30 is wasteful here
```

### c. Use hardware H.264 encoding if available

Software encoding raw high-res frames is a major load source.

```
movie_codec mp4          # H.264 in MP4
```

- **Pi 4:** has a hardware H.264 *encoder* — this offloads the CPU. Good.
- **Pi 5:** has **no** hardware H.264 *encoder* (decode only). Encoding is on the
  CPU, so keep framerate/resolution modest, or capture clips and let the processor
  Pi do any re-encoding.

### d. Tame the live MJPEG stream (continuous load)

The stream the dashboard shows runs *all the time* — at high res it's a constant
drain. Cap it hard; it's a preview, not the recording.

```
stream_maxrate 5         # fps of the live stream
stream_quality 50        # JPEG quality of the stream
stream_localhost off     # leave reachable for the dashboard
```

### e. Shrink buffers / pre-capture (memory + freeze safety)

Each pre-capture frame is held in RAM at full resolution — the prime OOM risk when
you scale up.

```
pre_capture 2            # was maybe 5–10; each frame is now much bigger
post_capture 2
minimum_motion_frames 1
```

### f. Keep detection cheap

Motion only needs to *trigger*; the processor Pi does the real detection.

```
threshold 1500           # tune to your scene to avoid bird/cloud false triggers
# optionally analyse a downscaled frame; full-res only for the saved movie
```

Restart and watch it under load:

```sh
sudo systemctl restart motion
vcgencmd get_throttled        # re-check after a few triggers
sudo dmesg -w | grep -iE 'oom|under-volt|throttl'   # live
```

### Recommended starting profile (Pi 4)

```
width 1280
height 720
framerate 8
movie_codec mp4
stream_maxrate 5
stream_quality 50
pre_capture 2
post_capture 2
```

Bring resolution up **one step at a time**, re-running the §2 checks after each
change. If `get_throttled` ever goes non-zero, the ceiling is your **power supply**,
not Motion.

---

## 4. Forwarding to the processor Pi

The `.wav` shares the `.mp4` basename, so one transfer carries both. Push from here
(or pull from `observer` — see its install guide):

```sh
rsync -av --include='*.mp4' --include='*.wav' --exclude='*' \
    /var/lib/motion/  pi@observer.local:~/observer/data/incoming/
```

Set up an SSH key (`ssh-keygen` → `ssh-copy-id pi@observer.local`) so it runs
unattended from cron.

---

## Checklist

- [ ] Mic found (`arecord -l`), test WAV has real audio, windscreen fitted
- [ ] Hooks `chmod +x` and wired into `motion.conf`; `.wav` lands beside `.mp4`
- [ ] `vcgencmd get_throttled` = `0x0` under load (power is solid)
- [ ] Resolution is 64/16-aligned (no green bar)
- [ ] Framerate ≤ 8, stream rate capped, pre/post-capture small
- [ ] Clips + WAVs arriving in `observer:~/observer/data/incoming/`
