# Observer — Audio Capture on the Camera Pi

How to capture an audio track alongside each motion clip, while keeping the camera
Pi dumb and low-load. All filtering and detection happens later on the processor
Pi — this doc only covers getting the sound recorded.

---

## Why a sidecar WAV (and not "audio in Motion")

**Motion cannot merge a local microphone into its recordings.** It only records
audio when the *source* is a stream that already carries it (an RTSP/netcam URL).
With a directly-attached Pi camera (CSI or USB/V4L2) plus a separate mic, there is
no motion.conf option to mux the mic in.

So instead of fighting Motion, we let it keep doing exactly what it does and
capture a **matching audio file per clip** using Motion's event hooks. For each
movie `clip.mp4`, we record `clip.wav` next to it.

This is the right fit for a barn Pi:
- `arecord` only runs *during* a clip — near-zero idle load.
- No ffmpeg, no continuous encoding, no extra always-on service.
- Muxing (if ever wanted) and all analysis happen on the processor Pi.

> If you genuinely need a single audio+video file produced at capture, the only
> Motion-native route is to expose the camera as a local RTSP stream that carries
> audio (rpicam-vid + ffmpeg + an RTSP server) and point Motion at it — but that's
> continuous encoding load on the Pi. Not recommended for this setup.

---

## Hardware

- A USB microphone is simplest (shows up as an ALSA capture device automatically).
  An I2S MEMS mic (e.g. INMP441) also works but needs device-tree/ALSA setup.
- **A windscreen (foam / dead-cat) is essential outdoors.** Wind rumble sits in the
  same low frequencies as a helicopter's rotor signature — without a windscreen
  you'll bury the signal you care about.

Find the mic's ALSA address:

```sh
arecord -l
# e.g. card 1: Device [USB Audio Device], device 0  ->  plughw:1,0
```

Quick capture test (Ctrl-C to stop, then play it back on the processor):

```sh
arecord -D plughw:1,0 -f S16_LE -c1 -r16000 test.wav
```

16 kHz mono is plenty for aircraft sound and matches common audio models (YAMNet),
while keeping files tiny.

---

## Setup

### 1. Two small scripts

`/home/pi/audio_start.sh` — start recording, named to match the clip:

```sh
#!/bin/sh
# $1 = full path of the movie Motion just started (from %f)
wav="${1%.*}.wav"
# -d 70 is a safety cap so it self-stops even if on_movie_end is missed
arecord -q -D plughw:1,0 -f S16_LE -c1 -r16000 -d 70 "$wav" &
echo $! > "${1}.apid"
```

`/home/pi/audio_stop.sh` — stop it when the clip ends:

```sh
#!/bin/sh
# $1 = full path of the movie Motion just finished (from %f)
[ -f "${1}.apid" ] && kill "$(cat "${1}.apid")" 2>/dev/null
rm -f "${1}.apid"
```

Make them executable:

```sh
chmod +x /home/pi/audio_start.sh /home/pi/audio_stop.sh
```

Adjust `plughw:1,0` to match your `arecord -l` output.

### 2. Wire them into Motion

In `motion.conf` (or the relevant camera's config):

```
on_movie_start /home/pi/audio_start.sh %f
on_movie_end   /home/pi/audio_stop.sh  %f
```

`%f` expands to the full path of the movie file for that event. Restart Motion:

```sh
sudo systemctl restart motion   # or: motion -b
```

---

## Verify

Trigger a recording (wave at the camera) and check that a `.wav` lands next to the
`.mp4`:

```sh
ls -la /var/lib/motion/      # or wherever target_dir points
# 2026-06-23T14-02-11.mp4
# 2026-06-23T14-02-11.wav    <- same basename
```

Confirm the WAV has real audio (not silence):

```sh
# duration & format
soxi 2026-06-23T14-02-11.wav 2>/dev/null || aplay --dump-hw-params /dev/null
# or just copy it to a desktop and listen
```

If the WAV is missing or empty, see Troubleshooting below.

---

## Shipping to the processor Pi

The `.wav` shares its basename with the `.mp4`, so your existing clip-forwarding
(rsync / Syncthing into the processor's `data/incoming/`) carries both as long as
it isn't filtered to `*.mp4` only. For rsync, include both:

```sh
rsync -av --include='*.mp4' --include='*.wav' --exclude='*' \
    /var/lib/motion/  pi-processor:~/observer/data/incoming/
```

On the processor Pi, the audio model runs on the WAV and is fused with (or replaces)
the visual verdict; combined audio+video playback, if wanted, is muxed there with a
single ffmpeg call. *(That processing side is built separately — validate the audio
model on a few real recordings first.)*

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| No `.wav` produced | Hooks not firing — check `on_movie_start/end` paths and that the scripts are `chmod +x`; check Motion's log |
| WAV is empty / silent | Wrong `plughw:` device — recheck `arecord -l`; test `arecord` manually |
| `arecord: device busy` | Another process holds the mic; ensure only the hook records |
| WAV cut short | Clip ran longer than the `-d` safety cap — raise `-d` above your max clip length |
| Constant low rumble | Wind on an unshielded mic — add a foam/dead-cat windscreen |
| `.apid` files left behind | `on_movie_end` not firing; harmless, but check the stop hook path |

---

## Summary

- Camera Pi: Motion unchanged + two tiny hook scripts → one `.wav` per `.mp4`.
- Negligible load; no encoding; nothing clever on the barn Pi.
- All audio filtering, detection, and (optional) muxing happen on the processor Pi.
