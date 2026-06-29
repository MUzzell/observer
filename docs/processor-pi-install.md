# Observer — Processor Pi (`observer`) Full Install

A from-scratch, **headless** build of the processing + web-client Pi: flash the SD
card, boot it with SSH already enabled (no keyboard/monitor), then install and run
Observer as a service. Hailo accelerator setup is referenced out to
[`hailo-deployment.md`](hailo-deployment.md).

This Pi:
- receives clips (+ sidecar WAVs) from the camera Pi `reconone`,
- runs detection (`observer serve` → worker + dashboard),
- serves the web UI on port 8000 and proxies the camera's live stream.

Target: **Raspberry Pi 5** (or Pi 4, 4 GB+), **Raspberry Pi OS Bookworm 64-bit
Lite**. Lite is correct — this is a server, no desktop.

---

## 1. Flash with `dd` and edit the boot config directly

Grab the **Raspberry Pi OS Bookworm 64-bit Lite** image
([raspberrypi.com/software/operating-systems](https://www.raspberrypi.com/software/operating-systems/)),
verify it, and write it with `dd`.

```sh
# identify the card FIRST — get the device wrong and you wipe a disk
lsblk

# decompress + write (replace sdX with your card; NO partition number)
xz -dc 2024-*-raspios-bookworm-arm64-lite.img.xz \
    | sudo dd of=/dev/sdX bs=4M conv=fsync status=progress

sudo sync
```

After `dd`, re-plug the card so the partitions re-read. Two appear: **`bootfs`**
(FAT, ~512 MB — what we edit) and `rootfs`. Mount `bootfs`:

```sh
sudo mkdir -p /mnt/bootfs
sudo mount /dev/sdX1 /mnt/bootfs       # sdX1 = the small FAT boot partition
```

Drop the headless settings into the boot partition before first boot:

```sh
# 1. enable SSH
sudo touch /mnt/bootfs/ssh

# 2. create the first user — this IS the username change.
#    Pi OS no longer ships a default "pi" account; the first field of
#    userconf.txt is the account name, so just set it to whatever you want.
#    We use "obs" throughout this guide — substitute your own and keep it
#    consistent in the paths and the systemd unit below.
printf 'obs:%s\n' "$(openssl passwd -6 'YOURPASSWORD')" \
    | sudo tee /mnt/bootfs/userconf.txt

# 3. Wi-Fi — SKIP if using Ethernet (recommended for this Pi)
sudo tee /mnt/bootfs/wpa_supplicant.conf >/dev/null <<'EOF'
country=GB
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
network={
    ssid="YOUR_SSID"
    psk="YOUR_WIFI_PASSWORD"
}
EOF
```

These three files are consumed by `firstboot`/`cloud-init` on first boot and then
removed. `ssh` + `userconf.txt` is the minimum to get in headless.

> Set the username **now**, at flash time. Renaming an account on a running Pi
> (`usermod -l` + `usermod -d` + group/home fix-ups) can't be done while that user
> is logged in and is easy to get wrong — doing it via `userconf.txt` sidesteps all
> of that.

Set the hostname so it comes up as `observer.local` (mDNS). The boot partition
doesn't own `/etc/hostname`, so either:
- set it on first login (`sudo hostnamectl set-hostname observer && sudo reboot`), or
- mount `rootfs` now and edit `/etc/hostname` + the `127.0.1.1` line in `/etc/hosts`.

Unmount cleanly and eject:

```sh
sudo umount /mnt/bootfs
sudo sync
```

> Prefer Ethernet for the processor Pi — steadier than Wi-Fi for moving clips and
> serving the dashboard. If on Ethernet you can skip `wpa_supplicant.conf` entirely.

---

## 2. First boot + SSH in

Insert the card, power the Pi, give it ~60–90 s, then from your desktop:

```sh
ssh obs@observer.local
```

If `observer.local` doesn't resolve (some networks block mDNS), find the IP from
your router's DHCP leases and `ssh obs@<ip>` instead.

Update the base system and reboot once:

```sh
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

---

## 3. System dependencies

```sh
sudo apt install -y git python3-venv python3-pip ffmpeg libgl1 rsync
```

- `ffmpeg` / `libgl1` — OpenCV decode + headless runtime libs.
- `rsync` — receives clips from the camera Pi (pull or push).

---

## 4. Get Observer + Python environment

```sh
cd ~
git clone <your-observer-repo-url> observer
cd observer

python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .          # core (FastAPI, OpenCV, Ultralytics, …)
```

The YOLO-World detector weights (`yolov8x-worldv2.pt`) are large — either commit
them with the repo, `scp` them over, or let Ultralytics fetch on first run. Confirm
`yolov8x-worldv2.pt` sits in the repo root (where `observer serve` looks).

> **CPU-only note:** YOLO-World-X @1280 is slow on a Pi CPU (tens of seconds/clip).
> That's fine for ~10 clips/day, but the **Hailo** path is the intended production
> detector — see step 7.

Quick smoke test:

```sh
observer --help
observer process samples/<some-clip>.mp4    # prints the verdict
```

---

## 5. Data directory + receiving clips

Observer watches `data/incoming/` (relative to the repo, per `config.py`). Clips +
their sidecar `.wav` land there from `reconone`.

```sh
mkdir -p ~/observer/data/incoming
```

Pick **one** transfer direction:

- **Camera Pi pushes** (simplest, set up on `reconone` — see its guide):
  ```sh
  rsync -av --include='*.mp4' --include='*.wav' --exclude='*' \
      /var/lib/motion/  obs@observer.local:~/observer/data/incoming/
  ```
- **Processor Pi pulls** (cron on `observer`):
  ```sh
  */5 * * * * rsync -av --remove-source-files \
      --include='*.mp4' --include='*.wav' --exclude='*' \
      obs@reconone.local:/var/lib/motion/  ~/observer/data/incoming/
  ```

For passwordless transfer, set up an SSH key between the two Pis
(`ssh-keygen` then `ssh-copy-id`).

---

## 6. Run as a service (auto-start on boot)

Create `/etc/systemd/system/observer.service`:

```ini
[Unit]
Description=Observer dashboard + ingestion worker
After=network-online.target
Wants=network-online.target

[Service]
User=obs
WorkingDirectory=/home/obs/observer
ExecStart=/home/obs/observer/.venv/bin/observer serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now observer
systemctl status observer
journalctl -u observer -f      # live logs
```

`observer serve` binds `0.0.0.0:8000` (from `config.py`). Open the dashboard at:

```
http://observer.local:8000
```

The live-camera tile in the sidebar points at `camera_stream_url`
(`http://172.16.50.10:8081/` by default). Override without code changes via env:

```ini
# add under [Service] in the unit file
Environment=OBSERVER_CAMERA_STREAM_URL=http://reconone.local:8081/
```

(Any `Settings` field is overridable as `OBSERVER_<FIELD>` — `OBSERVER_PORT`,
`OBSERVER_DETECT_IMGSZ`, etc.)

---

## 7. Hailo acceleration (HailoRT runtime)

Once the CPU path works end-to-end, switch the detector to the Hailo HEF for fast,
low-power inference. **Building the HEF happens on an x86 desktop** — that, plus the
distillation/training, is the full runbook in
[`hailo-deployment.md`](hailo-deployment.md). What runs *here* on the Pi is only the
**HailoRT runtime + PCIe driver**, installed below.

Hardware: the Hailo-8/8L on the Pi 5 connects over the PCIe/M.2 HAT (or AI HAT).
Make sure it's seated and the M.2 HAT is fitted before booting.

### a. Enable PCIe and install the stack

On Raspberry Pi OS Bookworm the whole runtime is packaged — driver, firmware,
HailoRT, and CLI in one:

```sh
sudo apt update
sudo apt install -y hailo-all
sudo reboot
```

`hailo-all` pulls in the `hailort` library, the `hailo_pci` kernel driver, the
device firmware, and `hailortcli`. If your firmware predates PCIe-HAT support,
enable it explicitly in `/boot/firmware/config.txt`:

```
dtparam=pciex1
# Gen 3 is faster but not officially guaranteed on every board:
# dtparam=pciex1_gen=3
```

then reboot.

### b. Verify the chip is alive

```sh
# kernel sees the PCIe device + driver bound
lspci | grep -i hailo
dmesg | grep -i hailo

# HailoRT can talk to the firmware (prints serial, fw version, arch)
hailortcli fw-control identify
```

If `fw-control identify` returns the device details, the runtime is good. If `lspci`
shows nothing, it's a seating/PCIe-enable problem, not software.

### c. Install the Python binding into the venv

The `hailort` Python wheel is provided by the apt package (system site). Expose it
to Observer's venv — simplest is to allow system packages, or install the matching
wheel:

```sh
. ~/observer/.venv/bin/activate
python -c "import hailo_platform; print(hailo_platform.__version__)"
# if ImportError, recreate the venv with: python3 -m venv --system-site-packages .venv
# (HailoRT's Python API is versioned to the apt runtime — don't pip a mismatched one)
```

### d. Point Observer at your HEF

Copy the `.hef` you compiled on the desktop into the repo, then select the Hailo
backend via env (no code change) on the systemd unit:

```ini
# add under [Service] in /etc/systemd/system/observer.service
Environment=OBSERVER_DETECTOR_BACKEND=hailo
Environment=OBSERVER_HAILO_HEF_PATH=/home/obs/observer/models/aircraft_yolov8n.hef
Environment=OBSERVER_HAILO_IMGSZ=640
```

```sh
sudo systemctl daemon-reload
sudo systemctl restart observer
journalctl -u observer -f      # confirm it loads the Hailo backend, not yoloworld
```

Process a known-aircraft clip and confirm the verdict matches the CPU run — that's
the parity check before trusting it. Detail and the compile steps:
[`hailo-deployment.md`](hailo-deployment.md).

---

## 8. Verify end-to-end

1. `systemctl status observer` → active, no restart loop.
2. Browse `http://observer.local:8000` → dashboard loads, live-cam tile shows
   `reconone`'s stream.
3. Drop a clip into `~/observer/data/incoming/` (or trigger motion on `reconone`)
   → it appears in the timeline, processes, and the verdict renders.
4. `journalctl -u observer -f` shows the receive → process → done sequence.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `observer.local` won't resolve | Use the IP from the router; ensure `avahi-daemon` is running |
| Service won't start | `journalctl -u observer -e`; usually venv path or missing weights |
| Dashboard up, no clips | Check `data/incoming/` is being filled; verify rsync/SSH-key path |
| Live-cam tile blank | `reconone` stream URL/port wrong, or camera Pi down — set `OBSERVER_CAMERA_STREAM_URL` |
| Processing very slow | Expected on CPU at imgsz=1280 — move to the Hailo backend |
