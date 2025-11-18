# Third Eye User Guide

This guide walks you through setup, running, controls, and monitoring for the Third Eye runtime.

## Prerequisites
- macOS or Linux with Python 3.9+
- Recommended: a USB/UVC camera or integrated webcam
- Install dependencies:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- Optional platform-specific extras: `requirements_m4.txt` for Apple Silicon environments.

## Models
- Default YOLO weights: `yolov8n.pt` (present in repo root).
- Depth model defaults to `Depth-Anything-V2-Small-hf`.
- To (re)download models, check `download_models.py` and `models/`.

## Configuration
- Runtime uses `config.json`. Key sections:
  - `camera`: resolution, FPS, camera ID
  - `yolo`: weights, `img_size`, `max_det`, thresholds
  - `depth`: model, resolution, interval, skip frames
  - `performance`: target fps, adaptive quality, skip limits
  - `audio`: speech volume, earcons, queue limits
  - `announcements`: toggles and heuristics
    - `realtime_mode`: when `true`, prioritizes immediate hazard prompts and reduces cooldowns (scenario-driven, not matrix-based)
    - `instant_hazard`: when `true`, speaks critical obstacles instantly, bypassing cooldown
    - `smart_verbosity`: when `true`, suppresses non-critical chatter under load (low FPS or high audio queue latency)
  - `audio`: queue and speech behavior
    - `urgent_preempt`: when `true` on macOS, urgent messages preempt ongoing speech via AVFoundation
    - `enqueue_rate_limit_s`: rate-limit between queued messages (urgent/warnings relax this)
    - `dedup_window_s`: suppress identical messages spoken within this window
    - `smart_fps_threshold`: FPS threshold below which non-critical messages are suppressed
    - `smart_queue_latency_s`: average queue latency threshold (seconds) to suppress non-critical messages
    - `ttc_urgent_s` / `ttc_warning_s` / `ttc_caution_s`: TTC (time-to-collision) thresholds in seconds; tighter TTC escalates or permits approach warnings, long TTC suppresses slow approaches
    - `summary_min_items` / `summary_group_window_s`: enable grouped “Nearby” summaries when multiple low-importance objects are present; reduces chatter by batching
  - `announcements`: TTC and summaries
    - `smart_ttc`: when `true`, uses TTC to gate approach variants (e.g., suppress very long TTC “slow approach”)
    - `summary_interval_s`: minimum interval between grouped summary announcements
  - `network_depth_freshness_s`: max age (seconds) for external LiDAR depth frames
  - `network_depth_freshness_s_per_mode`: per-travel-mode freshness overrides, e.g. `{ "WALKING": 0.8, "CYCLING": 1.0, "DRIVING": 1.2 }`
  - `speed_detection.center_corridor_blend_in` / `speed_detection.center_corridor_blend_out`: center-corridor speed blending weights used in `eff_speed` composition
  - `announcements.crowd_slow_person_importance_mult`: raises importance threshold for slow persons when `scene_tags` includes `crowd`
  - `watchdog`: stall thresholds and logging
  - `announcements.shortcuts_bind` / `announcements.shortcuts_port`: HTTP server bind/port
  - `announcements.shortcuts_token` or `THIRD_EYE_TOKEN`: set access token if binding to non-local addresses
  - `haptics`: subtle heartbeat cues when audio is suppressed
    - `heartbeat_enabled`: toggle periodic haptic pulses for persistent hazards
    - `heartbeat_interval_s`: global minimum interval between heartbeats
    - `per_track_interval_s`: per-hazard minimum interval to avoid repetition
    - `heartbeat_level`: semantic intensity hint (`nearby`, `warning`, `critical`)

## Running
- Start the runtime:
```
python main.py
```
- Useful flags:
  - `--headless` → no UI window (for servers or headless use)
  - `--no-announce` → disable TTS announcements
  - `--no-haptics` → disable haptic cues (platform dependent)

- On first run, macOS may prompt for camera access; allow it.

## Monitoring
- Open the HTML dashboard:
  - `http://127.0.0.1:8765/dashboard`
- Machine-readable endpoints:
  - `GET /metrics` → telemetry and performance
  - `GET /health` → stall flags and thread liveness
  - `GET /ready` → readiness probe
- Sample scripts:
  - `scripts/readiness.sh`
  - `scripts/healthcheck.sh`

## Controls (HTTP API)
- Toggle features:
```
curl "http://127.0.0.1:8765/toggle?feature=announce&state=off"
curl "http://127.0.0.1:8765/toggle?feature=realtime&state=on"
```
- Set travel mode:
```
curl "http://127.0.0.1:8765/mode?travel=WALKING"
```
- Audio controls:
```
curl "http://127.0.0.1:8765/audio?action=quieter"
curl "http://127.0.0.1:8765/audio?action=set_volume&value=0.7"
curl "http://127.0.0.1:8765/audio?action=silence&secs=30"
```
- Announce a message:
```
curl "http://127.0.0.1:8765/announce?msg=Crosswalk ahead"
```
- Haptics and heartbeat:
```
# Enable/disable heartbeat haptics at runtime
curl "http://127.0.0.1:8765/toggle?feature=heartbeat&state=on"
# Adjust intervals (persist via config.json for long-term changes)
curl "http://127.0.0.1:8765/param?key=haptics.heartbeat_interval_s&value=3.0"
```
- Voice commands (HTTP feed):
```
# Enable voice mode (starts listener and optional mic STT if available)
curl "http://127.0.0.1:8765/toggle?feature=voice&state=on"
# Send commands via /voice (useful when no STT backend is installed)
curl "http://127.0.0.1:8765/voice?cmd=navigate to Central Park"
curl "http://127.0.0.1:8765/voice?cmd=use apple maps"
curl "http://127.0.0.1:8765/voice?cmd=walking mode"
```
Notes:
- Supported phrases include: `navigate to ...`, `search ...`, `use google maps`, `use apple maps`, `walking mode`, `driving mode`, `quieter`, `louder`, `mute`, `focus center`, `crowd mode on|off`.
- Maps provider and travel mode are reflected in `/metrics` and `/dashboard`.
- Optional microphone STT is available when `speech_recognition` and a compatible microphone are installed.

If bound to a non-local address, add an auth token, e.g.:
```
curl -H "Authorization: Bearer $THIRD_EYE_TOKEN" "http://<host>:8765/health"
```

## Logs and Telemetry
- Primary log: `logs/third_eye.log`
- Watchdog/system log: `logs/depth_system.log`
- Telemetry persistence: enabled by default; see `docs/telemetry.md` for details.

## Performance Tips
- See `docs/hardware_profiles.md` for tuned profiles and recommendations.
- Adjust:
  - `performance.target_fps`, `min_fps`, and skip parameters (`frame_skip`, `depth_skip`, `yolo_skip`)
  - YOLO `img_size` and `max_det` to balance accuracy vs speed
  - Depth `resolution` and `min_interval`
  - Lane guidance stability: `blind_navigation.path_smoothing_factor` and
    `lane_guidance.steering_guidance` thresholds for angle classification

## Lane Guidance Usage
- Enable lane guidance in `config.json` (`lane_guidance.audio_guidance_enabled: true`).
- The system computes lane candidates from depth, merges them, and selects a path.
- Guidance messages are rate limited; urgent cases use shorter intervals.
- Confidence reflects path quality and width; narrow passages reduce confidence.
- Check `/metrics` for `lane_guidance_provided` events and steer angle values.

## Heartbeat Haptics Usage
- Heartbeat pulses occur when hazards persist but repeated voice alerts are suppressed.
- Configure cadence via `haptics.heartbeat_interval_s` and `haptics.per_track_interval_s`.
- Set `haptics.heartbeat_enabled` to control behavior; platform must support haptics.
- Telemetry includes `heartbeat` events with `track_id` and `depth_m`.

## Service Deployment (Linux)
- Use container-based or process managers of your choice (e.g., Docker, `tmux`, or `pm2`).
- Typical steps:
  - Create a dedicated user, install Python venv, copy repo, or build a container
  - Point your process runner to the venv and `scripts/launch.sh` (or container entrypoint)
  - Configure environment variables (e.g., `THIRD_EYE_TOKEN`) in your runner
  - Monitor `/health` and `/ready` endpoints for basic liveness/readiness

## Safety and Etiquette
- This system assists but does not replace attention or judgment.
- Use in controlled environments first. Validate obstacle announcements and latency.
- Monitor `/health` and `/metrics` during longer sessions.

## Uninstall / Clean Up
- Stop any running services and processes.
- Remove the venv and repo directory.
- Clean `logs/` and any persisted telemetry if desired.
