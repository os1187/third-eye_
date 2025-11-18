# Deployment Troubleshooting Guide

This guide lists common issues and fixes when deploying and running the Third Eye runtime on macOS or Linux.

## Quick Checklist
- Python venv active and dependencies installed (`pip install -r requirements.txt`)
- Models present or reachable (e.g., `yolov8n.pt`, depth model)
- Camera permissions granted (macOS) and device accessible (Linux: `/dev/video*`)
- Correct bind/port configuration for shortcuts server
- Auth token set when binding to non-local addresses

## Shortcuts Server Not Responding
- Symptom: `curl http://127.0.0.1:8765/metrics` times out.
- Checks:
  - Ensure runtime is started: `python run_depth.py`
  - Confirm `announcements.shortcuts_port` is open, no conflicts
  - Verify `announcements.shortcuts_bind` is reachable (default `127.0.0.1`)
  - If binding to `0.0.0.0` or public IP, set `THIRD_EYE_TOKEN` or `announcements.shortcuts_token` and include it in requests
- Use scripts:
  - `scripts/readiness.sh` for `/ready`
  - `scripts/healthcheck.sh` for `/health`
- Logs:
  - Check `logs/third_eye.log` and `logs/depth_system.log` for server startup errors

## Unauthorized (401) Errors
- Symptom: `/metrics` or `/dashboard` returns 401 when accessed remotely.
- Cause: Token protection is enabled for non-local binds.
- Fix:
  - Set token via env: `export THIRD_EYE_TOKEN=...`
  - Or set in `config.json`: `announcements.shortcuts_token`
  - Include token using headers or query:
    - `Authorization: Bearer <token>`
    - `Authorization: Token <token>`
    - `Authorization: Basic <base64(user:token)>`
    - `X-ThirdEye-Token: <token>`
    - `?token=<token>`

## Camera Not Detected / Permission Issues
- macOS:
  - Ensure you accepted the camera permission prompt.
  - Check System Settings → Privacy & Security → Camera.
- Linux:
  - Verify device: `ls /dev/video*`
  - Ensure your user has access; consider `udev` rules.
  - Check `camera.camera_id` in `config.json` (e.g., 0, 1).

## Low FPS / Performance Degradation
- Symptoms: `/health` shows `ok=false` or observed FPS below `performance.min_fps`.
- Fixes:
  - Reduce YOLO image size: `yolo.img_size`
  - Lower `yolo.max_det`; adjust `yolo.conf`, `yolo.iou`
  - Increase skips: `performance.frame_skip`, `depth_skip`, `yolo_skip`
  - Reduce depth `resolution` and set `depth.min_interval`
  - Ensure CPU/GPU resources are available; close other heavy apps
  - See `docs/hardware_profiles.md` for tuned presets

## Audio Queue Overflows / Speech Too Chatty
- Symptoms: Delayed speech or repeated messages.
- Fixes:
  - Increase `audio.enqueue_rate_limit_s` and lower verbosity
  - Enable earcons with lower speech volume for less chatter
  - Use `/audio?action=quieter` or `set_volume` to tame output
  - Leverage `audio.dedup_window_s` to suppress repeat messages

## Token Set but Still Unauthorized
- Double-check header format and whitespace.
- Try `X-ThirdEye-Token: <token>` or `?token=<token>`.
- Ensure server is bound to non-local and token requirement is active; restart runtime after changing token.

## Port Conflicts
- Symptom: Shortcut server fails to start; no `/metrics`.
- Fixes:
  - Change `announcements.shortcuts_port` in `config.json`
  - Check for other services using the port

## Depth or YOLO Threads Stalling
- Symptom: `/health` reports stall flags.
- Fixes:
  - Inspect logs in `logs/depth_system.log` and `logs/third_eye.log`
  - Verify model paths and any GPU acceleration setup
  - Ensure watchdog is running; it will attempt restarts

## Background Process Issues (Linux)
- Symptom: Background process not starting or immediately stops.
- Fixes:
  - Confirm your process runner (e.g., Docker, `tmux`, `pm2`, system service) uses the correct venv and working directory
  - Ensure environment variables (e.g., `THIRD_EYE_TOKEN`) are provided
  - Review logs from your runner (e.g., `docker logs`, `tmux capture-pane`, `pm2 logs`)
  - Add exponential backoff and retries in your runner to avoid tight restart loops

## Readiness vs Health Failures
- `GET /ready` returns non-200:
  - Core threads may not have initialized; check logs and give it a few seconds.
- `GET /health` shows `ok=false`:
  - Either stalls or FPS below minimum; tune performance per above.

## Debugging Tips
- Enable verbose metrics: `GET /metrics?verbose=1`
- Snapshot telemetry in logs; review recent `events_recent` and `per_class` counters.
- Temporarily disable announcements: `/toggle?feature=announce&state=off` while triaging.

## When All Else Fails
- Reset config to defaults and re-run.
- Recreate venv and reinstall dependencies.
- Test with `--headless` to rule out UI issues.
- Try a different camera or lower resolutions.
- Open an issue with log snippets and `/metrics` JSON.