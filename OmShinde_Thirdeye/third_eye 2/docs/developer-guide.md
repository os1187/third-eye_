# Developer Guide

This document helps new contributors understand the architecture, core threads, error handling, configuration, and testing strategy for Third Eye.

## Architecture Overview
- Core modules: `main.py` (runtime), `depth_anything_v2/` (depth model), `providers.py` (optional integrations).
- Threads:
  - `yolo_thread` — detection loop (Ultralytics/Core ML, ROI-focused, optional ensemble).
  - `depth_thread` — depth acquisition/inference/postprocessing (smoothing, center ROI).
  - `announcement_thread` — transforms detections/context into phrased messages.
  - `audio_worker` — consumes `audio_queue`, handles segmentation and TTS.
  - `shortcuts_server_thread` — lightweight HTTP control surface.
  - `watchdog_thread` — monitors liveness and restarts stalled threads.

Shared runtime state lives in the `State` class (module-level `state`) and is protected by `state.lock` for multi-field updates.

## Error Handling Helpers
- `log_error(tag, exc_or_msg, level='warning', rate_limit_s=1.5)`
  - Increments `state.error_counts[tag]` and logs with per-tag rate limiting.
- `safe_execute(tag, func, fallback=None, ...)`
  - Wraps risky calls, logs on error, and returns `fallback` to keep the pipeline stable.
  - Used in YOLO inference (Core ML/Ultralytics), depth preprocessing/inference/postprocessing, and camera read paths.

## Configuration
- Main configs: `config.json`, `config_pi.json`, values read in `main.py` at startup.
- Important sections:
  - `yolo`: weights, image size, thresholds, `use_coreml`.
  - `depth`: model variant (`DEPTH_MODEL`), resolution, alpha smoothing.
  - `performance`: frame and model skip cadences, target FPS, queue behavior.
  - `hazard`: intervals for increased cadence, ROI shifting.
  - `watchdog`: stall thresholds and restart rate limits.
  - `announcements`/`audio`: phrasing, rate limits, earcons, smart verbosity.

## Thread Communication & Watchdog
- Liveness timestamps maintained in `state` (e.g., `last_yolo_time`, `_last_depth_time`).
- `watchdog_thread` compares timestamps against thresholds and replaces threads.
- Restarting uses `state.threads[...]` and small stubs for targets where needed.

## Spatial Audio & Announcements
- `speak_status(message)` applies:
  - Tone formatting, dedup windows, smart verbosity under load.
  - Segmentation for alert prefixes (e.g., "Warning!") with fast prefix rate and normal tail.
  - Priority queue and optional earcons.
- `should_announce(detection, speed, context)` evaluates proximity, type, suppression windows.

## Testing
- Run: `pytest -q` (discoveries restricted to `tests/`).
- Notable tests:
  - Detection fusion: `test_detection_output.py` for `combine_detections`.
  - Spatial audio: `test_spatial_audio.py` for segmentation, rate limits, overflow.
  - Watchdog: `test_watchdog_thread_comm.py` for stall-triggered restarts.

## Code Style & Docstrings
- Prefer concise Google-style docstrings for functions and classes.
- Inline comments only where logic is non-obvious or error-prone.
- Avoid broad refactors unless necessary; keep changes minimal and targeted.

## Contribution Workflow
- Create a virtual environment and install dependencies (`requirements.txt`).
- Make changes with focused scope; update or add tests near the change.
- Validate with `pytest -q` and update docs (`README.md`, `docs/user-guide.md`, `docs/developer-guide.md`) as appropriate.
- Pull requests should include:
  - Rationale and scope.
  - Tests and doc updates.
  - Notes on configuration changes.

## Observability
- Logs: `logs/third_eye.log`, `logs/depth_system.log`.
- Telemetry: `state.telemetry` and persisted files noted in `docs/telemetry.md`.
- Performance stats in `state.performance_stats` track FPS, inference times, and failure counters.