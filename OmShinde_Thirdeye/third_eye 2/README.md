# Third Eye

Real-time obstacle detection and depth guidance for navigation on macOS. Combines YOLO object detection with optional DepthAnythingV2 depth to provide situational awareness, lane guidance, and hazard monitoring. Includes a watchdog to keep critical threads healthy and telemetry for performance insights.

## Overview

- Detection: Ultralytics YOLO (Core ML optional on macOS) with ROI-focused inference.
- Depth: DepthAnythingV2 (local/Hugging Face).
- Watchdog: Monitors YOLO, depth, camera, audio, and restarts stalled threads.
- Telemetry: FPS, inference times, stalls, and runtime stats.
  
This repository is streamlined for macOS Continuity Camera (or any local UVC webcam). iOS/Xcode app and network LiDAR streaming have been removed.

## Requirements

- Python 3.10+ (tested with 3.13) and `pip`.
- macOS with camera access (Continuity Camera preferred if available).

Install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# On Apple Silicon M4 or optimized configs, you may use:
pip install -r requirements_m4.txt
```

Download/prepare models (optional; weights may already be present):

```bash
python download_models.py
```

## Quick Start

Run locally on macOS:

```bash
python main.py
```

Health checks and supervision scripts are in `scripts/`.

## Configuration

Key configs are in `config.json` (macOS). Important sections:

- `yolo`:
  - `weights`: Path to YOLO weights (e.g., `yolov8n.pt`).
  - `img_size`: Input resolution for YOLO.
  - `conf`, `iou`: Confidence and IOU thresholds.
  - `custom_classes`: Optional list of class names to restrict detections.
  - `use_coreml`: Use Core ML model if available on macOS.

- `depth`:
  - `model`: DepthAnything model variant (e.g., `depth_anything_v2_vits`).
  - `resolution`: Base width for depth inference (dynamic during hazards).
  - `alpha`: Temporal smoothing factor.
  - `source`: Set to `local` (external network depth removed).

- `performance`:
  - `frame_skip`, `yolo_skip`, `depth_skip`: Skip counters for throughput.
  - Additional parameters control preprocessing and queue behavior.

- `hazard`:
  - `yolo_interval_s`, `yolo_interval_critical_s`: Hazard-aware cadence.
  - `depth_min_interval_s`: Minimum depth cadence during hazard boost.
  - ROI shifting options for lane-centric detection.

- `watchdog`:
  - `yolo_stall_s`, `depth_stall_s`, `camera_stall_s`: Stall thresholds.
  - `min_restart_interval_s`: Prevents rapid restarts.

`network_depth_*` settings are deprecated in continuity-only mode.

See `docs/api.md` and `docs/user-guide.md` for detailed endpoints and usage.

## Features

- ROI-first detection with adaptive shift toward nearest hazard lane.
- External depth via `/depth/push` (base64 PNG) with freshness management.

### Depth Calibration (Grayscale → Meters)

- If your neural depth model produces 0–255 grayscale values, you can apply a manual linear calibration to convert pixel values to meters.
- Set `depth.calibration_factor_m_per_255` in `config.json` to the number of meters per 255 units. Mapping is: `meters = pixel_value * calibration_factor_m_per_255`.
- Example: if a wall at ~3 m appears near value ~150, then `calibration_factor_m_per_255 ≈ 3 / 150 ≈ 0.02`. Tune with a few known test distances.
- When this factor is greater than `0`, the runtime uses raw grayscale values (with smoothing) for meter conversion, bypassing relative per-frame normalization.
- Depth postprocessing is robust against NaNs/Infs and shape errors.
- Performance metrics:
  - `yolo_inference_times`, `depth_inference_times`, `processing_times`.
  - `camera_capture_times`, `camera_read_failures`, `camera_skips`, `depth_skips`, `yolo_skips`.

## Health & Watchdog

- Health endpoint and scripts:
  - `scripts/healthcheck.sh` and `/health` expose stall flags and FPS.
  - Watchdog restarts YOLO/depth threads if stalled beyond configured windows.
  - Timestamps tracked: `last_yolo_time`, `_last_depth_time`, `_last_frame_time`.

Logs:

- Runtime logs in `logs/` and `archive_unused/logs/` include telemetry and stall events.

## Hardware Profiles

- Hardware profiles and performance guidance: `docs/hardware_profiles.md`.

## Troubleshooting

- Camera not opening:
  - The system tries AVFoundation and default backends; enable `mock_sensor` to simulate frames.
  - Monitor `camera_read_failures` in performance stats.

- YOLO/Depth stalls:
  - Check `/health` for `yolo`, `depth`, and `camera` stall flags.
  - Watchdog restarts threads; adjust `watchdog` thresholds if too aggressive.

- Slow FPS:
  - Increase `performance.frame_skip`, reduce `yolo.img_size`, tune `depth.resolution`.
  - See `docs/accuracy_improvements.md` for trade-offs and tuning tips.

## Contributing

- Keep changes consistent with current style.
- Update docs and comments when modifying core processing paths.

## License

Project files are provided without an explicit license. Please consult the repository owner for licensing details.