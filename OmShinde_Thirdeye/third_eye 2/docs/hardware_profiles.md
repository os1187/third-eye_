# Hardware Profiles and Config Presets

This guide lists recommended hardware profiles with expected end-to-end FPS ranges and config presets to quickly tune Third Eye for your machine.

## Mac (Apple Silicon)

- MacBook Air/Pro M1: 18–25 FPS (CoreML YOLO + Depth S at 384)
- MacBook Air/Pro M2: 22–30 FPS (CoreML YOLO + Depth S at 384)
- Mac Studio M2 Max/Ultra: 28–40 FPS (CoreML YOLO + Depth S at 448)

Notes:
- Prefer CoreML for YOLO on macOS for best throughput. Set `yolo.use_coreml = true` and provide `yolo.coreml_package_path`.
- DepthAnythingV2 Small at `resolution=384` is a good balance for real-time.
- If thermal throttling occurs, reduce `yolo.img_size` or increase skip settings.
- For maximum accuracy on capable hardware, set `THIRD_EYE_PROFILE=high-performance` to automatically use `yolov8s.pt` and `yolo.img_size=448`. Balanced profile keeps config defaults (now `img_size=384`).

Config preset (Mac M1/M2 balanced):

```
{
  "yolo": {
    "use_coreml": true,
    "coreml_package_path": "models/yolo11n.mlpackage",
    "img_size": 448,
    "conf": 0.4,
    "iou": 0.5
  },
  "depth": {
    "resolution": 384,
    "skip_frames": 2,
    "fast_mode": true,
    "quantize_int8": true
  },
  "performance": {
    "target_fps": 25,
    "min_fps": 20,
    "yolo_skip": 1,
    "depth_skip": 2,
    "frame_skip": 1
  }
}
```

Mac Studio M2 Max/Ultra performance preset:

```
{
  "yolo": {
    "use_coreml": true,
    "img_size": 512
  },
  "depth": {
    "resolution": 448,
    "skip_frames": 1
  },
  "performance": {
    "target_fps": 30,
    "min_fps": 24
  }
}
```

## NVIDIA GPU (CUDA)

- RTX 3060/3070 (Laptop/Desktop): 30–45 FPS (YOLOv8n PyTorch + Depth S 384)
- RTX 4070/4080: 45–60 FPS (YOLOv8n + Depth S 448)
- RTX 4090: 70–100+ FPS (YOLOv8n + Depth S 512, depending on thermal budget)

Notes:
- Use PyTorch CUDA for YOLO and Depth on Linux/Windows; set `yolo.use_coreml = false`.
- Increase `img_size` and `depth.resolution` progressively; watch min FPS via `/health`.
- If CPU becomes a bottleneck (audio or telemetry), raise skip settings slightly.

Config preset (RTX 3060/3070 balanced):

```
{
  "yolo": {
    "use_coreml": false,
    "img_size": 480
  },
  "depth": {
    "resolution": 384,
    "skip_frames": 1,
    "fast_mode": true
  },
  "performance": {
    "target_fps": 30,
    "min_fps": 22,
    "frame_skip": 1
  }
}
```

Config preset (RTX 4070/4080 performance):

```
{
  "yolo": {
    "use_coreml": false,
    "img_size": 512
  },
  "depth": {
    "resolution": 448,
    "skip_frames": 1
  },
  "performance": {
    "target_fps": 45,
    "min_fps": 30
  }
}
```

Config preset (RTX 4090 high-throughput):

```
{
  "yolo": {
    "use_coreml": false,
    "img_size": 640
  },
  "depth": {
    "resolution": 512,
    "skip_frames": 0
  },
  "performance": {
    "target_fps": 60,
    "min_fps": 40
  }
}
```

## Tuning Tips

- Use `/metrics` and `/health` to monitor FPS, stalls, and queue latencies.
- Maintain `min_fps` at or above your operational threshold; the system marks unhealthy if FPS drops below it.
- For crowded environments, keep `yolo.img_size` modest to avoid detector backlogs; depth can be slightly higher.
- If audio queue grows, raise `audio.enqueue_rate_limit_s` or reduce announcement verbosity.

## Readiness and Health

- Readiness: `GET /ready` returns 200 when core threads are alive and the runtime is marked ready.
- Health: `GET /health` returns 200 with `{"ok": true}` when running, not stalled, and above `performance.min_fps`.

## Logging and Rotation (brief)

- Configure rotation in `config.logging` to keep log sizes controlled. Supported keys:
  - `rotate_mode`: `size` or `time`
  - `rotate_max_bytes`: e.g., `10485760` (10 MiB)
  - `rotate_backup_count`: e.g., `5`
  - `rotate_when`: e.g., `"midnight"`
  - `rotate_interval`: integer multiplier for `when`

Adjust presets to your environment and verify with `/metrics` and `/health` before long runs.