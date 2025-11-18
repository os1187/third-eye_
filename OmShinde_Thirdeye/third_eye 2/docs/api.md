# Third Eye Runtime API

This document describes the HTTP API exposed by the Third Eye runtime, intended for local control, monitoring, and integration with automations (e.g., Apple Shortcuts).

## Base URL and Binding

- Default base URL: `http://127.0.0.1:8765`
- Configure bind/port via `config.json`:
  - `announcements.shortcuts_bind` (default `127.0.0.1`)
  - `announcements.shortcuts_port` (default `8765`)

When binding to a non-local address, the server requires an access token for protection.

## Authentication

- Token source:
  - Env var `THIRD_EYE_TOKEN`, or
  - `announcements.shortcuts_token` in `config.json`
- Pass the token using:
  - `Authorization: Bearer <token>` (preferred; required when production gate is enabled)
  - In non-production, you may also use: `Authorization: Token <token>`, `Authorization: Basic <base64(user:token)>`, `X-ThirdEye-Token: <token>`, or `?token=<token>` (discouraged).

If bound to `127.0.0.1/localhost`, auth is not required by default.

- JWT (short‑lived) support:
  - When `THIRD_EYE_TOKEN_SECRET` (and optionally `THIRD_EYE_TOKEN_SECRET_NEXT`) are set, Bearer tokens can be JWTs signed with HS256.
  - Tokens must include an `exp` claim; expired tokens are rejected.
  - Rotation: during secret rotation, tokens signed by either current or next secret are accepted.

## Endpoints

### Readiness
- `GET /ready`
  - Returns `200` when the runtime is ready and core threads are alive.
  - Body may include `{"ready": true}`.

### Health
- `GET /health`
  - Returns JSON with system liveness and stall flags.
  - Shape includes:
    - `ok` (bool) overall health
    - `running` (bool)
    - `threads` (object: `camera`, `yolo`, `depth`, `watchdog`, `shortcuts` → bool)
    - `stall` (object: `yolo`, `depth`, `camera` → bool)
    - `last_fps` (number), `min_fps` (number)

Example:
```
GET /health → 200
{
  "ok": true,
  "running": true,
  "threads": {"camera": true, "yolo": true, "depth": true, "watchdog": true, "shortcuts": true},
  "stall": {"yolo": false, "depth": false, "camera": false},
  "last_fps": 24.7,
  "min_fps": 20.0
}
```

### Metrics
- `GET /metrics`
  - Returns JSON telemetry and performance metrics.
  - Common keys: `uptime_s`, `fps`, `quality`, `cpu_avg`, `mem_avg`, `announce_total`, `critical_total`, `summary_total`, `audio_queue_size`, `tts_queue_avg_ms_10`, `tts_speak_avg_ms_10`, `speech_volume`, `earcons_enabled`, `events_recent`, `per_class`, `tts_provider`, `phrasing_enabled`, `phrasing_provider`, `tts_cb`, `phr_cb`.
    - `tts_cb` / `phr_cb` objects include: `open` (bool), `fail_count` (int), `threshold` (int), `cooldown_s` (float), `cooldown_remaining_s` (float).
  - Add `?verbose=1` to include `events_sample`.

### Dashboard
- `GET /dashboard`
  - Returns an HTML dashboard view for quick manual monitoring.

### Feature Toggle
- `GET /toggle?feature=<feature>&state=<on|off>`
  - `feature`: `announce` | `voice` | `ai` | `context` | `realtime`
  - `state`: `on` | `off` (also accepts `true|false|1|0`)
  - Response: plain text confirmation.

Additional toggles:
- `GET /toggle?feature=tts_provider&value=<local|elevenlabs>`
  - Switches the Text-to-Speech backend. Persists in `logs/prefs.json`.
- `GET /toggle?feature=phrasing&state=<on|off>[&provider=<builtin|openai>]`
  - Enables/disables AI phrasing rewrite for non-urgent messages. Optional provider selection. Persists in `logs/prefs.json`.

Examples:
- `GET /toggle?feature=announce&state=off`
- `GET /toggle?feature=ai&state=on`
 - `GET /toggle?feature=realtime&state=on`
 - `GET /toggle?feature=tts_provider&value=elevenlabs`
 - `GET /toggle?feature=tts_provider&value=local`
 - `GET /toggle?feature=phrasing&state=on&provider=openai`
 - `GET /toggle?feature=phrasing&state=off`

### Travel Mode
- `GET /mode?travel=<WALKING|DRIVING>`
  - Sets travel mode, which may influence heuristics and announcements.

### Audio Controls
- `GET /audio?action=<action>[&value=<0..1>][&secs=<0..300>]`
  - Actions:
    - `quieter` → decrease speech volume by 0.1 (min 0.1)
    - `louder` → increase speech volume by 0.1 (max 1.0)
    - `mute` → set speech volume to 0.0
    - `unmute` → set speech volume to 0.6 if currently 0
    - `set_volume&value=<float>` → set speech volume `0.0..1.0`
    - `set_earcon_volume&value=<float>` → set earcon volume `0.0..1.0`
    - `enable_earcons` / `disable_earcons` → toggle earcon playback
    - `silence&secs=<0..300>` → globally silence hints for `secs`
    - `unsilence` → remove global silence
  - Responses: JSON echo of changed values, e.g. `{ "speech_volume": 0.8 }`, `{ "earcons_enabled": true }`, `{ "global_silence_until": <timestamp> }`.

### Announce (Speak)
- `GET /announce?msg=<text>`
  - Speak a short message immediately (rate-limited by audio queue settings).
  - Response: `spoken` or error if `msg` missing.

### Voice Command
- `GET /voice?cmd=<text>`
  - Enqueue a voice command into the runtime (used by external STT or automation).
  - Examples:
    - `navigate to <destination>` → opens Maps for navigation using current provider and travel mode
    - `search <query>` → opens Maps search for `<query>`
    - `use google maps` / `use apple maps` → switches provider
    - `walking mode` / `driving mode` → sets travel mode
    - `quieter` / `louder` / `mute` → adjust audio
    - `focus center` / `crowd mode on|off` → behavior toggles
  - Response: `queued` or `missing cmd`/`command queue full`.

### IMU Update
- `GET /imu?ax=<>&ay=<>&az=<>&gx=<>&gy=<>&gz=<>`
  - Update IMU values: `user_acceleration` (`ax, ay, az`) and `rotation_rate` (`gx, gy, gz`).
  - Response: `imu updated`.

### Push Depth (LiDAR)
- `POST /depth/push`
  - Auth required when not bound to localhost.
  - Body: JSON
    - `depth_png_b64` (string, required): base64 PNG of single-channel depth.
      - Accepts `uint16` (millimeters or meters) or `float32` PNG.
    - `units` (string, optional): `meters` or `millimeters`.
    - `rgb_jpeg_b64` or `rgb_png_b64` (string, optional): base64 RGB for alignment.
  - Behavior:
    - Stores external LiDAR depth and optional RGB in runtime state.
    - When `depth.source` is `network`, the runtime uses pushed depth for processing.
    - Depth is converted to meters using `depth.network_units` and `depth.network_scale`.
  - Response: `ok`.

Example:
```
curl -X POST \
  -H "Authorization: Bearer $THIRD_EYE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "depth_png_b64": "<base64>",
    "units": "millimeters"
  }' \
  http://127.0.0.1:8765/depth/push
```

### Push RGB Frame
- `POST /frame/push`
  - Auth required when not bound to localhost.
  - Body: JSON
    - `rgb_jpeg_b64` or `rgb_png_b64` (string, required): base64-encoded RGB image.
  - Response: `ok`.

## Configuration for Network Depth
- In `config.json` under `depth`:
  - `source`: `model` (default) or `network`.
  - `network_units`: `meters` or `millimeters` (default `meters`).
  - `network_scale`: multiplier applied when converting to meters (default `1.0`).

When `source` is `network`, local depth models are not loaded and the runtime expects external depth via `POST /depth/push`.

## Rate Limits
- Configurable via environment variables (per 10-second window):
  - `THIRD_EYE_RATE_LIMIT_PUSH_DEPTH` (default `30`)
  - `THIRD_EYE_RATE_LIMIT_PUSH_FRAME` (default `60`)

### Permission Check
- `GET /permissions`
  - Trigger an immediate permission/liveness check (e.g., camera availability).
  - Response: `permissions ok=<true|false>`.

## Examples

- Readiness with token:
```
curl -H "Authorization: Bearer $THIRD_EYE_TOKEN" http://127.0.0.1:8765/ready
```

- Health JSON:
```
curl http://127.0.0.1:8765/health | jq
```

- Quieter and global silence for 30s:
```
curl "http://127.0.0.1:8765/audio?action=quieter"
curl "http://127.0.0.1:8765/audio?action=silence&secs=30"
```

- Toggle announcements off:
```
curl "http://127.0.0.1:8765/toggle?feature=announce&state=off"
```

- Switch mode to driving:
```
curl "http://127.0.0.1:8765/mode?travel=DRIVING"
```

## Notes
- Some endpoints may be rate-limited internally to preserve UX.
- Metrics may redact recent events; see `config.logging.telemetry_redact_fields`.
- For deployment probes, see `scripts/readiness.sh` and `scripts/healthcheck.sh`.
- Repeated rate-limit breaches trigger short backoff penalties per IP/path.

## Security
- HTTPS and HSTS:
  - When behind an HTTPS proxy that sets `X-Forwarded-Proto: https`, or when `THIRD_EYE_SHORTCUTS_HTTPS=1` is set, responses include `Strict-Transport-Security: max-age=31536000; includeSubDomains`.
- Auth requirements:
  - When bound to a non-local address or production gate is enabled, all non-public endpoints require authentication. Public endpoints are `GET /ready` and `GET /health`.
- Auth scheme hints:
  - Unauthorized responses include `WWW-Authenticate: Bearer` to guide clients toward Bearer tokens.
- CORS:
  - If enabled, CORS allowlist is enforced and `Access-Control-*` headers are set accordingly.