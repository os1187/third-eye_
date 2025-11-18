# Telemetry Persistence and Redaction

This project can periodically snapshot runtime telemetry and persist it to a JSONL file. Telemetry can be enabled/disabled, include or exclude recent events, and redact sensitive fields before writing.

## Configuration Keys

All telemetry configuration lives under `logging` in your config file.

- `telemetry_enabled` (bool, default: `true`)
  - Gates telemetry across the app. When `false`, the telemetry persistence thread is not started and snapshots are skipped.
- `telemetry_interval_s` (number, default: `30.0`)
  - Interval in seconds between snapshots. Values are clamped to a reasonable range by the app.
- `telemetry_file` (string, optional)
  - Absolute path to the JSONL file to write snapshots to. If omitted, a default `telemetry.jsonl` is created in the runtime directory.
- `telemetry_include_recent_events` (bool, default: `true`)
  - Controls whether `events_recent` is included in metrics and snapshots. When `false`, recent events are omitted entirely.
- `telemetry_redact_fields` (array of strings, default: `[]`)
  - A list of event fields to remove from telemetry before persisting or returning via `/metrics`. Typical values might include `text`, `image`, `screenshot`, `audio`, or other sensitive keys your events carry.

## What Gets Persisted

Snapshots include high-level counters and samples of events. When `telemetry_include_recent_events` is `true`, `events_recent` is added to snapshots and returned by `/metrics`. If disabled, `events_recent` is omitted.

Before any telemetry data is written to disk (or returned via `/metrics`), fields listed in `telemetry_redact_fields` are removed from event objects. Redaction applies to:

- `events_recent` (when included)
- `events_sample` (if present)

Other non-event metadata may also be present in snapshots; redaction only removes fields from event objects.

## Example Configuration

```yaml
logging:
  telemetry_enabled: true
  telemetry_interval_s: 30
  telemetry_file: "/var/log/third_eye/telemetry.jsonl"
  telemetry_include_recent_events: true
  telemetry_redact_fields:
    - text
    - screenshot
    - image
```

## File Format

Telemetry is written as JSON Lines (one JSON object per line). Each line contains a snapshot of key metrics and event samples at the time of capture.

- Append-only writes
- Safe to tail with tools like `tail -f`
- Easy to ingest into log analysis systems

## Event Fields

Recent events (when included) carry a standard set of fields to aid analysis and tuning:

- `time`: timestamp in seconds
- `type`: announcement variant (e.g., `urgent_critical`, `warning_approach`, `nearby`)
- `label`: object class label
- `side`: `left` | `right` | `ahead`
- `depth_m`: raw measured depth in meters
- `smoothed_m`: smoothed depth used for gating decisions
- `speed_mps`: track-derived object speed estimate
- `eff_speed_mps`: effective speed used for gating (blends center looming)
- `importance`: final importance score used for prioritization
- `message`: spoken phrase

Note: When streaming external depth via `/depth/push`, freshness affects responsiveness. Configure `network_depth_freshness_s` in `config.json` (default `0.8`) to tune how long external frames are considered fresh.

## Endpoint Behavior

The `/metrics` endpoint respects the same settings:

- Honors `telemetry_include_recent_events` for including/excluding `events_recent`
- Applies `telemetry_redact_fields` to event objects before responding

## Operational Notes

- If `telemetry_enabled` is set to `false`, the persistence thread is not started and snapshots are skipped entirely.
- If an error occurs while writing the JSONL file, a warning is logged and the loop continues.
- Be mindful of redaction: removing large fields (e.g., images) can significantly reduce telemetry file size.