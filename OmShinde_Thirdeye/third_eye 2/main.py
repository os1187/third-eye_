import cv2
import time
import threading
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ops

# Helper: consistent YOLO preprocessing (BGR→RGB + letterbox to square)
def preprocess_frame_for_yolo(frame, target_size=640):
    try:
        if frame is None:
            return None
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        new_shape = (int(target_size), int(target_size))
        r = min(new_shape[0] / float(h), new_shape[1] / float(w))
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2.0
        dh /= 2.0
        # Resize
        if new_unpad[0] > 0 and new_unpad[1] > 0:
            img_resized = cv2.resize(img, (new_unpad[0], new_unpad[1]), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
        # Pad to square
        top = int(round(dh - 0.1)); bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1)); right = int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img_padded
    except Exception:
        return frame

def letterbox_with_meta(frame, target_size=640):
    """Return RGB letterboxed image plus (ratio, pad) for precise box remapping."""
    try:
        if frame is None:
            return None, 1.0, (0.0, 0.0)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        new_shape = (int(target_size), int(target_size))
        r = min(new_shape[0] / float(h), new_shape[1] / float(w))
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2.0
        dh /= 2.0
        if new_unpad[0] > 0 and new_unpad[1] > 0:
            img_resized = cv2.resize(img, (new_unpad[0], new_unpad[1]), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
        top = int(round(dh - 0.1)); bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1)); right = int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img_padded, float(r), (float(left), float(top))
    except Exception:
        return preprocess_frame_for_yolo(frame, target_size=target_size), 1.0, (0.0, 0.0)
from transformers import AutoImageProcessor, pipeline
try:
    # Optional: local DepthAnythingV2 (for fine-tuned weights)
    from depth_anything_v2.dpt import DepthAnythingV2
except Exception:
    DepthAnythingV2 = None
from PIL import Image
import subprocess
import queue
import asyncio
import logging
import json
import os
import argparse
from datetime import datetime
import signal
import sys
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import psutil
import socket
import struct
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from collections import deque
import base64
import math
import random
import contextlib

# Enable PyTorch MPS fallback on macOS to improve stability/perf
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
if True:
 
 # --- CPU Threading and Defaults ---
 try:
     _CPU_COUNT = os.cpu_count() or 4
 except Exception:
     _CPU_COUNT = 4
 # Hint native backends to use available cores when env not set
 os.environ.setdefault('OMP_NUM_THREADS', str(_CPU_COUNT))
 os.environ.setdefault('MKL_NUM_THREADS', str(_CPU_COUNT))
 try:
     torch.set_num_threads(_CPU_COUNT)
     # Interop threads control intra-op parallelism coordination; conservative half
     torch.set_num_interop_threads(max(1, _CPU_COUNT // 2))
 except Exception:
     pass
 try:
     # Let OpenCV parallelize transform-heavy stages when possible
     cv2.setNumThreads(_CPU_COUNT)
 except Exception:
     pass

 # Compute dynamic defaults based on CPU cores
 DEFAULT_MAX_WORKERS = max(4, _CPU_COUNT - 1)
DEFAULT_PREPROC_BUF = min(32, max(6, _CPU_COUNT))



# --- Non-blocking Speech Worker (macOS-safe fallback) ---
# Purpose: Avoid blocking inference by moving TTS to a background queue.
# Usage: call `speak_async(text)` from detection or hot paths.
_speech_queue = None
_speech_thread = None
_speech_lock = threading.Lock()

def _ensure_speech_queue():
    """Initialize the speech queue lazily with a reasonable max size."""
    global _speech_queue
    if _speech_queue is not None:
        return
    size = 64
    try:
        if 'config' in globals() and isinstance(config, dict):
            size = int(config.get('audio', {}).get('max_queue_size', 64))
    except Exception:
        size = 64
    try:
        _speech_queue = queue.Queue(maxsize=max(8, min(size, 512)))
    except Exception:
        _speech_queue = queue.Queue(maxsize=64)

def _mac_say_fallback(text: str, voice: str = None, rate: float = None):
    """Best-effort macOS 'say' fallback without blocking callers."""
    try:
        # Use runtime helper if available
        if 'mac_say' in globals() and callable(mac_say):
            mac_say(text, voice=voice, rate=rate, block=True)
            return
    except Exception:
        pass
    # Minimal direct fallback
    try:
        args = ['say']
        if isinstance(voice, str) and voice.strip():
            args += ['-v', voice]
        if isinstance(rate, (int, float)):
            # Map to a practical WPM range if possible; otherwise let 'say' default
            wpm = max(80, min(400, int(rate * 200)))
            args += ['-r', str(wpm)]
        args += [str(text)]
        subprocess.run(args)
    except Exception:
        try:
            subprocess.run(['say', str(text)])
        except Exception:
            # Last resort: log the failure to avoid silent drops
            try:
                logging.error(f"Speech fallback failed: {text}")
            except Exception:
                pass

def _speech_worker():
    """Background worker that serializes TTS, shielding inference from blocking."""
    while True:
        try:
            # Stop if application requests shutdown and queue is drained
            try:
                running = bool(getattr(state, 'running', True))
            except Exception:
                running = True
            if (not running) and (_speech_queue is not None) and _speech_queue.empty():
                break
            item = _speech_queue.get(timeout=0.5)
        except Exception:
            time.sleep(0.05)
            continue
        try:
            msg = str(item.get('text', '') or '')
            if not msg:
                continue
            # Prefer runtime TTS when available; otherwise macOS 'say'
            try:
                if 'tts_speak' in globals() and callable(tts_speak):
                    tts_speak(msg, block=True)
                else:
                    _mac_say_fallback(msg, voice=item.get('voice'), rate=item.get('rate'))
            except Exception:
                _mac_say_fallback(msg, voice=item.get('voice'), rate=item.get('rate'))
        finally:
            try:
                _speech_queue.task_done()
            except Exception:
                pass

def start_speech_worker():
    """Start the worker thread if not already running."""
    global _speech_thread
    with _speech_lock:
        _ensure_speech_queue()
        try:
            if (_speech_thread is not None) and _speech_thread.is_alive():
                return
        except Exception:
            pass
        try:
            _speech_thread = threading.Thread(target=_speech_worker, name='speech_worker', daemon=True)
            _speech_thread.start()
        except Exception:
            try:
                logging.exception('Failed to start speech worker')
            except Exception:
                pass

def speak_async(text: str, rate: float = None, voice: str = None, urgent: bool = False):
    """Non-blocking enqueue of speech routed through the main audio queue.

    Centralizes all speech to avoid overlapping between multiple workers.
    Drops oldest only if the main queue is saturated and urgent is False.

    Args:
        text: Message to speak.
        rate: Optional speech rate hint (0.0-1.0 typical).
        voice: Optional voice identifier (unused in local TTS).
        urgent: Hint for prioritization and preemption handling.
    """
    try:
        msg = str(text or '')
        now = time.time()
        payload = {
            'msg': msg,
            'enqueue_ts': now,
            'urgent': bool(urgent),
            'rate': rate if rate is not None else float(config.get('audio', {}).get('speech_rate_default', 0.5)),
            'level': 'info'
        }
        item = (0 if urgent else 2, now, payload)
        # Attempt enqueue; if queue is full and not urgent, drop oldest to keep responsive
        try:
            audio_queue.put_nowait(item)
        except Exception:
            if not urgent:
                try:
                    _ = audio_queue.get_nowait()
                    audio_queue.task_done()
                except Exception:
                    pass
                try:
                    audio_queue.put_nowait(item)
                except Exception:
                    pass
            else:
                # For urgent, set a preempt flag so worker can stop ongoing AVFoundation speech
                try:
                    with state.lock:
                        state._preempt_now_ts = now
                except Exception:
                    pass
                try:
                    audio_queue.put_nowait(item)
                except Exception:
                    pass
    except Exception:
        # Last resort: direct speak, but respect global gap
        try:
            speak(text)
        except Exception:
            pass


# -------- Default Config --------
DEFAULT_CONFIG = {
    "camera": {
        "width": 1280,
        "height": 720,
        "fps": 25,
        "camera_id": 0,
        "prefer_continuity": True
    },
    "spatial_audio": {
        "enabled": True,
        "intensity": 0.7,
        "distance_scaling": 1.0,
        "use_stereo": True,
        "horizontal_fov_deg": 60.0,  # Approximate camera horizontal FOV for angle mapping
        "min_angle_diff": 15,  # Minimum angle difference in degrees to trigger direction change
        "max_distance": 10.0,   # Maximum distance in meters for spatial audio cues
        "near_distance": 1.0,   # Distance at which volume scaling is near maximum
        "far_distance": 10.0,   # Distance at which volume scaling approaches minimum
        "curve": "linear",     # Volume scaling curve: linear|log|exp
        "min_interval_s": 0.2,  # Minimum interval between spatial earcons to reduce spam
        "noise_trigger_level": 0.7,  # If ambient noise >= this, boost earcon volume
        "noise_gain_boost": 0.15,    # Fractional volume boost under noise
        "pan_gain": 1.0,             # Gain applied to computed pan from bbox center
        "deadzone_pan_gain": 0.2,    # Pan gain when inside center corridor (deadzone)
        "pan_hint_ttl_s": 0.5        # How long a computed pan hint remains valid
    },
    "blind_navigation": {
        "enabled": True,
        "path_width_factor": 0.2,  # Width of central path as fraction of frame
        "trip_hazard_priority": 3,  # Extra priority for low obstacles
        "overhead_hazard_priority": 2,  # Extra priority for high obstacles
        "path_obstacle_priority": 3,  # Extra priority for obstacles in walking path
        "critical_distance_priority": 5,  # Extra priority for very close objects
        "important_classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck", 
                             "traffic light", "stop sign"],
        # Data-driven tuning
        "adaptive_tuning_enabled": True,
        "tuning_window_s": 900.0,  # Sliding window to aggregate real-world stats
        "weights": {
            "trip": 3.0,
            "overhead": 2.0,
            "path": 3.0,
            "critical": 5.0,
            "class_boost": 2.0
        },
        # Lane guidance
        "lane_guidance_enabled": True,
        "lane_min_free_ratio": 0.85,
        "lane_min_width_frac": 0.12,
        "lane_band_height_frac": 0.22,
        # Advanced lane detection sensitivity
        "depth_sensitivity": 0.8,  # Sensitivity for depth-based free space detection
        "obstacle_avoidance_margin": 1.2,  # Safety margin around obstacles (meters)
        "path_smoothing_factor": 0.3,  # Smoothing factor for path calculations (0-1)
        "confidence_threshold": 0.25,  # Minimum confidence for lane guidance
        "steering_responsiveness": 0.7,  # How quickly to respond to path changes (0-1)
        "narrow_path_threshold": 0.15,  # Threshold for detecting narrow passages
        "wide_path_preference": 1.5  # Preference multiplier for wider paths
    },
    "yolo": {
        "weights": "yolov8n.pt",
        "img_size": 448,
        "conf": 0.4,
        "iou": 0.5,
        "max_det": 50,
        "ensemble": False,
        "augment": False,
        "custom_classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign", "dog", "cat"],
        "tracker": True,
        "ema_smoothing": True,
        "use_coreml": False,
        "coreml_package_path": "models/yolo11n.mlpackage"
    },
    "depth": {
        "model": "depth-anything/Depth-Anything-V2-Small-hf",
        "alpha": 0.3,
        "min_interval": 0.04,
        "fast_mode": True,
        "resolution": 384,
        "skip_frames": 2,
        "batch_processing": False,
        "quantize_int8": True,
        # Source: 'model' (local/HF) or 'network' (external LiDAR)
        "source": "model",
        # Optional linear calibration for 0-255 grayscale depth → meters
        # If > 0, maps pixel value directly: meters = (value_0_255) * factor
        "calibration_factor_m_per_255": 0.0,
        # Expected units and scale for network depth (meters by default)
        "network_units": "meters",
        "network_scale": 1.0,
        # Fine-tuned local DepthAnythingV2 support
        "use_local_v2": False,
        "local_weights": "",
        "local_encoder": "vits"
    },
    "distances": {
        "warning": 2.0,
        "critical": 1.0,
        "nearby": 4.0
    },
    "speed_detection": {
        "fast_speed_threshold": 1.5,  # m/s
        "very_fast_threshold": 3.0,   # m/s
        "min_samples_for_speed": 5,
        "speed_window_seconds": 2.0,
        "kalman_q": 0.04,
        "kalman_r": 0.10,
        "filter_type": "kalman",  # "kalman" or "alpha_beta"
        "alpha": 0.65,
        "beta": 0.25,
        "roi_outside_speed_weight": 0.6
    },
    "announcements": {
        "enabled": True,
        "cooldown": 3.0,
        "class_cooldown": 4.0,
        "min_conf": 0.4,
        "ai_decider": True,
        "context_aware": True,
        "min_importance_score": 0.6,
        "silence_threshold": 5.0,
        "movement_threshold": 0.1,
        "realtime_mode": False,
        "smart_verbosity": True,
        "adaptive_cooldown": True,
        "cooldown_under_load_multiplier": 1.5,
        "summary_interval_s": 9.0,
        "per_track_cooldown_s": 2.5,
        "hint_cooldown_multiplier_in_warning": 2.0,
        "fixed_cadence_s": 0.0,
        "tone_style": "neutral",
        "verbosity_level": "standard",
        "supportive_prefixes": [],
        "supportive_suffixes": [],
        "person_importance_base_boost": 0.15,
        "person_speed_boost_mult": 1.15,
        "person_phrases_enabled": True
    },
    "audio": {
      "enqueue_rate_limit_s": 0.8,
      "max_queue_size": 2,
      "message_repeat_window_s": 4.0,
      "info_repeat_window_s": 12.0,
      "global_silence_after_critical_s": 2.5,
      "dedup_window_s": 2.0,
      "hint_cooldown_multiplier_in_warning": 2.0,
      "summary_interval_s": 9.0,
      "hint_block_after_critical_s": 3.0,
      "speech_volume": 1.0,
      "earcon_volume": 0.8,
      "earcons_enabled": True,
      "per_track_cooldown_s": 2.5
    },
    "recording": {
        "enabled": False,
        "output_dir": "recordings",
        "max_duration": 300  # seconds
    },
    "logging": {
        "level": "INFO",
        "file": "logs/third_eye.log"
    },
    "performance": {
        "max_workers": DEFAULT_MAX_WORKERS,
        "adaptive_quality": True,
        "target_fps": 25,
        "min_fps": 20,
        "quality_levels": [0.5, 0.7, 0.9, 1.0],
        "profiling": False,
        "frame_skip": 1,
        "depth_skip": 2,
        "yolo_skip": 1,
        "yolo_batch_enabled": True,
        "yolo_batch_size": 2,
        "yolo_batch_delay_ms": 5,
        "preproc_buffer_size": DEFAULT_PREPROC_BUF,
        "max_frame_skip": 3,
        "max_depth_skip": 3,
        "max_yolo_skip": 2
    }
    ,
    "watchdog": {
        "yolo_stall_s": 2.5,
        "depth_stall_s": 2.5,
        "min_restart_interval_s": 5.0,
        "log_file": "logs/depth_system.log"
    }
    ,
    "hazard": {
        "boost_duration_s": 1.6,
        "yolo_interval_s": 0.05,
        "yolo_interval_critical_s": 0.03,
        "depth_min_interval_s": 0.02,
        "yolo_imgsz_boost": 448,
        "yolo_conf_floor": 0.25,
        "yolo_conf_critical_floor": 0.20,
        "roi_shift_on_hazard": True,
        "roi_lane_bias": 0.75,
        "center_probe_width_frac": 0.34,
        "center_probe_height_frac": 0.34,
        "looming_speed_threshold_mps": 1.2,
        "looming_depth_threshold_m": 10.0,
        "looming_cooldown_s": 2.5
    }
    ,
    "free_space": {
        "depth_free_m": 3.0,
        "kernel_open": 5,
        "center_band_height_frac": 0.22,
        "min_corridor_frac": 0.55,
        "persistence_s": 1.2
    }
    ,
    "proximity": {
        "weight_depth": 0.50,
        "weight_speed": 0.30,
        "weight_area": 0.20,
        "max_speed_for_conf_mps": 2.0,
        "area_frac_max_for_conf": 0.30,
        "importance_weight": 0.60,
        "warn_gate_threshold": 0.65,
        "crit_gate_threshold": 0.85
    }
}

# Load config from file or use defaults
def load_config(config_path="config.json"):
    if os.path.exists(config_path):
        try:
            # Read raw content first for better error reporting on parse failures
            with open(config_path, 'r') as f:
                raw = f.read()
                config = json.loads(raw)
                # Merge with defaults
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
                return config
        except Exception as e:
            # Config parse failed: preserve user edits and warn loudly
            try:
                strict = str(os.environ.get('THIRDEYE_STRICT_CONFIG', '') or '').strip() == '1'
            except Exception:
                strict = False
            try:
                os.makedirs('logs', exist_ok=True)
            except Exception:
                pass
            backup_path = None
            try:
                # Save the raw, problematic config to a timestamped backup for recovery
                ts = time.strftime('%Y%m%d-%H%M%S')
                backup_path = os.path.join('logs', f'config_load_error_{ts}.json')
                # If raw was defined, write it; otherwise re-open the file
                try:
                    raw_to_write = raw  # may not exist if failure happened before read
                except Exception:
                    try:
                        with open(config_path, 'r') as f:
                            raw_to_write = f.read()
                    except Exception:
                        raw_to_write = ''
                with open(backup_path, 'w') as bf:
                    bf.write(raw_to_write)
                # Also write a small marker file with the error message
                with open(os.path.join('logs', 'LAST_CONFIG_ERROR.txt'), 'w') as mf:
                    mf.write(f'Error loading {config_path}: {e}\nBackup: {backup_path}\n')
            except Exception:
                pass
            try:
                logging.error(f"Error loading config '{config_path}': {e}. Falling back to defaults. Backup at: {backup_path or 'logs/'}")
            except Exception:
                pass
            if strict:
                # Fail fast in strict mode so the app does not start with defaults silently
                raise
    return DEFAULT_CONFIG

# Centralized gate for Core ML usage
def coreml_is_enabled(config):
    try:
        # Optional environment override to force Ultralytics-only (.pt) usage
        if str(os.environ.get('THIRDEYE_FORCE_ULTRA_ONLY', '')).strip() == '1':
            return False
    except Exception:
        # If environment lookup fails, ignore and fall back to config
        pass
    return bool(config.get('yolo', {}).get('use_coreml', False))

def validate_config(config):
    # Strengthened schema validation with clamping for safety
    def clamp(getter, setter, min_val, max_val, cast=float):
        try:
            val = cast(getter())
        except Exception:
            return
        if val < min_val:
            setter(min_val)
        elif val > max_val:
            setter(max_val)

    # Camera
    clamp(
        lambda: config["camera"].get("width", 1280),
        lambda v: config["camera"].__setitem__("width", int(v)),
        160,
        7680,
        int,
    )
    clamp(
        lambda: config["camera"].get("height", 720),
        lambda v: config["camera"].__setitem__("height", int(v)),
        120,
        4320,
        int,
    )
    clamp(
        lambda: config["camera"].get("fps", 25),
        lambda v: config["camera"].__setitem__("fps", int(v)),
        1,
        120,
        int,
    )
    # Continuity Camera preference toggle
    try:
        config["camera"]["prefer_continuity"] = bool(config["camera"].get("prefer_continuity", sys.platform == 'darwin'))
    except Exception:
        config["camera"]["prefer_continuity"] = (sys.platform == 'darwin')

    # Performance FPS bounds and relationships
    clamp(
        lambda: config.get("performance", {}).get("target_fps", 25),
        lambda v: config.setdefault("performance", {}).__setitem__("target_fps", int(v)),
        1,
        120,
        int,
    )
    clamp(
        lambda: config.get("performance", {}).get("min_fps", 12),
        lambda v: config.setdefault("performance", {}).__setitem__("min_fps", int(v)),
        1,
        120,
        int,
    )
    try:
        tfps = int(config.get("performance", {}).get("target_fps", config["camera"].get("fps", 25)))
        mfps = int(config.get("performance", {}).get("min_fps", 12))
        if mfps > tfps:
            config.setdefault("performance", {})["min_fps"] = tfps
    except Exception:
        pass

    # Queues
    clamp(
        lambda: config.get("performance", {}).get("preproc_buffer_size", 6),
        lambda v: config.setdefault("performance", {}).__setitem__("preproc_buffer_size", int(v)),
        1,
        64,
        int,
    )
    # YOLO micro-batching knobs
    try:
        config.setdefault("performance", {})["yolo_batch_enabled"] = bool(
            config.get("performance", {}).get("yolo_batch_enabled", True)
        )
    except Exception:
        config.setdefault("performance", {})["yolo_batch_enabled"] = True
    clamp(
        lambda: config.get("performance", {}).get("yolo_batch_size", 1),
        lambda v: config.setdefault("performance", {}).__setitem__("yolo_batch_size", int(v)),
        1,
        16,
        int,
    )
    clamp(
        lambda: config.get("performance", {}).get("yolo_batch_delay_ms", 0),
        lambda v: config.setdefault("performance", {}).__setitem__("yolo_batch_delay_ms", int(v)),
        0,
        50,
        int,
    )
    clamp(
        lambda: config.get("audio", {}).get("max_queue_size", 2),
        lambda v: config.setdefault("audio", {}).__setitem__("max_queue_size", int(v)),
        1,
        100,
        int,
    )

    # Intervals (seconds)
    clamp(
        lambda: config.get("audio", {}).get("enqueue_rate_limit_s", 0.8),
        lambda v: config.setdefault("audio", {}).__setitem__("enqueue_rate_limit_s", float(v)),
        0.0,
        10.0,
        float,
    )
    clamp(
        lambda: config.get("audio", {}).get("dedup_window_s", 2.0),
        lambda v: config.setdefault("audio", {}).__setitem__("dedup_window_s", float(v)),
        0.0,
        60.0,
        float,
    )
    clamp(
        lambda: config.get("hazard", {}).get("yolo_interval_s", 0.05),
        lambda v: config.setdefault("hazard", {}).__setitem__("yolo_interval_s", float(v)),
        0.01,
        1.0,
        float,
    )
    clamp(
        lambda: config.get("hazard", {}).get("yolo_interval_critical_s", 0.03),
        lambda v: config.setdefault("hazard", {}).__setitem__("yolo_interval_critical_s", float(v)),
        0.005,
        1.0,
        float,
    )
    clamp(
        lambda: config.get("hazard", {}).get("depth_min_interval_s", 0.02),
        lambda v: config.setdefault("hazard", {}).__setitem__("depth_min_interval_s", float(v)),
        0.01,
        1.0,
        float,
    )
    # Confidence floors
    try:
        cf = float(config.get("hazard", {}).get("yolo_conf_floor", 0.3))
        config.setdefault("hazard", {})["yolo_conf_floor"] = max(0.0, min(1.0, cf))
    except Exception:
        config.setdefault("hazard", {})["yolo_conf_floor"] = 0.3
    try:
        cfc = float(config.get("hazard", {}).get("yolo_conf_critical_floor", 0.25))
        config.setdefault("hazard", {})["yolo_conf_critical_floor"] = max(0.0, min(1.0, cfc))
    except Exception:
        config.setdefault("hazard", {})["yolo_conf_critical_floor"] = 0.25
    # ROI hazard shift options
    try:
        rshift = bool(config.get("hazard", {}).get("roi_shift_on_hazard", True))
        config.setdefault("hazard", {})["roi_shift_on_hazard"] = rshift
    except Exception:
        config.setdefault("hazard", {})["roi_shift_on_hazard"] = True
    try:
        rbias = float(config.get("hazard", {}).get("roi_lane_bias", 0.75))
        config.setdefault("hazard", {})["roi_lane_bias"] = max(0.3, min(0.95, rbias))
    except Exception:
        config.setdefault("hazard", {})["roi_lane_bias"] = 0.75
    clamp(
        lambda: config.get("watchdog", {}).get("yolo_stall_s", 2.5),
        lambda v: config.setdefault("watchdog", {}).__setitem__("yolo_stall_s", float(v)),
        0.1,
        30.0,
        float,
    )
    clamp(
        lambda: config.get("watchdog", {}).get("depth_stall_s", 2.5),
        lambda v: config.setdefault("watchdog", {}).__setitem__("depth_stall_s", float(v)),
        0.1,
        30.0,
        float,
    )
    clamp(
        lambda: config.get("watchdog", {}).get("min_restart_interval_s", 5.0),
        lambda v: config.setdefault("watchdog", {}).__setitem__("min_restart_interval_s", float(v)),
        0.5,
        120.0,
        float,
    )

    # Distances: enforce sensible bounds and relationships
    clamp(
        lambda: config.get("distances", {}).get("critical", 1.0),
        lambda v: config.setdefault("distances", {}).__setitem__("critical", float(v)),
        0.1,
        10.0,
        float,
    )
    clamp(
        lambda: config.get("distances", {}).get("warning", 2.0),
        lambda v: config.setdefault("distances", {}).__setitem__("warning", float(v)),
        0.2,
        20.0,
        float,
    )
    clamp(
        lambda: config.get("distances", {}).get("nearby", 4.0),
        lambda v: config.setdefault("distances", {}).__setitem__("nearby", float(v)),
        0.5,
        50.0,
        float,
    )
    try:
        d = config.setdefault("distances", {})
        # Ensure critical <= warning <= nearby
        if d["warning"] < d["critical"]:
            d["warning"] = d["critical"]
        if d["nearby"] < d["warning"]:
            d["nearby"] = d["warning"]
    except Exception:
        pass

    # Depth calibration factor: linear mapping from 0-255 grayscale to meters
    clamp(
        lambda: config.get("depth", {}).get("calibration_factor_m_per_255", 0.0),
        lambda v: config.setdefault("depth", {}).__setitem__("calibration_factor_m_per_255", float(v)),
        0.0,
        5.0,
        float,
    )

    # YOLO schema
    clamp(
        lambda: config.get("yolo", {}).get("img_size", 640),
        lambda v: config.setdefault("yolo", {}).__setitem__("img_size", int(v)),
        128,
        2048,
        int,
    )
    clamp(
        lambda: config.get("yolo", {}).get("conf", 0.25),
        lambda v: config.setdefault("yolo", {}).__setitem__("conf", float(v)),
        0.0,
        1.0,
        float,
    )
    clamp(
        lambda: config.get("yolo", {}).get("iou", 0.4),
        lambda v: config.setdefault("yolo", {}).__setitem__("iou", float(v)),
        0.0,
        1.0,
        float,
    )
    clamp(
        lambda: config.get("yolo", {}).get("max_det", 25),
        lambda v: config.setdefault("yolo", {}).__setitem__("max_det", int(v)),
        1,
        300,
        int,
    )
    try:
        ys = config.setdefault("yolo", {})
        ys["tracker"] = bool(ys.get("tracker", True))
        ys["ema_smoothing"] = bool(ys.get("ema_smoothing", True))
        ys["use_coreml"] = bool(ys.get("use_coreml", False))
        cc = ys.get("custom_classes", [])
        if not isinstance(cc, list):
            cc = []
        ys["custom_classes"] = [str(x) for x in cc]
        pkg = str(ys.get("coreml_package_path", "models/yolo11n.mlpackage") or "models/yolo11n.mlpackage")
        ys["coreml_package_path"] = pkg
    except Exception:
        pass

    # Hazard schema (additional bounds)
    clamp(
        lambda: config.get("hazard", {}).get("looming_speed_threshold_mps", 1.2),
        lambda v: config.setdefault("hazard", {}).__setitem__("looming_speed_threshold_mps", float(v)),
        0.1,
        10.0,
        float,
    )
    clamp(
        lambda: config.get("hazard", {}).get("looming_depth_threshold_m", 2.0),
        lambda v: config.setdefault("hazard", {}).__setitem__("looming_depth_threshold_m", float(v)),
        0.2,
        50.0,
        float,
    )
    clamp(
        lambda: config.get("hazard", {}).get("looming_cooldown_s", 3.0),
        lambda v: config.setdefault("hazard", {}).__setitem__("looming_cooldown_s", float(v)),
        0.0,
        60.0,
        float,
    )

    # Ports (if provided)
    def clamp_port(val):
        try:
            p = int(val)
            return max(1024, min(65535, p))
        except Exception:
            return None
    ann = config.setdefault("announcements", {})
    sp = ann.get("shortcuts_port")
    if sp is not None:
        cp = clamp_port(sp)
        if cp is not None:
            ann["shortcuts_port"] = cp
    imu = ann.get("imu_udp_port")
    if imu is not None:
        cp = clamp_port(imu)
        if cp is not None:
            ann["imu_udp_port"] = cp
    # Shortcuts bind default and timeout clamp
    try:
        bind = str(ann.get("shortcuts_bind", "127.0.0.1") or "127.0.0.1")
        ann["shortcuts_bind"] = bind
    except Exception:
        ann["shortcuts_bind"] = "127.0.0.1"
    clamp(
        lambda: ann.get("shortcuts_timeout_s", 1.0),
        lambda v: ann.__setitem__("shortcuts_timeout_s", float(v)),
        0.05,
        60.0,
        float,
    )
    # Telemetry settings
    logg = config.setdefault("logging", {})
    try:
        logg["telemetry_enabled"] = bool(logg.get("telemetry_enabled", True))
    except Exception:
        logg["telemetry_enabled"] = True
    clamp(
        lambda: logg.get("telemetry_interval_s", 30.0),
        lambda v: logg.__setitem__("telemetry_interval_s", float(v)),
        1.0,
        86400.0,
        float,
    )
    try:
        logg["telemetry_include_recent_events"] = bool(logg.get("telemetry_include_recent_events", True))
    except Exception:
        logg["telemetry_include_recent_events"] = True
    try:
        trf = logg.get("telemetry_redact_fields", [])
        if not isinstance(trf, list):
            trf = []
        logg["telemetry_redact_fields"] = trf
    except Exception:
        logg["telemetry_redact_fields"] = []

    # Logging console level sanity
    try:
        lvl = str(logg.get("console_level", "WARNING") or "WARNING").upper()
        if lvl not in ("DEBUG","INFO","WARNING","ERROR","CRITICAL"):
            logg["console_level"] = "WARNING"
    except Exception:
        logg["console_level"] = "WARNING"

    # Audio schema and volume clamping
    clamp(
        lambda: config.get("audio", {}).get("global_min_speak_gap_s", 0.9),
        lambda v: config.setdefault("audio", {}).__setitem__("global_min_speak_gap_s", float(v)),
        0.1,
        5.0,
        float,
    )
    try:
        config.setdefault("audio", {})["earcons_enabled"] = bool(config.get("audio", {}).get("earcons_enabled", False))
    except Exception:
        config.setdefault("audio", {})["earcons_enabled"] = False
    clamp(
        lambda: config.get("audio", {}).get("speech_volume", 1.0),
        lambda v: config.setdefault("audio", {}).__setitem__("speech_volume", float(v)),
        0.0,
        1.0,
        float,
    )
    clamp(
        lambda: config.get("audio", {}).get("earcon_volume", 0.8),
        lambda v: config.setdefault("audio", {}).__setitem__("earcon_volume", float(v)),
        0.0,
        1.0,
        float,
    )


def apply_profile(config, name=None):
    """Apply performance profiles: low-power, balanced, high-performance.
    Balanced applies no changes. Profiles override select keys.
    The profile can be supplied via argument, config['profile'], or env var THIRD_EYE_PROFILE.
    """
    profile = (name or config.get("profile") or os.environ.get("THIRD_EYE_PROFILE") or "balanced").lower()
    overrides = {}
    if profile == "low-power":
        overrides = {
            "performance": {
                "target_fps": 20,
                "min_fps": 18,
                "frame_skip": 2,
                "depth_skip": 3,
                "yolo_skip": 2,
                "preproc_buffer_size": 4,
                "max_workers": max(1, int(config.get("performance", {}).get("max_workers", 4) - 1)),
            },
            "yolo": {"img_size": 256},
            "depth": {"resolution": 192, "min_interval": max(0.06, float(config.get("depth", {}).get("min_interval", 0.04)))},
            "audio": {"enqueue_rate_limit_s": 1.2, "max_queue_size": 1},
            "hazard": {"yolo_interval_s": 0.08, "depth_min_interval_s": 0.06},
        }
    elif profile == "high-performance":
        overrides = {
            "performance": {
                "target_fps": 30,
                "min_fps": 25,
                "frame_skip": 1,
                "depth_skip": 1,
                "yolo_skip": 1,
                "preproc_buffer_size": 8,
                "max_workers": max(4, int(config.get("performance", {}).get("max_workers", 4) + 2)),
            },
            "yolo": {"img_size": 448, "weights": "yolov8s.pt", "conf": 0.5, "max_det": 30},
            "depth": {"resolution": 384, "min_interval": 0.03},
            "audio": {"enqueue_rate_limit_s": 0.5, "max_queue_size": 3},
            "hazard": {"yolo_interval_s": 0.05, "depth_min_interval_s": 0.02},
        }
    elif profile == "ultra-performance":
        overrides = {
            "performance": {
                "target_fps": 25,
                "min_fps": 20,
                "frame_skip": 1,
                "depth_skip": 1,
                "yolo_skip": 1,
                "preproc_buffer_size": 12,
                "max_workers": max(6, int(config.get("performance", {}).get("max_workers", 4) + 3)),
            },
            "yolo": {"img_size": 512, "weights": "yolov8m.pt", "conf": 0.45, "max_det": 50, "augment": True},
            "depth": {"resolution": 448, "min_interval": 0.025},
            "audio": {"enqueue_rate_limit_s": 0.4, "max_queue_size": 4},
            "hazard": {"yolo_interval_s": 0.04, "depth_min_interval_s": 0.015},
        }
    else:
        # balanced: do nothing
        return

    # Merge overrides
    for section, vals in overrides.items():
        cfg_sec = config.setdefault(section, {})
        for k, v in vals.items():
            cfg_sec[k] = v

    # Re-validate after profile application
    validate_config(config)

# Global config
config = load_config()
apply_profile(config)
validate_config(config)

# Debug: log active YOLO class filter to verify runtime behavior
try:
    logging.info(f"YOLO custom_classes: {config.get('yolo', {}).get('custom_classes', [])}")
except Exception:
    pass

# Load persisted user preferences and apply overrides
try:
    import json as _json, os as _os
    _prefs_path = _os.path.join('logs', 'prefs.json')
    if _os.path.exists(_prefs_path):
        try:
            with open(_prefs_path, 'r') as _pf:
                _prefs = _json.load(_pf)
        except Exception:
            _prefs = {}
    else:
        _prefs = {}
    ann = config.setdefault('announcements', {})
    aud = config.setdefault('audio', {})
    # Style profile
    sp = str(_prefs.get('style_profile', ann.get('style_profile', 'concise')) or 'concise').lower()
    ann['style_profile'] = sp
    try:
        with state.lock:
            state.ai_style_profile = sp
    except Exception:
        pass
    # AI phrasing toggles
    if 'phrasing_enabled' in _prefs:
        try:
            ann['phrasing_enabled'] = bool(_prefs.get('phrasing_enabled'))
        except Exception:
            pass
    if 'phrasing_provider' in _prefs:
        ann['phrasing_provider'] = str(_prefs.get('phrasing_provider') or ann.get('phrasing_provider', 'builtin'))
    # TTS provider and voice
    if 'tts_provider' in _prefs:
        aud['tts_provider'] = str(_prefs.get('tts_provider') or aud.get('tts_provider', 'local'))
    if 'voice_id' in _prefs:
        aud['voice_id'] = str(_prefs.get('voice_id') or aud.get('voice_id', ''))
except Exception:
    pass

# Extract config values
CAMW = config["camera"]["width"]
CAMH = config["camera"]["height"]
TARGET_FPS = config["camera"]["fps"]
CAMERA_ID = config["camera"]["camera_id"]
MOCK_SENSOR = str(os.environ.get('THIRD_EYE_MOCK_SENSOR', '0')).lower() in ('1','true','yes')

YOLO_WEIGHTS = config["yolo"]["weights"]
YOLO_IMG_SIZE = config["yolo"]["img_size"]
YOLO_CONF = config["yolo"]["conf"]
YOLO_IOU = config["yolo"]["iou"]
MAX_DET = config["yolo"]["max_det"]

DEPTH_MODEL = config["depth"]["model"]
DEPTH_ALPHA = config["depth"]["alpha"]
DEPTH_MIN_INTERVAL = config["depth"]["min_interval"]
FAST_DEPTH = config["depth"]["fast_mode"]
# Quantization toggle: env overrides config when set
DEPTH_QUANTIZE = (
    str(os.environ.get('THIRD_EYE_DEPTH_QUANTIZE', '')).lower() in ('1','true','yes')
    if os.environ.get('THIRD_EYE_DEPTH_QUANTIZE') is not None
    else bool(config.get('depth', {}).get('quantize_int8', False))
)

WARNING_DISTANCE_M = config["distances"]["warning"]
CRITICAL_DISTANCE_M = config["distances"]["critical"]
NEARBY_DISTANCE_M = config["distances"]["nearby"]

FAST_SPEED_THRESHOLD = config["speed_detection"]["fast_speed_threshold"]
VERY_FAST_THRESHOLD = config["speed_detection"]["very_fast_threshold"]
MIN_SAMPLES_FOR_SPEED = config["speed_detection"]["min_samples_for_speed"]
SPEED_WINDOW_SECONDS = config["speed_detection"]["speed_window_seconds"]

# Build logging handlers with optional rotation
_app_log_file = config.get('logging', {}).get('file', 'logs/third_eye.log')
_rotation_type = str(config.get('logging', {}).get('rotation', '') or '').lower()  # 'size' or 'time'
_handlers = []
try:
    if _app_log_file:
        if _rotation_type == 'size':
            _max_bytes = int(config.get('logging', {}).get('max_bytes', 10 * 1024 * 1024))
            _backup_count = int(config.get('logging', {}).get('backup_count', 5))
            _fh = logging.handlers.RotatingFileHandler(_app_log_file, maxBytes=_max_bytes, backupCount=_backup_count)
        elif _rotation_type == 'time':
            _when = str(config.get('logging', {}).get('when', 'midnight'))
            _interval = int(config.get('logging', {}).get('interval', 1))
            _backup_count = int(config.get('logging', {}).get('backup_count', 7))
            _fh = logging.handlers.TimedRotatingFileHandler(_app_log_file, when=_when, interval=_interval, backupCount=_backup_count)
        else:
            _fh = logging.FileHandler(_app_log_file)
        _handlers.append(_fh)
except Exception:
    _handlers.append(logging.FileHandler(_app_log_file))
# Console handler with configurable level (default WARNING to reduce terminal noise)
_console_level_name = str(config.get('logging', {}).get('console_level', 'WARNING')).upper()
_console_level = getattr(logging, _console_level_name, logging.WARNING)
_ch = logging.StreamHandler()
try:
    _ch.setLevel(_console_level)
except Exception:
    # Fallback in case level setting fails
    _ch.setLevel(logging.WARNING)
_handlers.append(_ch)
logging.basicConfig(
    level=getattr(logging, str(config.get('logging', {}).get('level', 'INFO')).upper(), logging.INFO),
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=_handlers
)

# Dedicated system health logger for watchdog events
try:
    _wd_log_file = config.get('watchdog', {}).get('log_file', 'logs/depth_system.log')
    system_logger = logging.getLogger('DepthSystem')
    system_logger.setLevel(logging.INFO)
    if _wd_log_file:
        if _rotation_type == 'size':
            _max_bytes = int(config.get('logging', {}).get('max_bytes', 10 * 1024 * 1024))
            _backup_count = int(config.get('logging', {}).get('backup_count', 5))
            _wd_handler = logging.handlers.RotatingFileHandler(_wd_log_file, maxBytes=_max_bytes, backupCount=_backup_count)
        elif _rotation_type == 'time':
            _when = str(config.get('logging', {}).get('when', 'midnight'))
            _interval = int(config.get('logging', {}).get('interval', 1))
            _backup_count = int(config.get('logging', {}).get('backup_count', 7))
            _wd_handler = logging.handlers.TimedRotatingFileHandler(_wd_log_file, when=_when, interval=_interval, backupCount=_backup_count)
        else:
            _wd_handler = logging.FileHandler(_wd_log_file)
        _wd_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] DepthSystem: %(message)s'))
        system_logger.addHandler(_wd_handler)
    system_logger.propagate = False
except Exception:
    system_logger = logging.getLogger('DepthSystemFallback')

# --- Standardized error handling utilities ---
_EH_DEFAULT_RATE_LIMIT_S = 5.0

def log_error(tag, exc_or_msg, level='warning', rate_limit_s=_EH_DEFAULT_RATE_LIMIT_S):
    """Log an error/warning/info with rate limiting and counters.

    This helper centralizes error reporting across threads and hardware/inference
    paths. It increments `state.error_counts[tag]`, applies a per-tag rate limit
    to avoid log spam, and emits via the standard `logging` module.

    Args:
        tag: Short category (e.g., 'camera_read', 'yolo_predict', 'depth_post') used
            for both counters and rate-limiting buckets.
        exc_or_msg: Exception instance or message string to be logged.
        level: Logging level name ('warning', 'error', 'info').
        rate_limit_s: Minimum seconds between logs for the same tag.

    Returns:
        None. Side effects only: updates `state.error_counts` and writes logs.
    """
    """
    Rate-limited structured error logging that also tracks per-tag counts.

    Args:
        tag: Short context key (e.g., 'YOLO CoreML predict', 'Camera read').
        exc_or_msg: Exception instance or string message to log.
        level: Logging level name ('warning', 'error', 'debug', 'info').
        rate_limit_s: Minimum seconds between logs for the same tag.
    """
    try:
        now = time.time()
        last = getattr(state, '_last_error_logs', {}).get(tag, 0.0)
        if (now - float(last)) >= float(rate_limit_s):
            msg = exc_or_msg if isinstance(exc_or_msg, str) else str(exc_or_msg)
            try:
                getattr(logging, level)(f"{tag}: {msg}")
            except Exception:
                logging.warning(f"{tag}: {msg}")
            try:
                state._last_error_logs[tag] = now
            except Exception:
                pass
        # Increment counters regardless of rate limit
        try:
            state.error_counts[tag] = int(state.error_counts.get(tag, 0)) + 1
        except Exception:
            pass
    except Exception:
        # Fallback plain warning if logging bookkeeping fails
        try:
            msg = exc_or_msg if isinstance(exc_or_msg, str) else str(exc_or_msg)
            logging.warning(f"{tag}: {msg}")
        except Exception:
            pass


def safe_execute(tag, func, fallback=None, level='warning', rate_limit_s=_EH_DEFAULT_RATE_LIMIT_S):
    """Execute a callable and gracefully handle exceptions.

    Wraps call sites prone to transient failures (I/O, inference, postprocessing)
    and ensures the pipeline continues by returning a reasonable fallback value.
    Errors are reported through `log_error()` with rate limiting and counters.

    Args:
        tag: Error category (same semantics as `log_error`).
        func: Zero-argument callable to execute.
        fallback: Value to return if `func()` raises. If callable, it will be
            invoked to produce the fallback.
        level: Logging level for errors.
        rate_limit_s: Rate limit for error logs.

    Returns:
        The result of `func()` on success, otherwise the provided `fallback` or
        `None` when no fallback is specified.
    """
    """
    Execute a callable with standardized error logging and graceful fallback.

    Args:
        tag: Short context key for log messages.
        func: Callable to execute.
        fallback: Value or callable to return on failure.
        level: Logging level for exceptions.
        rate_limit_s: Minimum seconds between logs for the same tag.

    Returns:
        The function result, or the fallback result/value on error.
    """
    try:
        return func()
    except Exception as e:
        log_error(tag, e, level=level, rate_limit_s=rate_limit_s)
        try:
            return fallback() if callable(fallback) else fallback
        except Exception as fe:
            # If fallback itself fails, log once and return None
            log_error(f"{tag} fallback", fe, level=level, rate_limit_s=rate_limit_s)
            return None

# Global runtime flags
HEADLESS = False


# Lightweight profiling context manager and resource sampler
class profile_scope(contextlib.ContextDecorator):
    """Measure and record duration for a named code section.

    Records elapsed time into `state.performance_stats['profile'][name]` with a
    rolling window. Logs a debug message when the duration exceeds the
    configurable threshold `performance.profile_log_threshold_ms` (default 50ms).

    Usage:
        with profile_scope('yolo.inference'):
            ...
    """
    def __init__(self, name: str, threshold_ms: float = None):
        self.name = str(name)
        try:
            self.threshold_ms = float(threshold_ms) if threshold_ms is not None else float(config.get('performance', {}).get('profile_log_threshold_ms', 50.0))
        except Exception:
            self.threshold_ms = 50.0
        try:
            self.window = int(config.get('performance', {}).get('profile_window', 50))
        except Exception:
            self.window = 50
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self._t0
        try:
            prof = state.performance_stats.setdefault('profile', {})
            arr = prof.setdefault(self.name, [])
            arr.append(dt)
            if len(arr) > self.window:
                arr.pop(0)
            if (dt * 1000.0) >= self.threshold_ms:
                logging.debug(f"Profile[{self.name}]: {dt*1000.0:.1f}ms")
        except Exception:
            pass
        # Don't suppress exceptions
        return False


def sample_resources():
    """Append current resource metrics (CPU, memory, GPU if available).

    - GPU CUDA memory (allocated/reserved) when `torch.cuda.is_available()`.
    - Keeps short rolling windows to limit memory use.
    """
    try:
        # GPU memory sampling (CUDA)
        if torch.cuda.is_available():
            try:
                alloc_mb = float(torch.cuda.memory_allocated(0) / (1024 * 1024))
                reserved_mb = float(torch.cuda.memory_reserved(0) / (1024 * 1024))
            except Exception:
                alloc_mb = 0.0
                reserved_mb = 0.0
            ps = state.performance_stats
            ps.setdefault('gpu_mem_alloc_mb', []).append(alloc_mb)
            ps.setdefault('gpu_mem_reserved_mb', []).append(reserved_mb)
            if len(ps['gpu_mem_alloc_mb']) > 30:
                ps['gpu_mem_alloc_mb'].pop(0)
            if len(ps['gpu_mem_reserved_mb']) > 30:
                ps['gpu_mem_reserved_mb'].pop(0)
    except Exception:
        pass


# Optimized shared preprocessing and tensor management for faster YOLO-Depth interconnection
class OptimizedFrameProcessor:
    """Shared frame processor to optimize YOLO and Depth Anything interconnection"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.yolo_cache = {}
        self.depth_cache = {}
        self.tensor_pool = {}
        self._init_tensor_pool()
    
    def _init_tensor_pool(self):
        """Pre-allocate tensors for common sizes to avoid repeated allocation"""
        common_sizes = [(640, 480), (512, 384), (448, 336), (320, 240)]
        for w, h in common_sizes:
            # Pre-allocate RGB and BGR tensors
            self.tensor_pool[f'rgb_{w}x{h}'] = np.zeros((h, w, 3), dtype=np.uint8)
            self.tensor_pool[f'bgr_{w}x{h}'] = np.zeros((h, w, 3), dtype=np.uint8)
    
    def get_or_create_tensor(self, shape, dtype=np.uint8, key_prefix='temp'):
        """Get pre-allocated tensor or create new one"""
        h, w = shape[:2]
        key = f'{key_prefix}_{w}x{h}'
        if key not in self.tensor_pool:
            self.tensor_pool[key] = np.zeros((h, w, 3), dtype=dtype)
        return self.tensor_pool[key]
    
    def process_frame_unified(self, frame, yolo_size=None, depth_size=None):
        """Unified frame processing for both YOLO and depth with shared operations"""
        frame_id = id(frame)
        current_time = time.time()
        
        # Check cache for recent processing
        if frame_id in self.yolo_cache and frame_id in self.depth_cache:
            yolo_data, yolo_time = self.yolo_cache[frame_id]
            depth_data, depth_time = self.depth_cache[frame_id]
            if (current_time - yolo_time < 0.1) and (current_time - depth_time < 0.1):
                return yolo_data, depth_data
        
        h, w = frame.shape[:2]
        
        # Determine optimal sizes
        yolo_w = yolo_size or YOLO_IMG_SIZE
        yolo_h = int(h * yolo_w / w)
        depth_w = depth_size or int(getattr(state, 'dynamic_depth_resolution', config["depth"]["resolution"]))
        depth_h = int(h * depth_w / w)
        
        # Shared preprocessing - convert to RGB once
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Optimized resizing with shared interpolation
        yolo_processed = None
        depth_processed = None
        
        if yolo_w == depth_w and yolo_h == depth_h:
            # Same size - process once
            resized = cv2.resize(rgb_frame, (yolo_w, yolo_h), interpolation=cv2.INTER_LINEAR)
            yolo_processed = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)  # YOLO expects BGR
            depth_processed = resized  # Depth expects RGB
        else:
            # Different sizes - optimize with parallel processing
            if yolo_w and yolo_h:
                yolo_processed = cv2.resize(frame, (yolo_w, yolo_h), interpolation=cv2.INTER_LINEAR)
            if depth_w and depth_h:
                depth_processed = cv2.resize(rgb_frame, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
        
        # Cache results
        if yolo_processed is not None:
            self.yolo_cache[frame_id] = (yolo_processed, current_time)
        if depth_processed is not None:
            self.depth_cache[frame_id] = (depth_processed, current_time)
        
        # Cleanup old cache entries
        self._cleanup_cache()
        
        return yolo_processed, depth_processed
    
    def _cleanup_cache(self):
        """Remove old cache entries to prevent memory leaks"""
        current_time = time.time()
        cutoff = current_time - 0.5  # Keep cache for 500ms
        
        # Cleanup YOLO cache
        old_keys = [k for k, (_, t) in self.yolo_cache.items() if t < cutoff]
        for k in old_keys:
            del self.yolo_cache[k]
        
        # Cleanup depth cache
        old_keys = [k for k, (_, t) in self.depth_cache.items() if t < cutoff]
        for k in old_keys:
            del self.depth_cache[k]
    
    def process_frame_parallel(self, frame, yolo_size=None, depth_size=None):
        """Parallel processing for YOLO and depth frames using threading"""
        import concurrent.futures
        
        frame_id = id(frame)
        current_time = time.time()
        
        # Check cache first
        if frame_id in self.yolo_cache and frame_id in self.depth_cache:
            yolo_data, yolo_time = self.yolo_cache[frame_id]
            depth_data, depth_time = self.depth_cache[frame_id]
            if (current_time - yolo_time < 0.1) and (current_time - depth_time < 0.1):
                return yolo_data, depth_data
        
        h, w = frame.shape[:2]
        yolo_w = yolo_size or YOLO_IMG_SIZE
        yolo_h = int(h * yolo_w / w)
        depth_w = depth_size or int(getattr(state, 'dynamic_depth_resolution', config["depth"]["resolution"]))
        depth_h = int(h * depth_w / w)
        
        def process_yolo():
            """Process frame for YOLO in parallel"""
            if yolo_w == depth_w and yolo_h == depth_h:
                # Will be handled by unified path
                return None
            return cv2.resize(frame, (yolo_w, yolo_h), interpolation=cv2.INTER_LINEAR)
        
        def process_depth():
            """Process frame for depth in parallel"""
            if yolo_w == depth_w and yolo_h == depth_h:
                # Will be handled by unified path
                return None
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return cv2.resize(rgb_frame, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
        
        # Handle same-size optimization
        if yolo_w == depth_w and yolo_h == depth_h:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_frame, (yolo_w, yolo_h), interpolation=cv2.INTER_LINEAR)
            yolo_processed = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            depth_processed = resized
        else:
            # Parallel processing for different sizes
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                yolo_future = executor.submit(process_yolo)
                depth_future = executor.submit(process_depth)
                
                yolo_processed = yolo_future.result()
                depth_processed = depth_future.result()
        
        # Cache results
        if yolo_processed is not None:
            self.yolo_cache[frame_id] = (yolo_processed, current_time)
        if depth_processed is not None:
            self.depth_cache[frame_id] = (depth_processed, current_time)
        
        self._cleanup_cache()
        return yolo_processed, depth_processed

# Global optimized processor instance
optimized_processor = None

def init_optimized_processor(device='cpu'):
    """Initialize the optimized frame processor"""
    global optimized_processor
    optimized_processor = OptimizedFrameProcessor(device)

def benchmark_frame_processing(frame, iterations=10):
    """Benchmark frame processing performance with and without optimization"""
    import time
    global optimized_processor
    
    if optimized_processor is None:
        logging.warning("Optimized processor not initialized for benchmarking")
        return {}
    
    results = {
        'original_yolo_times': [],
        'original_depth_times': [],
        'optimized_unified_times': [],
        'optimized_parallel_times': [],
        'cache_hit_times': []
    }
    
    # Benchmark original processing
    for _ in range(iterations):
        # Original YOLO processing
        start = time.time()
        yolo_frame = cv2.resize(frame, (YOLO_IMG_SIZE, int(frame.shape[0] * YOLO_IMG_SIZE / frame.shape[1])))
        results['original_yolo_times'].append(time.time() - start)
        
        # Original depth processing
        start = time.time()
        depth_w = int(getattr(state, 'dynamic_depth_resolution', config["depth"]["resolution"]))
        small = cv2.resize(frame, (depth_w, int(frame.shape[0] * depth_w / frame.shape[1])))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results['original_depth_times'].append(time.time() - start)
    
    # Benchmark optimized unified processing
    for _ in range(iterations):
        start = time.time()
        optimized_processor.process_frame_unified(frame, yolo_size=YOLO_IMG_SIZE)
        results['optimized_unified_times'].append(time.time() - start)
    
    # Benchmark optimized parallel processing
    for _ in range(iterations):
        start = time.time()
        optimized_processor.process_frame_parallel(frame, yolo_size=YOLO_IMG_SIZE)
        results['optimized_parallel_times'].append(time.time() - start)
    
    # Benchmark cache hits (should be very fast)
    optimized_processor.process_frame_unified(frame, yolo_size=YOLO_IMG_SIZE)  # Prime cache
    for _ in range(iterations):
        start = time.time()
        optimized_processor.process_frame_unified(frame, yolo_size=YOLO_IMG_SIZE)
        results['cache_hit_times'].append(time.time() - start)
    
    # Calculate averages
    for key in results:
        if results[key]:
            avg = sum(results[key]) / len(results[key])
            results[f'{key}_avg'] = avg
    
    return results

def log_performance_improvements():
    """Log performance improvements from optimization"""
    try:
        if not hasattr(state, 'performance_stats'):
            return
        
        stats = state.performance_stats
        
        # Calculate processing efficiency
        yolo_times = stats.get('yolo_inference_times', [])
        depth_times = stats.get('depth_inference_times', [])
        post_times = stats.get('postprocess_depth_times', [])
        smooth_times = stats.get('temporal_smooth_times', [])
        preproc_times = stats.get('yolo_preproc_times', [])
        roi_times = stats.get('yolo_roi_times', [])
        ultra_times = stats.get('yolo_ultra_times', [])
        coreml_times = stats.get('yolo_coreml_times', [])
        
        if yolo_times and depth_times:
            avg_yolo = sum(yolo_times[-100:]) / min(len(yolo_times), 100)  # Last 100 samples
            avg_depth = sum(depth_times[-100:]) / min(len(depth_times), 100)
            avg_post = (sum(post_times[-100:]) / min(len(post_times), 100)) if post_times else 0.0
            avg_smooth = (sum(smooth_times[-100:]) / min(len(smooth_times), 100)) if smooth_times else 0.0
            avg_preproc = (sum(preproc_times[-100:]) / min(len(preproc_times), 100)) if preproc_times else 0.0
            avg_roi = (sum(roi_times[-100:]) / min(len(roi_times), 100)) if roi_times else 0.0
            avg_ultra = (sum(ultra_times[-100:]) / min(len(ultra_times), 100)) if ultra_times else 0.0
            avg_coreml = (sum(coreml_times[-100:]) / min(len(coreml_times), 100)) if coreml_times else 0.0

            logging.info(
                f"Perf: YOLO {avg_yolo:.3f}s (pre {avg_preproc:.3f}s, ROI {avg_roi:.3f}s), "
                f"Depth {avg_depth:.3f}s (post {avg_post:.3f}s, smooth {avg_smooth:.3f}s), "
                f"Backends: Ultra {avg_ultra:.3f}s/{stats.get('yolo_ultra_calls',0)}, CoreML {avg_coreml:.3f}s/{stats.get('yolo_coreml_calls',0)}"
            )
            
            # Log cache efficiency if available
            if optimized_processor:
                yolo_cache_size = len(optimized_processor.yolo_cache)
                depth_cache_size = len(optimized_processor.depth_cache)
                logging.info(f"Cache efficiency - YOLO cache: {yolo_cache_size}, Depth cache: {depth_cache_size}")
        # Camera capture and read failures
        cam_times = stats.get('camera_capture_times', [])
        if cam_times:
            avg_cam = sum(cam_times[-100:]) / min(len(cam_times), 100)
            logging.info(f"Camera: capture {avg_cam*1000:.1f}ms avg, read failures {stats.get('camera_read_failures', 0)}")
    
    except Exception as e:
        logging.debug(f"Performance logging error: {e}")


class State:
    """Mutable runtime state shared across threads.

    Holds thread handles, timing, configuration-derived flags, performance stats,
    and recent observations. This object is passed around implicitly (module-
    level `state`) and guarded by `state.lock` when updating multi-field values.

    Notable attributes:
        threads: Dict of thread name → `threading.Thread`.
        error_counts: Dict of tag → int counter, incremented by `log_error()`.
        _last_error_logs: Dict of tag → timestamp, for rate-limited logging.
        last_yolo_time, _last_depth_time, _last_frame_time: liveness timestamps.
        performance_stats: Dict tracking FPS, inference times, skips, failures.
        telemetry: Dict of arrays for event timing and usage metrics.
    """
    def __init__(self):
        self.running = True
        self.ready = False
        self.frame = None
        self.depth_map_m = None
        self.depth_vis = None
        self.detections = []
        self.show_depth = False
        self.show_overlay = True
        self.show_edges = False
        self.last_fps = 0.0
        self.depth_ready = False
        self.lock = threading.RLock()
        # Depth calibration params
        try:
            self.min_m = float(config.get('depth', {}).get('min_m', 0.2))
        except Exception:
            self.min_m = 0.2
        try:
            self.max_m = float(config.get('depth', {}).get('max_m', 20.0))
        except Exception:
            self.max_m = 20.0
        try:
            self.depth_calibration_factor_m_per_255 = float(config.get('depth', {}).get('calibration_factor_m_per_255', 0.0))
        except Exception:
            self.depth_calibration_factor_m_per_255 = 0.0
        self.depth_alpha = DEPTH_ALPHA
        # Announcer / voice
        self.enable_announce = True
        self.last_announce_time = 0.0
        # Initialize cooldown from config to allow 1-second cadence
        try:
            self.announce_cooldown_s = float(config.get('announcements', {}).get('cooldown', 3.0))
        except Exception:
            self.announce_cooldown_s = 3.0
        self.enable_voice_listen = False
        self.command_queue = queue.Queue(maxsize=4)
        self.degraded_mode = False
        self.mock_sensor = bool(MOCK_SENSOR)
        # Performance knobs
        self.fast_depth = FAST_DEPTH
        self.depth_min_interval = DEPTH_MIN_INTERVAL
        self._last_depth_time = 0.0
        # Last successful inference timestamps (updated only on valid outputs)
        self.last_success_yolo_ts = 0.0
        self.last_success_depth_ts = 0.0
        # IMU fusion thresholds
        try:
            self.movement_threshold = float(config.get('announcements', {}).get('movement_threshold', 0.1))
        except Exception:
            self.movement_threshold = 0.1
        # IMU state
        self.imu = {
            'user_acceleration': [0.0, 0.0, 0.0],
            'rotation_rate': [0.0, 0.0, 0.0],
            'last_update': 0.0,
        }
        # Speed tracking for fast objects
        self.object_tracks = {}  # track_id -> list of (time, depth, center)
        self.track_id_counter = 0
        self.speed_cooldowns = {}  # track_id -> last_announce_time
        # Kalman filter states per track (1D depth smoothing)
        self.kalman_states = {}
        # Alpha-Beta filter states per track (position+velocity)
        self.alpha_beta_states = {}
        # Per-track bounding box area history for approach corroboration
        self.track_areas = {}
        # Re-identification and encounter memory
        self.appearance_gallery = {}  # track_id -> {'desc': np.ndarray, 'last_seen': float, 'last_center': (x,y)}
        self.encounter_memory = {}    # track_id -> {'last_pass': float}
        # Context modes: None | 'silent_crowd' | 'indoor_calm'
        self.context_mode = None
        # AI announcer and navigation
        self.ai_decider = True
        self.context_aware = True
        self.min_importance_score = 0.6
        self.silence_threshold = 5.0
        self.movement_threshold = 0.1
        # Real-time announcer mode: lower gating and prioritize immediate hazards
        try:
            self.realtime_mode = bool(config.get('announcements', {}).get('realtime_mode', False))
        except Exception:
            self.realtime_mode = False
        
        self.class_prefs = {
            'person': {'priority': 3, 'max_m': 6.0, 'phrase': 'Person', 'urgency_multiplier': 1.5},
            'bicycle': {'priority': 2, 'max_m': 6.0, 'phrase': 'Bicycle', 'urgency_multiplier': 1.3},
            'motorbike': {'priority': 2, 'max_m': 8.0, 'phrase': 'Motorbike', 'urgency_multiplier': 1.4},
            'car': {'priority': 2, 'max_m': 12.0, 'phrase': 'Car', 'urgency_multiplier': 1.2},
            'bus': {'priority': 2, 'max_m': 15.0, 'phrase': 'Bus', 'urgency_multiplier': 1.1},
            'truck': {'priority': 2, 'max_m': 15.0, 'phrase': 'Truck', 'urgency_multiplier': 1.1},
            'chair': {'priority': 2, 'max_m': 6.0, 'phrase': 'Chair', 'urgency_multiplier': 1.0},
            'bed': {'priority': 2, 'max_m': 6.0, 'phrase': 'Bed', 'urgency_multiplier': 0.9},
            'traffic light': {'priority': 1, 'max_m': 20.0, 'phrase': 'Traffic light', 'urgency_multiplier': 0.8},
            'stop sign': {'priority': 1, 'max_m': 20.0, 'phrase': 'Stop sign', 'urgency_multiplier': 0.9},
        }
        try:
            self.min_announce_conf = float(config.get('announcements', {}).get('min_conf', 0.25))
        except Exception:
            self.min_announce_conf = 0.25
        self.provider = 'apple'  # 'apple' or 'google'
        # Navigation preferences
        self.travel_mode = 'WALKING'  # 'WALKING' or 'DRIVING'
        self._last_class_announce = {}
        self.class_cooldown_s = 4.0
        
        # Context awareness
        self.last_announcement_time = 0.0
        self.announcement_history = []
        self.user_movement_history = []
        self.environment_context = {
            'last_major_change': 0.0,
            'scene_stability': 1.0,
            'noise_level': 0.0,
            'roi_balance': {'left': 0.0, 'right': 0.0},
            'safe_corridor': {'side': None, 'width': 0.0},
            'scene_tags': [],
            'clear_path': {'persist': False, 'corridor_frac': 0.0, 'last_true_since': 0.0}
        }

        # Positioning state (GPS outdoors, BLE/Wi-Fi/CV indoors)
        try:
            pos_cfg = config.get('positioning', {}) or {}
        except Exception:
            pos_cfg = {}
        # Config knobs with safe defaults
        self.positioning_config = {
            'gps_accuracy_threshold_m': float(pos_cfg.get('gps_accuracy_threshold_m', 30.0)),
            'gps_stale_s': float(pos_cfg.get('gps_stale_s', 5.0)),
            'prefer_ble_over_wifi': bool(pos_cfg.get('prefer_ble_over_wifi', True)),
            'cv_enabled': bool(pos_cfg.get('cv_enabled', True)),
            'pixel_to_meter_scale': float(pos_cfg.get('pixel_to_meter_scale', 1.0)),
            'cv_feature_count': int(pos_cfg.get('cv_feature_count', 200)),
            'cv_min_quality': float(pos_cfg.get('cv_min_quality', 0.01)),
            'cv_min_points': int(pos_cfg.get('cv_min_points', 40))
        }
        # Live positioning snapshot
        self.positioning = {
            'provider': None,           # 'gps' | 'ble' | 'wifi' | 'cv'
            'lat': None, 'lon': None,   # Outdoors
            'x': 0.0, 'y': 0.0,         # Indoors local frame
            'indoor': False,
            'accuracy_m': None,
            'last_update_ts': 0.0
        }
        # Internal buffers for CV odometry
        self._cv_prev_gray = None
        self._cv_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self._last_gps_update_ts = 0.0
        self._gps_supported = False
        self._ble_wifi_supported = False
        # External positioning updates (for future beacon/Wi-Fi injectors)
        self._position_updates = queue.Queue(maxsize=32)

        # Global announcement cooldown trackers
        self._last_summary = 0.0
        self._last_clearance = 0.0
        self._last_resolution = 0.0
        self._last_bearing_prompt = 0.0
        self._last_mode_prompt = 0.0
        self._last_warning_time = 0.0
        self._last_closing_fast = 0.0
        self._global_silence_until = 0.0
        self._last_variant_spoken = {}

        # Event-edge announcement state (per-track)
        self._track_first_seen = {}
        self._track_last_zone = {}
        self._track_last_center_in = {}
        self._track_last_behavior = {}
        self._track_last_event_ts = {}

        # Hazard boost and looming metrics
        self.hazard_boost_active_until = 0.0
        self._yolo_img_override = None
        self.center_depth_m = None
        self.center_approach_mps = 0.0
        self._last_center_depth = None
        self._last_center_depth_time = 0.0
        self._last_looming_alert = 0.0
        # Persistence trackers to avoid false positives
        self._center_speed_above_since = 0.0
        self.per_track_last_announce = {}
        
        # Performance monitoring
        self.performance_stats = {
            'fps_history': [],
            'processing_times': [],
            'cpu_usage': [],
            'memory_usage': [],
            'quality_level': 1.0,
            'last_fps_update': time.time(),
            'frame_count': 0,
            'capture_count': 0,
            'total_processing_time': 0.0,
            # Latency arrays (rolling)
            'yolo_inference_times': [],
            'yolo_preproc_times': [],
            'yolo_roi_times': [],
            'yolo_coreml_times': [],
            'yolo_ultra_times': [],
            'depth_inference_times': [],
            'postprocess_depth_times': [],
            'temporal_smooth_times': [],
            'depth_total_times': [],
            'camera_capture_times': [],
            'camera_preproc_times': [],
            'tts_queue_latencies': [],
            'tts_speak_latencies': [],
            # Profiling buckets
            'profile': {},
            # GPU metrics (CUDA only; arrays kept short)
            'gpu_mem_alloc_mb': [],
            'gpu_mem_reserved_mb': [],
            # Skip counters
            'camera_skips': 0,
            'yolo_skips': 0,
            'depth_skips': 0,
            # Error counters
            'camera_read_failures': 0,
            # Inference backend counters
            'yolo_coreml_calls': 0,
            'yolo_ultra_calls': 0
        }
        # Standardized error tracking
        self.error_counts = {}
        self._last_error_logs = {}
        # Telemetry counters and rolling windows
        self.telemetry = {
            'announce_ts': deque(maxlen=5000),
            'critical_ts': deque(maxlen=5000),
            'suppressed_ts': deque(maxlen=5000),
            'summary_ts': deque(maxlen=5000),
            'start_time': time.time(),
            'degraded_mode': False,
            'mock_sensor': False,
            # Per-class timestamps: class label -> deque of announce times
            'per_class_ts': {},
            # Approach-tier counters
            'moderate_approach_ts': deque(maxlen=5000),
            'slow_approach_ts': deque(maxlen=5000),
            'approach_repeat_suppressed_ts': deque(maxlen=5000),
            'moderate_left_ts': deque(maxlen=5000),
            'moderate_right_ts': deque(maxlen=5000),
            'slow_left_ts': deque(maxlen=5000),
            'slow_right_ts': deque(maxlen=5000),
            # Recent events (lightweight records of last announcements)
            'events': deque(maxlen=200),
            # Voice command telemetry
            'voice_received_ts': deque(maxlen=5000),
            'voice_handled_ts': deque(maxlen=5000),
            'voice_unknown_ts': deque(maxlen=5000),
            # User feedback telemetry
            'feedback_ts': deque(maxlen=5000),
            'feedback_type_counts': {},  # type -> deque timestamps
            # Event type counters
            'event_type_counts': {
                'system': deque(maxlen=5000),
                'critical': deque(maxlen=5000),
                'warning': deque(maxlen=5000),
                'clear': deque(maxlen=5000),
                'summary': deque(maxlen=5000)
            },
            # Importance level counters (0.0-1.0 ranges)
            'importance_counts': {
                'low': deque(maxlen=5000),      # 0.0-0.3
                'medium': deque(maxlen=5000),   # 0.3-0.7
                'high': deque(maxlen=5000)      # 0.7-1.0
            },
            # Queue latency histograms (in milliseconds)
            'queue_latencies': {
                'tts_queue': deque(maxlen=1000),
                'tts_speak': deque(maxlen=1000),
                'audio_queue': deque(maxlen=1000)
            },
            # Per-class history for trendlines (last 50 data points)
            'per_class_history': {}
        }
        # Recent announcements feed for external clients (phone polling)
        try:
            from collections import deque as _dq
            self._announce_feed = _dq(maxlen=100)
        except Exception:
            self._announce_feed = []
        # Metrics time window (seconds) for windowed counts in /metrics
        try:
            env_win = os.environ.get('THIRD_EYE_METRICS_WINDOW_SECONDS')
            cfg_win = config.get('logging', {}).get('metrics_window_seconds', 900.0)
            self.metrics_window_seconds = float(env_win) if (env_win is not None and env_win != '') else float(cfg_win)
        except Exception:
            self.metrics_window_seconds = 900.0
        # IMU state (from phone/device via local network)
        self.imu = {
            'user_acceleration': [0.0, 0.0, 0.0],
            'rotation_rate': [0.0, 0.0, 0.0],
            'last_update': 0.0
        }
        self.executor = ThreadPoolExecutor(max_workers=config["performance"]["max_workers"])
        
        # Tracker for smooth detection
        self.tracker_detections = []
        self.last_yolo_time = 0.0
        try:
            # Initialize to hazard baseline for higher cadence
            self.yolo_interval = float(config.get('hazard', {}).get('yolo_interval_s', 0.05))
        except Exception:
            self.yolo_interval = 0.05
        self.ema_boxes = {}  # EMA smoothing for bounding boxes
        
        # Rendering limits
        self.top_k_draw = 5

        # ROI-first detection settings (default to full-frame)
        self.roi_enabled = False
        self.roi_ratio = 1.0  # central crop ratio (0-1)
        self.roi_full_every_n = 1  # run full-frame every N YOLO runs

        # Debug metadata for letterbox/shape overlay
        self.debug_meta = None
        
        # Frame rate control
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.target_frame_time = 1.0 / TARGET_FPS
        self.frame_skip_counter = 0
        self.depth_skip_counter = 0
        self.yolo_skip_counter = 0

        # Core ML YOLO and preprocessing buffers
        self.yolo_coreml = None
        self.use_preproc_queue = True
        try:
            buf_size = int(config.get("performance", {}).get("preproc_buffer_size", 6))
        except Exception:
            buf_size = 6
        self.preproc_queue = queue.Queue(maxsize=buf_size)
        # Dynamic depth resolution control
        try:
            self.dynamic_depth_resolution = int(config.get("depth", {}).get("resolution", 384))
        except Exception:
            self.dynamic_depth_resolution = 384
        # External (network) RGB+depth ingress buffers (e.g., iPhone LiDAR)
        self._external_rgb = None
        self._external_rgb_time = 0.0
        self._external_depth_m = None
        self._external_depth_time = 0.0
        try:
            self._use_network_depth = (str(config.get('depth', {}).get('source', 'model')) == 'network')
        except Exception:
            self._use_network_depth = False
        # Cached display JPEG for web preview
        self._display_jpeg = None
        self._display_jpeg_time = 0.0
        # Bypass frame skip during criticals
        self._avoid_frame_skip_until = 0.0

        # Watchdog and thread management
        self.threads = {}
        self.restart_request = { 'yolo': False, 'depth': False, 'camera': False }
        self._last_restart = {}
        self._yolo_tuple = None
        self._depth_pipe = None

        # Audio backend defaults
        self._speech_backend = 'say'
        self._av_synth = None
        self._speech_rate = 0.5
        self._speech_volume = float(config.get('audio', {}).get('speech_volume', 1.0))
        self._earcon_volume = float(config.get('audio', {}).get('earcon_volume', 0.8))
        self._earcons_enabled = bool(config.get('audio', {}).get('earcons_enabled', True))

        # Optional async pipeline (producer/consumer) for concurrency and synchronization
        try:
            self.pipelines_async_enabled = bool(config.get('pipelines', {}).get('async_enabled', False))
        except Exception:
            self.pipelines_async_enabled = False
        if self.pipelines_async_enabled:
            # Small bounded queues; drop oldest on overflow for freshness
            try:
                fqsz = int(config.get('pipelines', {}).get('frame_queue_size', 4))
            except Exception:
                fqsz = 4
            try:
                dqsz = int(config.get('pipelines', {}).get('detection_queue_size', 6))
            except Exception:
                dqsz = 6
            try:
                dmsz = int(config.get('pipelines', {}).get('depth_queue_size', 6))
            except Exception:
                dmsz = 6
            self.frame_queue_yolo = queue.Queue(maxsize=fqsz)
            self.frame_queue_depth = queue.Queue(maxsize=fqsz)
            self.detection_queue = queue.Queue(maxsize=dqsz)
            self.depth_map_queue = queue.Queue(maxsize=dmsz)


state = State()


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# Lightweight wrapper so runtime can consume local DepthAnythingV2 like HF pipeline
class LocalDepthWrapper:
    def __init__(self, model, base_resolution=384):
        self.model = model
        try:
            self.model.eval()
        except Exception:
            pass
        self.base_resolution = int(base_resolution)

    def __call__(self, pil_image):
        try:
            arr = np.array(pil_image)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            depth_np = self.model.infer_image(bgr, input_size=self.base_resolution)
            return {"depth": depth_np}
        except Exception as e:
            logging.error(f"LocalDepthWrapper inference failed: {e}")
            # Return a dummy depth to avoid hard crash; upstream will treat it as flat
            h, w = pil_image.height, pil_image.width
            return {"depth": np.zeros((h, w), dtype=np.float32)}


def load_models():
    device = select_device()
    logging.info(f"Loading YOLO models on {device}...")

    # Record device info for health reporting
    try:
        info = {
            'device_type': device.type,
            'torch_version': str(getattr(torch, '__version__', 'unknown'))
        }
        if device.type == 'cuda':
            try:
                info['cuda_name'] = str(torch.cuda.get_device_name(0))
                major, minor = torch.cuda.get_device_capability(0)
                info['cuda_capability'] = f"{major}.{minor}"
            except Exception:
                pass
        elif device.type == 'mps':
            info['mps'] = True
        with state.lock:
            state.device_info = info
    except Exception:
        pass

    # Load primary YOLO model
    yolo = YOLO(YOLO_WEIGHTS)
    yolo.fuse()
    yolo.model.eval()
    if device.type == 'cuda' and hasattr(yolo.model, 'half'):
        yolo.model.half()

    # Optional: Load Core ML YOLO to run on ANE
    try:
        if coreml_is_enabled(config):
            # Pre-check: coremltools availability and Python version support
            try:
                import importlib.util as _ilu
                spec = _ilu.find_spec('coremltools')
            except Exception:
                spec = None
            py_ok = (sys.version_info.major == 3 and sys.version_info.minor <= 11)
            if spec is None or not py_ok:
                reason = 'missing coremltools' if spec is None else f'Python {sys.version_info.major}.{sys.version_info.minor} not supported'
                logging.info(f"Skipping Core ML YOLO ({reason}); using Ultralytics YOLO")
            else:
                import coremltools as ct
                mlp = str(config.get('yolo', {}).get('coreml_package_path', 'models/yolo11n.mlpackage'))
                if os.path.exists(mlp):
                    state.yolo_coreml = ct.models.MLModel(mlp)
                    logging.info(f"✓ Core ML YOLO loaded: {mlp}")
                else:
                    logging.warning(f"⚠ Core ML package not found at {mlp}; using Ultralytics YOLO")
        else:
            logging.info("Core ML disabled by config; using Ultralytics .pt exclusively")
    except Exception as e:
        logging.warning(f"Core ML YOLO load failed; falling back. Reason: {e}")

    # Load ensemble models for better detection
    ensemble_models = []
    if config["yolo"]["ensemble"]:
        try:
            # Load additional models for ensemble
            yolo_n = YOLO('yolov8n.pt')  # Fast model
            yolo_x = YOLO('yolov8x.pt')  # Most accurate model
            ensemble_models = [yolo_n, yolo_x]
            logging.info("✓ Ensemble models loaded")
        except Exception as e:
            logging.warning(f"⚠ Ensemble models not available: {e}")

    # Depth model: respect config; disable entirely when not enabled
    depth_pipe = None
    try:
        depth_enabled = bool(config.get('depth', {}).get('enabled', True))
    except Exception:
        depth_enabled = True
    if not depth_enabled:
        with state.lock:
            state.depth_ready = False
            try:
                state._use_network_depth = False
            except Exception:
                pass
        logging.info("Depth disabled by config; skipping depth model and thread")
        return yolo, device, depth_pipe, ensemble_models
    # If using network (external LiDAR) as depth source, skip model loading
    try:
        use_network_depth = (str(config.get('depth', {}).get('source', 'model')) == 'network')
    except Exception:
        use_network_depth = False
    if use_network_depth:
        with state.lock:
            state.depth_ready = True
            try:
                state._use_network_depth = True
            except Exception:
                pass
        logging.info("✓ Depth source set to 'network'; skipping local/HF depth model load")
    else:
        use_local = bool(config.get('depth', {}).get('use_local_v2', False))
        local_ok = (DepthAnythingV2 is not None)
        if use_local and local_ok:
            weights = str(config.get('depth', {}).get('local_weights', '') or '')
            encoder = str(config.get('depth', {}).get('local_encoder', 'vits'))
            if weights and os.path.exists(weights):
                try:
                    model = DepthAnythingV2(encoder=encoder)
                    sd = torch.load(weights, map_location='cpu')
                    model.load_state_dict(sd)
                    depth_pipe = LocalDepthWrapper(model, base_resolution=int(config.get('depth', {}).get('resolution', 384)))
                    logging.info(f"✓ Using local DepthAnythingV2 (encoder={encoder}) with weights: {weights}")
                except Exception as e:
                    logging.warning(f"⚠ Failed to load local DepthAnythingV2, falling back to HF pipeline: {e}")
            else:
                logging.warning("⚠ Local DepthAnythingV2 requested but weights not provided/found; using HF pipeline")

    if (not use_network_depth) and (depth_pipe is None):
        depth_device_id = 0 if device.type == 'cuda' else (-1)
        try:
            # Prefer fast image processor to avoid slow-processor warning
            image_processor = None
            try:
                image_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, use_fast=True)
            except Exception as ip_e:
                logging.info(f"Using default image processor (fast unavailable): {ip_e}")

            pipe_kwargs = {
                'model': DEPTH_MODEL,
                'device': depth_device_id,
                'dtype': torch.float16 if device.type == 'cuda' else torch.float32,
            }
            if image_processor is not None:
                pipe_kwargs['image_processor'] = image_processor

            depth_pipe = pipeline('depth-estimation', **pipe_kwargs)
            # Optional: quantize depth model on CPU using torchao when available; fallback to torch.ao
            try:
                if DEPTH_QUANTIZE:
                    # Quantization is only applicable when depth runs on CPU
                    if depth_device_id != -1:
                        logging.info("Skipping int8 quantization: only supported on CPU; using float16/float32")
                    elif hasattr(depth_pipe, 'model') and isinstance(depth_pipe.model, torch.nn.Module):
                        try:
                            # Prefer torchao dynamic int8 for Linear layers
                            from importlib import import_module
                            torchao_q = import_module('torchao.quantization')
                            quantize = getattr(torchao_q, 'quantize', None)
                            linear_dynamic_int8 = getattr(torchao_q, 'linear_dynamic_int8', None)
                            if callable(quantize) and callable(linear_dynamic_int8):
                                depth_pipe.model = quantize(depth_pipe.model, linear_dynamic_int8())
                                logging.info("✓ Applied torchao linear dynamic int8 quantization to depth model")
                            else:
                                raise ImportError("torchao.quantization APIs not found")
                        except Exception as taq_e:
                            # Silence noisy warning when torchao is not installed; try torch.ao quietly
                            logging.debug(f"torchao quantization unavailable; attempting torch.ao fallback: {taq_e}")
                            try:
                                import warnings
                                with warnings.catch_warnings():
                                    warnings.filterwarnings(
                                        "ignore",
                                        category=DeprecationWarning,
                                        message=r".*torch\.ao\.quantization.*",
                                    )
                                    import torch.ao.quantization as tq
                                # Check supported quantized engines before attempting fallback
                                engines = list(getattr(torch.backends.quantized, 'supported_engines', []))
                                if not engines:
                                    logging.debug("Skipping int8 quantization: no quantized engine available in this build")
                                else:
                                    if 'qnnpack' in engines:
                                        torch.backends.quantized.engine = 'qnnpack'
                                    elif 'fbgemm' in engines:
                                        torch.backends.quantized.engine = 'fbgemm'
                                    else:
                                        logging.debug(f"Skipping int8 quantization: unsupported engine(s) {engines}")
                                    if getattr(torch.backends.quantized, 'engine', 'none') != 'none':
                                        depth_pipe.model = tq.quantize_dynamic(depth_pipe.model, {torch.nn.Linear}, dtype=torch.qint8)
                                        logging.debug("Applied torch.ao dynamic int8 quantization to depth model (Linear parts)")
                            except Exception as qe:
                                logging.debug(f"Depth quantization fallback failed or not applicable: {qe}")
            except Exception as qe_outer:
                logging.debug(f"Depth quantization attempt error: {qe_outer}")
        except Exception as e:
            logging.error(f"Depth estimation pipeline/model error: {e}\n" +
                          "Troubleshooting:\n" +
                          "- Ensure you have internet access for model download.\n" +
                          "- Check that the model name is correct and available on Hugging Face.\n" +
                          "- Install required libraries: pip install diffusers accelerate\n" +
                          "- Try running on CPU if using MPS/GPU.\n" +
                          "- Test with: from transformers import pipeline; pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf')\n")
            raise SystemExit(1)

    state.depth_ready = True
    return yolo, device, depth_pipe, ensemble_models


def temporal_smooth(prev_map, cur_map, alpha=DEPTH_ALPHA):
    if prev_map is None:
        return cur_map
    # Enhanced temporal smoothing for better stability
    smoothed_map = alpha * cur_map + (1.0 - alpha) * prev_map
    # Additional median filtering to reduce noise spikes
    return cv2.medianBlur(smoothed_map.astype(np.float32), 3).astype(np.float32)

def _compute_brightness(frame_bgr):
    try:
        if frame_bgr is None:
            return 0.0
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    except Exception:
        return 0.0

def _priority_zone_means(depth_m):
    try:
        h, w = depth_m.shape
        third = max(1, w // 3)
        left = depth_m[:, :third]
        center = depth_m[:, third:2*third]
        right = depth_m[:, 2*third:]
        def _median_safe(patch):
            valid = patch[np.isfinite(patch)]
            return float(np.median(valid)) if valid.size > 0 else None
        return _median_safe(left), _median_safe(center), _median_safe(right)
    except Exception:
        return None, None, None

def _update_histories(state, center_depth, global_avg):
    now = time.time()
    if not hasattr(state, 'center_depth_history'):
        from collections import deque
        state.center_depth_history = deque(maxlen=120)
    if not hasattr(state, 'global_depth_history'):
        from collections import deque
        state.global_depth_history = deque(maxlen=240)
    if center_depth is not None and np.isfinite(center_depth):
        state.center_depth_history.append((now, float(center_depth)))
    if global_avg is not None and np.isfinite(global_avg):
        state.global_depth_history.append((now, float(global_avg)))

def _analyze_approach_and_clear(state, frame_bgr, depth_m):
    """Return approach flag/direction and path clear based on recent history.

    - Approach: if mean center depth drops >25% over last 5 samples.
    - Path clear: if global avg depth > 4m for last 2s.
    """
    approach_detected = False
    approach_direction = None
    estimated_distance = None
    path_clear = False

    # Compute zone medians and global average
    try:
        left_m, center_m, right_m = _priority_zone_means(depth_m)
        global_avg = float(np.nanmean(depth_m)) if np.isfinite(depth_m).any() else None
    except Exception:
        left_m = center_m = right_m = global_avg = None

    # Update histories
    _update_histories(state, center_m, global_avg)

    # Determine approach using center history
    try:
        recent = [d for _, d in getattr(state, 'center_depth_history', [])][-5:]
        prev5 = [d for _, d in getattr(state, 'center_depth_history', [])][-10:-5]
        if len(recent) == 5 and len(prev5) == 5:
            mean_recent = float(np.mean(recent))
            mean_prev = float(np.mean(prev5))
            # Adaptive sensitivity: raise threshold under low light
            brightness = _compute_brightness(frame_bgr)
            thr = 0.25
            if brightness < 60.0:
                thr = 0.30
            drop_frac = (mean_prev - mean_recent) / max(1e-6, mean_prev)
            approach_detected = drop_frac > thr
            estimated_distance = mean_recent if approach_detected else center_m
    except Exception:
        pass

    # Direction: where the largest relative drop occurred among zones
    try:
        # Compare current medians to previous medians over a short window
        h, w = depth_m.shape
        third = max(1, w // 3)
        # Previous window: use the depth from one second ago if available
        prev_center = None
        prev_left = None
        prev_right = None
        # Derive approximate previous medians from history using stored center only for center; left/right use current as fallback
        if getattr(state, 'center_depth_history', None):
            prev_vals = [d for t, d in state.center_depth_history if (time.time() - t) >= 0.8]
            if prev_vals:
                prev_center = float(prev_vals[-1])
        # Fallbacks
        prev_left = left_m
        prev_right = right_m
        def rel_drop(prev, cur):
            if prev is None or cur is None or prev <= 0:
                return 0.0
            return max(0.0, (prev - cur) / prev)
        drops = {
            'left': rel_drop(prev_left, left_m),
            'center': rel_drop(prev_center, center_m),
            'right': rel_drop(prev_right, right_m)
        }
        approach_direction = max(drops, key=drops.get) if approach_detected else None
    except Exception:
        pass

    # Path clear: global avg depth > 4m for last 2s
    try:
        now = time.time()
        window = [d for t, d in getattr(state, 'global_depth_history', []) if (now - t) <= 2.0]
        if window and (min(window) > 4.0):
            path_clear = True
    except Exception:
        path_clear = False

    return {
        'approach_detected': bool(approach_detected),
        'approach_direction': approach_direction,
        'estimated_distance': float(estimated_distance) if (estimated_distance is not None) else None,
        'path_clear': bool(path_clear)
    }


def remove_outliers_percentile(depth_np, low_p=5, high_p=95):
    valid = depth_np[np.isfinite(depth_np)]
    if valid.size == 0:
        return depth_np
    lo = np.percentile(valid, low_p)
    hi = np.percentile(valid, high_p)
    return np.clip(depth_np, lo, hi)


def guided_filter(guide_gray, src_float, radius=8, eps=1e-2):
    guide = guide_gray.astype(np.float32)
    src = src_float.astype(np.float32)
    k = (radius, radius)
    mean_g = cv2.boxFilter(guide, -1, k)
    mean_s = cv2.boxFilter(src, -1, k)
    mean_gs = cv2.boxFilter(guide * src, -1, k)
    cov_gs = mean_gs - mean_g * mean_s
    var_g = cv2.boxFilter(guide * guide, -1, k) - mean_g * mean_g
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    mean_a = cv2.boxFilter(a, -1, k)
    mean_b = cv2.boxFilter(b, -1, k)
    return mean_a * guide + mean_b


def postprocess_depth(depth_np, orig_shape, rgb_frame=None):
    """
    Post-process raw depth output into a colorized visualization and a meters map.

    This function is designed to be resilient to malformed inputs from upstream
    models or external sources. It sanitizes the input array, clips outliers,
    normalizes depth for visualization, and converts the guided depth to meters.

    Args:
        depth_np: Raw depth array from the model or external source. Can be HxW
            or HxWx1 and may contain NaNs/Infs.
        orig_shape: Original frame shape (H, W, C) used to size outputs.
        rgb_frame: Optional RGB frame used for guided filtering. When provided,
            the function enhances edge preservation around image structures.

    Returns:
        depth_vis: Colorized visualization image (`uint8`, shape HxWx3).
        depth_m: Depth in meters (`float32`, shape HxW), smoothed and resized
            to match the original frame.

    Notes:
        - NaNs and Infs are replaced with the median of valid values, or zeros
          when the array is wholly invalid.
        - All OpenCV operations are guarded; failures fall back to safe defaults.
        - Guided filtering and smoothing automatically adapt to `state.fast_depth`.
    """
    h, w = orig_shape[:2]
    try:
        # Ensure numeric array
        depth_np = np.asarray(depth_np)
        if depth_np.ndim == 3:
            depth_np = depth_np.squeeze()
    except Exception:
        # Fallback to zeros on any shape/type error
        depth_np = np.zeros((h, w), dtype=np.float32)

    # Replace NaNs/Infs defensively
    try:
        if not np.isfinite(depth_np).any():
            depth_np = np.zeros((h, w), dtype=np.float32)
        else:
            # Use nan_to_num with safe defaults
            med = float(np.nanmedian(depth_np)) if np.isfinite(depth_np).any() else 0.0
            depth_np = np.nan_to_num(depth_np, nan=med, posinf=med, neginf=med).astype(np.float32)
    except Exception:
        depth_np = np.zeros((h, w), dtype=np.float32)

    # Fast outlier clipping for real-time performance
    try:
        depth_np = remove_outliers_percentile(depth_np, 10, 90)  # Less aggressive clipping
    except Exception:
        pass

    # Decide visualization and mapping source: normalized vs. raw grayscale for calibration
    use_linear_calib = False
    try:
        use_linear_calib = float(getattr(state, 'depth_calibration_factor_m_per_255', 0.0) or 0.0) > 0.0
    except Exception:
        use_linear_calib = False

    if use_linear_calib:
        # Use raw grayscale values (expected 0..255) for visualization and mapping
        try:
            raw_u8 = np.clip(depth_np, 0, 255).astype(np.uint8)
        except Exception:
            # Fallback to normalized if clipping fails
            try:
                dmin, dmax = float(depth_np.min()), float(depth_np.max())
                if dmax > dmin:
                    raw_u8 = (((depth_np - dmin) / (dmax - dmin)) * 255.0).astype(np.uint8)
                else:
                    raw_u8 = np.zeros_like(depth_np, dtype=np.uint8)
            except Exception:
                raw_u8 = np.zeros_like(depth_np, dtype=np.uint8)
        # Smoothing tuned for speed
        if state.fast_depth:
            depth_guided = cv2.medianBlur(raw_u8, 3)
        else:
            depth_guided = cv2.medianBlur(cv2.bilateralFilter(raw_u8, 5, 30, 30), 3)
    else:
        # Normalize to 0..255 for visualization (relative depth)
        try:
            dmin, dmax = float(depth_np.min()), float(depth_np.max())
            if dmax > dmin:
                depth_norm_f = (depth_np - dmin) / (dmax - dmin)
            else:
                depth_norm_f = np.zeros_like(depth_np, dtype=np.float32)
        except Exception:
            depth_norm_f = np.zeros_like(depth_np, dtype=np.float32)
        
        # Enhanced smoothing for better stability
        if state.fast_depth:
            # Fast mode with better stability (smaller kernel for speed)
            depth_blur = cv2.medianBlur((depth_norm_f*255).astype(np.uint8), 3)
            depth_guided = depth_blur
        else:
            # Standard mode: higher quality smoothing (trimmed for performance)
            bilateral_d = 5
            depth_blur = cv2.bilateralFilter((depth_norm_f*255).astype(np.uint8), bilateral_d, 30, 30)
            depth_guided = cv2.medianBlur(depth_blur, 3)
    
    # Resize for visualization
    try:
        depth_vis = cv2.applyColorMap(cv2.resize(depth_guided, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLORMAP_PLASMA)
    except Exception:
        # Robust fallback visualization
        depth_vis = np.zeros((h, w, 3), dtype=np.uint8)

    # Optimized depth mapping for real-time
    try:
        if use_linear_calib:
            # Linear: meters = pixel_value * factor
            cf = float(getattr(state, 'depth_calibration_factor_m_per_255', 0.0) or 0.0)
            dm = depth_guided.astype(np.float32) * cf
        else:
            # Relative → meters using exponential mapping between [min_m, max_m]
            ratio = depth_guided.astype(np.float32) / 255.0
            min_m, max_m = state.min_m, state.max_m
            dm = min_m * np.power(max_m / min_m, 1.0 - ratio)
        depth_m = cv2.resize(dm, (w, h), interpolation=cv2.INTER_LINEAR)
        # Additional post-smoothing to reduce depth fluctuations
        depth_m = cv2.GaussianBlur(depth_m, (3, 3), 0)
    except Exception:
        # Safe default meters map
        depth_m = np.full((h, w), float(getattr(state, 'max_m', 20.0)), dtype=np.float32)

    return depth_vis, depth_m


def get_depth_at(center, depth_m):
    if depth_m is None:
        return None
    x, y = map(int, center)
    h, w = depth_m.shape
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    patch = depth_m[max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def get_bbox_median_depth(bbox, depth_m):
    if depth_m is None:
        return None
    x1, y1, x2, y2 = bbox
    h, w = depth_m.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth_m[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    valid = patch[np.isfinite(patch)]
    if valid.size == 0:
        # Near-field fallback: if box covers large area, assume very close
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        area_frac = (box_w * box_h) / float(max(1, w * h))
        if area_frac >= 0.30:
            return float(max(getattr(state, 'min_m', 0.2), 0.2))
        return None
    med = float(np.median(valid))
    # Clamp to valid depth range
    try:
        mn = float(getattr(state, 'min_m', 0.2))
        mx = float(getattr(state, 'max_m', 20.0))
    except Exception:
        mn, mx = 0.2, 20.0
    if not np.isfinite(med):
        return mn
    return float(max(mn, min(mx, med)))


def get_bbox_depth_stats(bbox, depth_m):
    """Compute robust LiDAR depth statistics inside a bbox.

    Returns a dict with:
      - median_m: median of valid depths (meters)
      - near_m: 20th percentile (closer slice) to capture nearest obstacle
      - valid_ratio: fraction of pixels with finite depth
      - valid_pixels: number of valid pixels
    """
    if depth_m is None:
        return {'median_m': None, 'near_m': None, 'valid_ratio': 0.0, 'valid_pixels': 0}
    x1, y1, x2, y2 = bbox
    h, w = depth_m.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return {'median_m': None, 'near_m': None, 'valid_ratio': 0.0, 'valid_pixels': 0}
    roi = depth_m[y1:y2, x1:x2]
    if roi.size == 0:
        return {'median_m': None, 'near_m': None, 'valid_ratio': 0.0, 'valid_pixels': 0}
    valid = roi[np.isfinite(roi) & (roi > 0)]
    total = int(roi.size)
    vp = int(valid.size)
    vr = float(vp / float(total)) if total > 0 else 0.0
    if vp == 0:
        # Fallback for near-field large boxes
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        area_frac = (box_w * box_h) / float(max(1, w * h))
        if area_frac >= 0.30:
            mn_val = float(max(getattr(state, 'min_m', 0.2), 0.2))
            return {'median_m': mn_val, 'near_m': mn_val, 'valid_ratio': vr, 'valid_pixels': vp}
        return {'median_m': None, 'near_m': None, 'valid_ratio': vr, 'valid_pixels': vp}
    try:
        mn = float(getattr(state, 'min_m', 0.2))
        mx = float(getattr(state, 'max_m', 20.0))
    except Exception:
        mn, mx = 0.2, 20.0
    med = float(np.median(valid))
    near = float(np.percentile(valid, 20))
    # Clamp
    if not np.isfinite(med):
        med = mn
    if not np.isfinite(near):
        near = med
    med = float(max(mn, min(mx, med)))
    near = float(max(mn, min(mx, near)))
    return {'median_m': med, 'near_m': near, 'valid_ratio': vr, 'valid_pixels': vp}


def calculate_object_speed(track_data, current_time):
    """Calculate speed of object based on depth change over time"""
    if len(track_data) < MIN_SAMPLES_FOR_SPEED:
        return 0.0
    
    # Get recent data within speed window
    cutoff_time = current_time - SPEED_WINDOW_SECONDS
    recent_data = [(t, d, c) for t, d, c in track_data if t >= cutoff_time and d is not None]
    
    if len(recent_data) < 3:
        return 0.0
    
    # Sort by time
    recent_data.sort(key=lambda x: x[0])
    
    # Calculate speed as depth change rate (negative = approaching)
    times = [x[0] for x in recent_data]
    depths = [x[1] for x in recent_data]
    
    if len(times) < 2:
        return 0.0
    
    # Use linear regression for more robust speed calculation
    time_diff = times[-1] - times[0]
    if time_diff <= 0:
        return 0.0
    
    depth_diff = depths[0] - depths[-1]  # positive = approaching

    # Reject jitter: require monotonic tendency and minimum delta
    step_diffs = [depths[i] - depths[i+1] for i in range(len(depths)-1)]
    approaching_steps = sum(1 for d in step_diffs if d > 0)
    receding_steps = sum(1 for d in step_diffs if d < 0)
    monotonic_tendency = approaching_steps >= (len(step_diffs) * 0.6)
    min_delta_ok = depth_diff >= 0.4 and time_diff >= 0.6
    if not (monotonic_tendency and min_delta_ok):
        return 0.0

    speed = depth_diff / time_diff
    return max(0.0, speed)  # Only positive speeds (approaching)


def assign_track_id(detection, max_distance=100):
    """Assign track ID to detection using proximity, with appearance-based re-ID for persons."""
    center = detection['center']
    current_time = detection['time']

    best_track_id = None
    min_distance = float('inf')

    # 1) Proximity to existing tracks
    for track_id, track_data in state.object_tracks.items():
        if not track_data:
            continue

        # Get last known position
        last_time, last_depth, last_center = track_data[-1]

        # Calculate distance between centers
        center_dist = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)

        # Weight by time (more recent = more important)
        time_weight = 1.0 / max(0.1, current_time - last_time)
        weighted_distance = center_dist / time_weight

        if weighted_distance < min_distance and center_dist < max_distance:
            min_distance = weighted_distance
            best_track_id = track_id

    # 2) Appearance-based re-identification for persons when proximity fails
    try:
        if (best_track_id is None) or (min_distance > (max_distance * 0.6)):
            if str(detection.get('label','')).lower() == 'person':
                desc = detection.get('descriptor')
                if desc is not None:
                    best_sim = 0.0
                    best_gallery_id = None
                    for tid, g in state.appearance_gallery.items():
                        # Only consider recently seen descriptors
                        if current_time - float(g.get('last_seen', 0.0)) > 6.0:
                            continue
                        sim = cosine_similarity(desc, g.get('desc'))
                        if sim <= 0.0:
                            continue
                        lc = g.get('last_center')
                        center_dist_g = float(np.linalg.norm(np.array(center) - np.array(lc))) if lc is not None else float('inf')
                        # Allow larger spatial tolerance when appearance is strong
                        if (sim > 0.85) and (center_dist_g < (max_distance * 2.0)) and (sim > best_sim):
                            best_sim = sim
                            best_gallery_id = tid
                    if best_gallery_id is not None:
                        best_track_id = best_gallery_id
    except Exception:
        pass

    if best_track_id is None:
        best_track_id = state.track_id_counter
        state.track_id_counter += 1

    return best_track_id


def update_performance_stats():
    """Update performance statistics"""
    current_time = time.time()
    
    # Update FPS calculation (use frame_count accumulated in camera_thread)
    elapsed = current_time - state.performance_stats['last_fps_update']
    if elapsed >= 1.0:  # Update every second
        fps = state.performance_stats.get('capture_count', 0) / max(1e-6, elapsed)
        state.performance_stats['fps_history'].append(fps)
        state.performance_stats['last_fps_update'] = current_time
        # Reset counters for next interval
        state.performance_stats['capture_count'] = 0
        state.performance_stats['frame_count'] = 0
        
        # Keep only last 10 FPS readings
        if len(state.performance_stats['fps_history']) > 10:
            state.performance_stats['fps_history'].pop(0)
        
        # Update smoothed FPS
        if state.performance_stats['fps_history']:
            state.last_fps = np.mean(state.performance_stats['fps_history'])
            # Periodic performance log
            last_log = state.performance_stats.get('last_perf_log', 0.0)
            if (current_time - last_log) >= 5.0:
                logging.info(f"Perf: FPS {state.last_fps:.1f}, Quality {state.performance_stats['quality_level']:.1f}, YOLO {state.yolo_interval:.2f}s, TopK {state.top_k_draw}")
                state.performance_stats['last_perf_log'] = current_time
    
    # Update system stats
    state.performance_stats['cpu_usage'].append(psutil.cpu_percent())
    state.performance_stats['memory_usage'].append(psutil.virtual_memory().percent)
    
    # Keep only recent stats
    for key in ['cpu_usage', 'memory_usage']:
        if len(state.performance_stats[key]) > 30:
            state.performance_stats[key].pop(0)


def adaptive_quality_control():
    """Adjust processing quality based on performance"""
    if not config["performance"]["adaptive_quality"]:
        return
    # Globals used for adaptive YOLO tuning
    global YOLO_IMG_SIZE, MAX_DET
    
    current_fps = state.last_fps
    target_fps = config["performance"]["target_fps"]
    min_fps = config["performance"]["min_fps"]
    # Throttle adjustments to avoid rapid oscillation
    now = time.time()
    last_adj = state.performance_stats.get('last_quality_adjust', 0.0)
    if (now - last_adj) < 1.0:
        return
    
    # Define adaptive bounds based on config
    min_img = max(224, config["yolo"]["img_size"] - 192)
    max_img = config["yolo"]["img_size"]
    min_det = 30
    max_det = config["yolo"]["max_det"]

    if current_fps < min_fps:
        # Reduce quality
        quality_levels = config["performance"]["quality_levels"]
        current_quality = state.performance_stats['quality_level']
        current_idx = quality_levels.index(current_quality) if current_quality in quality_levels else 3
        
        if current_idx > 0:
            new_quality = quality_levels[current_idx - 1]
            state.performance_stats['quality_level'] = new_quality
            logging.warning(f"⚠ Reducing quality to {new_quality:.1f} (FPS: {current_fps:.1f})")
        # Tighten drawing and YOLO cadence
        state.top_k_draw = max(3, state.top_k_draw - 1)
        state.yolo_interval = min(0.20, state.yolo_interval + 0.02)
        state.roi_enabled = True
        state.performance_stats['last_quality_adjust'] = now
        # Gradually reduce YOLO image size to boost FPS (multiples of 32) within bounds
        new_img = max(min_img, YOLO_IMG_SIZE - 32)
        if new_img != YOLO_IMG_SIZE:
            YOLO_IMG_SIZE = new_img
            logging.warning(f"⚠ Reducing YOLO img size to {YOLO_IMG_SIZE}")
        # Limit max detections to reduce overhead within bounds
        if MAX_DET > min_det:
            MAX_DET = max(min_det, MAX_DET - 5)
        # Clamp ROI ratio tighter when under load
        state.roi_ratio = max(0.5, min(0.9, state.roi_ratio))
        # Adapt skips under load (respect max bounds), prioritize frame_skip based on severity
        try:
            max_fs = int(config["performance"].get("max_frame_skip", 3))
            max_ds = int(config["performance"].get("max_depth_skip", 3))
            max_ys = int(config["performance"].get("max_yolo_skip", 2))
            current_fs = int(config["performance"].get("frame_skip", 1))
            # Severity: how far below minimum FPS we are
            severity = 0.0
            try:
                severity = max(0.0, (float(min_fps) - float(current_fps)) / max(1e-6, float(min_fps)))
            except Exception:
                severity = 0.0
            fs_step = 2 if severity > 0.30 else 1
            new_fs = min(max_fs, max(1, current_fs + fs_step))
            if new_fs != current_fs:
                config["performance"]["frame_skip"] = new_fs
                logging.warning(f"⚠ Adjusting frame_skip to {new_fs} (FPS {current_fps:.1f} < min {min_fps:.1f}, severity {severity:.2f})")
            # Nudge depth/yolo skips by one step to help throughput
            config["performance"]["depth_skip"] = min(max_ds, int(config["performance"]["depth_skip"]) + 1)
            config["performance"]["yolo_skip"] = min(max_ys, int(config["performance"]["yolo_skip"]) + 1)
        except Exception:
            pass
        # Dynamically lower depth resolution
        try:
            base_res = int(config["depth"].get("resolution", 384))
            new_res = max(192, int(state.dynamic_depth_resolution) - 32)
            if new_res != state.dynamic_depth_resolution:
                state.dynamic_depth_resolution = new_res
                logging.warning(f"⚠ Reducing depth resolution to {new_res}")
        except Exception:
            pass
    elif current_fps > target_fps:
        # Increase quality
        quality_levels = config["performance"]["quality_levels"]
        current_quality = state.performance_stats['quality_level']
        current_idx = quality_levels.index(current_quality) if current_quality in quality_levels else 0
        
        if current_idx < len(quality_levels) - 1:
            new_quality = quality_levels[current_idx + 1]
            state.performance_stats['quality_level'] = new_quality
            logging.info(f"✓ Increasing quality to {new_quality:.1f} (FPS: {current_fps:.1f})")
        # Relax drawing and YOLO cadence within bounds
        state.top_k_draw = min(8, state.top_k_draw + 1)
        state.yolo_interval = max(0.08, state.yolo_interval - 0.01)
        state.performance_stats['last_quality_adjust'] = now
        # Relax YOLO image size when comfortably above target within bounds
        new_img = min(max_img, YOLO_IMG_SIZE + 32)
        if new_img != YOLO_IMG_SIZE:
            YOLO_IMG_SIZE = new_img
            logging.info(f"✓ Increasing YOLO img size to {YOLO_IMG_SIZE}")
        # Restore max det if previously reduced, within bounds
        if MAX_DET < max_det:
            MAX_DET = min(max_det, MAX_DET + 5)
        # Relax skips when comfortably above target; aggressively reduce frame_skip on large headroom
        try:
            current_fs = int(config["performance"].get("frame_skip", 1))
            # When above target, reduce skips to utilize headroom and improve quality
            new_fs = max(1, current_fs - 1)
            if new_fs != current_fs:
                config["performance"]["frame_skip"] = new_fs
                logging.info(f"✓ Adjusting frame_skip to {new_fs} (FPS {current_fps:.1f} > target {target_fps:.1f})")
            config["performance"]["depth_skip"] = max(1, int(config["performance"]["depth_skip"]) - 1)
            config["performance"]["yolo_skip"] = max(1, int(config["performance"]["yolo_skip"]) - 1)
        except Exception:
            pass
        # Gradually increase depth resolution back up
        try:
            base_res = int(config["depth"].get("resolution", 384))
            new_res = min(base_res, int(state.dynamic_depth_resolution) + 32)
            if new_res != state.dynamic_depth_resolution:
                state.dynamic_depth_resolution = new_res
                logging.info(f"✓ Increasing depth resolution to {new_res}")
        except Exception:
            pass


from rgbd_stream import stream_rgbd

def camera_thread():
    # Prefer AVFoundation backend on macOS for lower latency
    # When using iPhone LiDAR via network, prefer external RGB frames
    prefer_network_rgb = True
    rgb_fresh_s = 0.6
    try:
        rgb_fresh_s = float(config.get('network_rgb_freshness_s', 0.6))
    except Exception:
        pass
    # External RGB only via TCP generator; never open local cameras.
    if prefer_network_rgb:
        logging.info('✓ Camera disabled; streaming RGB-D over TCP (generator)')
        try:
            with state.lock:
                state.degraded_mode = True
                try:
                    state.telemetry['degraded_mode'] = True
                    state.telemetry['network_rgb'] = True
                except Exception:
                    pass
        except Exception:
            pass
        frame_interval = 1.0 / TARGET_FPS
        frame_skip_counter = 0
        # Consume RGB-D stream and publish frames into runtime state
        # Respect bind host/port from config when provided
        try:
            rgbd_cfg = config.get('rgbd', {}) if isinstance(config.get('rgbd', {}), dict) else {}
        except Exception:
            rgbd_cfg = {}
        host = str(rgbd_cfg.get('bind_host', '0.0.0.0') or '0.0.0.0')
        try:
            port = int(rgbd_cfg.get('port', 6000) or 6000)
        except Exception:
            port = 6000
        logging.info(f"📡 RGB-D generator listening on {host}:{port} ...")
        for gray_img, depth_map, depth_vis in stream_rgbd(host=host, port=port):
            if not state.running:
                break
            frame_start = time.time()
            if gray_img is None:
                time.sleep(0.005)
                continue
            # Convert grayscale to BGR and enforce resolution/dtype
            try:
                frame = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            except Exception:
                frame = gray_img if gray_img.ndim == 3 else None
            if frame is None:
                continue
            try:
                h, w = frame.shape[:2]
                if (w != CAMW) or (h != CAMH):
                    frame = cv2.resize(frame, (CAMW, CAMH), interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass
            try:
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            except Exception:
                pass

            # Publish depth from stream (meters) and visualization when provided
            try:
                if depth_map is not None:
                    with state.lock:
                        state._external_depth_m = depth_map
                        state._external_depth_time = time.time()
                        state._use_network_depth = True
                if depth_vis is not None:
                    with state.lock:
                        state.depth_vis = depth_vis
            except Exception:
                pass

            # Capture/perf counters
            try:
                state.performance_stats['capture_count'] += 1
            except Exception:
                pass
            try:
                state.performance_stats['camera_capture_times'].append(time.time() - frame_start)
            except Exception:
                pass

            # Frame skipping
            frame_skip_counter += 1
            avoid_skip = (time.time() < state.hazard_boost_active_until) or (time.time() < getattr(state, '_avoid_frame_skip_until', 0.0))
            _fs_val = 1
            try:
                _fs_val = int(config.get("performance", {}).get("frame_skip", 1) or 1)
            except Exception:
                _fs_val = 1
            _fs_val = max(1, _fs_val)
            if (frame_skip_counter % _fs_val != 0) and (not avoid_skip):
                try:
                    state.performance_stats['camera_skips'] += 1
                except Exception:
                    pass
                elapsed = time.time() - frame_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif elapsed > frame_interval * 1.1:
                    frame_skip_counter += 1
                continue

            # Publish frame and queues
            with state.lock:
                state.frame = frame
                try:
                    state._last_frame_time = time.time()
                except Exception:
                    pass
                state.performance_stats['frame_count'] += 1
            try:
                if getattr(state, 'pipelines_async_enabled', False):
                    if state.frame_queue_yolo.full():
                        _ = state.frame_queue_yolo.get_nowait()
                    state.frame_queue_yolo.put_nowait(frame)
                    if state.frame_queue_depth.full():
                        _ = state.frame_queue_depth.get_nowait()
                    state.frame_queue_depth.put_nowait(frame)
            except Exception:
                pass

            # Optional preprocessing
            try:
                if getattr(state, 'use_preproc_queue', False):
                    _preproc_start = time.time()
                    pf = preprocess_frame(frame)
                    try:
                        state.performance_stats['camera_preproc_times'].append(time.time() - _preproc_start)
                    except Exception:
                        pass
                    if state.preproc_queue.full():
                        _ = state.preproc_queue.get_nowait()
                    state.preproc_queue.put_nowait(pf)
            except Exception:
                pass

            # Strict frame rate control
            elapsed = time.time() - frame_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > frame_interval * 1.1:
                frame_skip_counter += 1
        return
    cap = None
    try:
        if sys.platform == 'darwin':
            # If prefer_continuity is enabled, try common indices likely to be Continuity Camera first
            prefer_continuity = bool(config.get("camera", {}).get("prefer_continuity", True))
            selected_index = None
            if prefer_continuity:
                for idx in [1, 2, 0]:
                    try:
                        test_cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
                        if test_cap.isOpened():
                            ret, _ = test_cap.read()
                            if ret:
                                selected_index = idx
                                cap = test_cap
                                logging.info(f"✓ Using AVFoundation camera index {idx} (prefer_continuity)")
                                break
                        test_cap.release()
                    except Exception:
                        pass
            if cap is None:
                cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_AVFOUNDATION)
                if cap is not None and cap.isOpened():
                    logging.info(f"✓ Using AVFoundation camera index {CAMERA_ID}")
    except Exception:
        cap = None
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_ID)
        if cap is not None and cap.isOpened():
            logging.info(f"✓ Using default backend camera index {CAMERA_ID}")
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if cap is not None and cap.isOpened():
            logging.info("✓ Fallback to camera index 1")
    if not cap.isOpened():
        # Fallback: use external RGB frames from iPhone sender when available
        if prefer_network_rgb:
            logging.warning('Camera open failed; switching to external RGB (network) frames')
            try:
                with state.lock:
                    state.degraded_mode = True
                    try:
                        state.telemetry['degraded_mode'] = True
                        state.telemetry['network_rgb'] = True
                    except Exception:
                        pass
            except Exception:
                pass
            frame_interval = 1.0 / TARGET_FPS
            frame_skip_counter = 0
            while state.running:
                frame_start = time.time()
                # Read latest external RGB if fresh
                ext = None
                ext_time = 0.0
                with state.lock:
                    ext = getattr(state, '_external_rgb', None)
                    ext_time = float(getattr(state, '_external_rgb_time', 0.0) or 0.0)
                if (ext is not None) and ((time.time() - ext_time) <= rgb_fresh_s):
                    frame = ext
                    # Normalize format
                    try:
                        if len(frame.shape) == 3 and frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        elif len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    except Exception:
                        pass
                    # Enforce expected resolution
                    try:
                        h, w = frame.shape[:2]
                        if (w != CAMW) or (h != CAMH):
                            frame = cv2.resize(frame, (CAMW, CAMH), interpolation=cv2.INTER_LINEAR)
                    except Exception:
                        pass
                    # Ensure dtype
                    try:
                        if frame.dtype != np.uint8:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                    except Exception:
                        pass
                    # Capture/perf counters
                    try:
                        state.performance_stats['capture_count'] += 1
                    except Exception:
                        pass
                    try:
                        state.performance_stats['camera_capture_times'].append(time.time() - frame_start)
                    except Exception:
                        pass
                    # Frame skipping
                    frame_skip_counter += 1
                    avoid_skip = (time.time() < state.hazard_boost_active_until) or (time.time() < getattr(state, '_avoid_frame_skip_until', 0.0))
                    # Guard against invalid configuration values (e.g., 0) to avoid modulo errors
                    _fs_val = 1
                    try:
                        _fs_val = int(config.get("performance", {}).get("frame_skip", 1) or 1)
                    except Exception:
                        _fs_val = 1
                    _fs_val = max(1, _fs_val)
                    if (frame_skip_counter % _fs_val != 0) and (not avoid_skip):
                        try:
                            state.performance_stats['camera_skips'] += 1
                        except Exception:
                            pass
                        elapsed = time.time() - frame_start
                        sleep_time = frame_interval - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        elif elapsed > frame_interval * 1.1:
                            frame_skip_counter += 1
                        continue
                    # Publish frame and queues
                    with state.lock:
                        state.frame = frame
                        try:
                            state._last_frame_time = time.time()
                        except Exception:
                            pass
                        state.performance_stats['frame_count'] += 1
                    try:
                        if getattr(state, 'pipelines_async_enabled', False):
                            if state.frame_queue_yolo.full():
                                _ = state.frame_queue_yolo.get_nowait()
                            state.frame_queue_yolo.put_nowait(frame)
                            if state.frame_queue_depth.full():
                                _ = state.frame_queue_depth.get_nowait()
                            state.frame_queue_depth.put_nowait(frame)
                    except Exception:
                        pass
                    # Optional preprocessing
                    try:
                        if getattr(state, 'use_preproc_queue', False):
                            _preproc_start = time.time()
                            pf = preprocess_frame(frame)
                            try:
                                state.performance_stats['camera_preproc_times'].append(time.time() - _preproc_start)
                            except Exception:
                                pass
                            if state.preproc_queue.full():
                                _ = state.preproc_queue.get_nowait()
                            state.preproc_queue.put_nowait(pf)
                    except Exception:
                        pass
                else:
                    # No fresh external RGB; brief sleep to avoid busy-waiting
                    time.sleep(0.01)
                # Strict frame rate control
                elapsed = time.time() - frame_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif elapsed > frame_interval * 1.1:
                    frame_skip_counter += 1
            return
        if getattr(state, 'mock_sensor', False):
            logging.warning('Camera open failed; switching to mock sensor frames')
            try:
                with state.lock:
                    state.degraded_mode = True
                    try:
                        state.telemetry['degraded_mode'] = True
                        state.telemetry['mock_sensor'] = True
                    except Exception:
                        pass
            except Exception:
                pass
            # Synthetic frame generation loop at target FPS
            frame_interval = 1.0 / TARGET_FPS
            frame_skip_counter = 0
            t0 = time.time()
            while state.running:
                frame_start = time.time()
                # Generate a simple moving pattern to simulate motion
                img = np.zeros((CAMH, CAMW, 3), dtype=np.uint8)
                x = int(((time.time() - t0) * 120) % (CAMW + 200)) - 100
                x0 = max(0, x)
                x1 = min(CAMW - 1, x + 100)
                y0 = CAMH // 3
                y1 = 2 * CAMH // 3
                if x0 < CAMW and x1 >= 0:
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 50), -1)
                # Count capture and latency like real camera
                try:
                    state.performance_stats['capture_count'] += 1
                except Exception:
                    pass
                try:
                    state.performance_stats['camera_capture_times'].append(time.time() - frame_start)
                except Exception:
                    pass
                # Frame skipping for performance
                frame_skip_counter += 1
                avoid_skip = (time.time() < state.hazard_boost_active_until) or (time.time() < getattr(state, '_avoid_frame_skip_until', 0.0))
                _fs_val = 1
                try:
                    _fs_val = int(config.get("performance", {}).get("frame_skip", 1) or 1)
                except Exception:
                    _fs_val = 1
                _fs_val = max(1, _fs_val)
                if (frame_skip_counter % _fs_val != 0) and (not avoid_skip):
                    try:
                        state.performance_stats['camera_skips'] += 1
                    except Exception:
                        pass
                    elapsed = time.time() - frame_start
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif elapsed > frame_interval * 1.1:
                        frame_skip_counter += 1
                    continue
                # Mirror for natural viewing symmetry
                frame = cv2.flip(img, 1)
                with state.lock:
                    state.frame = frame
                    try:
                        state._last_frame_time = time.time()
                    except Exception:
                        pass
                    state.performance_stats['frame_count'] += 1
                # Optional preprocessing pathway
                try:
                    if getattr(state, 'use_preproc_queue', False):
                        _preproc_start = time.time()
                        pf = preprocess_frame(frame)
                        try:
                            state.performance_stats['camera_preproc_times'].append(time.time() - _preproc_start)
                        except Exception:
                            pass
                        if state.preproc_queue.full():
                            _ = state.preproc_queue.get_nowait()
                        state.preproc_queue.put_nowait(pf)
                except Exception:
                    pass
                # Update performance and adaptive controls
                update_performance_stats()
                adaptive_quality_control()
                # Strict frame rate control
                elapsed = time.time() - frame_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif elapsed > frame_interval * 1.1:
                    frame_skip_counter += 1
            return
        else:
            logging.error('Camera open failed')
            # Do not shut down entire runtime; mark degraded and allow watchdog to recover
            try:
                with state.lock:
                    try:
                        state.telemetry['degraded_mode'] = True
                    except Exception:
                        pass
                    try:
                        setattr(state, 'camera_failed', True)
                    except Exception:
                        pass
                    try:
                        state._last_frame_time = 0.0
                    except Exception:
                        pass
            except Exception:
                pass
            return
    
    # Optimize camera settings for 25 FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMH)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # Frame rate control variables
    frame_interval = 1.0 / TARGET_FPS
    last_frame_time = time.time()
    frame_skip_counter = 0
    camera_fail_counter = 0

    def _reopen_camera():
        nonlocal cap
        try:
            candidates = [CAMERA_ID, 0, 1, 2, 3]
            backends = []
            if sys.platform == 'darwin':
                backends = [cv2.CAP_AVFOUNDATION, 0]
            else:
                backends = [0]
            for backend in backends:
                for idx in candidates:
                    try:
                        test_cap = cv2.VideoCapture(idx, backend) if backend != 0 else cv2.VideoCapture(idx)
                    except Exception:
                        test_cap = cv2.VideoCapture(idx)
                    if not test_cap or (not test_cap.isOpened()):
                        try:
                            if test_cap:
                                test_cap.release()
                        except Exception:
                            pass
                        continue
                    # Probe a frame to ensure the device is delivering
                    for _ in range(3):
                        ret_probe, frame_probe = test_cap.read()
                        if ret_probe and frame_probe is not None and hasattr(frame_probe, 'size') and frame_probe.size > 0:
                            try:
                                if cap and cap.isOpened():
                                    cap.release()
                            except Exception:
                                pass
                            cap = test_cap
                            logging.info(f"✓ Reopened camera index {idx} using backend {backend} after read failures")
                            return True
                        time.sleep(0.02)
                    try:
                        test_cap.release()
                    except Exception:
                        pass
        except Exception:
            pass
        return False
    
    while state.running:
        frame_start = time.time()
        # Allow controlled restart by watchdog
        if state.restart_request.get('camera', False):
            break
        
        # Read a frame; guard against transient camera glitches or empty buffers
        ret, frame = cap.read()
        if (not ret) or (frame is None) or (not hasattr(frame, 'size')) or (frame.size == 0):
            try:
                state.performance_stats['camera_read_failures'] = state.performance_stats.get('camera_read_failures', 0) + 1
            except Exception:
                pass
            # Rate-limited warning to avoid log spam on noisy drivers
            try:
                log_error('Camera read failure', 'ret=False or invalid frame', level='warning', rate_limit_s=2.0)
            except Exception:
                pass
            camera_fail_counter += 1
            # If continuous failures observed, attempt to reopen camera on alternative indices/backends
            if camera_fail_counter >= int(TARGET_FPS * 3):  # ~3 seconds worth of failures
                if _reopen_camera():
                    camera_fail_counter = 0
                else:
                    # Avoid tight loop when no device is available
                    time.sleep(0.25)
            time.sleep(0.01)
            continue
        else:
            camera_fail_counter = 0

        # Normalize frame format for downstream pipelines
        try:
            # Some macOS/iPhone/IP webcam feeds deliver BGRA or grayscale; convert to BGR
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        except Exception:
            # If conversion fails, keep original for best-effort processing
            pass

        # Prefer external RGB from iPhone sender when fresh
        if prefer_network_rgb:
            try:
                ext = None
                ext_time = 0.0
                with state.lock:
                    ext = getattr(state, '_external_rgb', None)
                    ext_time = float(getattr(state, '_external_rgb_time', 0.0) or 0.0)
                if (ext is not None) and ((time.time() - ext_time) <= rgb_fresh_s):
                    frame = ext
                    # Normalize format for external source
                    if len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    elif len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            except Exception:
                pass

        # Enforce expected resolution when drivers ignore CAP_PROP settings
        try:
            h, w = frame.shape[:2]
            if (w != CAMW) or (h != CAMH):
                frame = cv2.resize(frame, (CAMW, CAMH), interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass

        # Ensure dtype is uint8
        try:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        except Exception:
            pass

        # Count every capture for accurate FPS measurement
        try:
            state.performance_stats['capture_count'] += 1
        except Exception:
            pass

        # Record capture latency
        try:
            state.performance_stats['camera_capture_times'].append(time.time() - frame_start)
        except Exception:
            pass
        
        # Frame skipping for performance
        frame_skip_counter += 1
        # Never skip frames during critical/hazard windows
        avoid_skip = (time.time() < state.hazard_boost_active_until) or (time.time() < state._avoid_frame_skip_until)
        _fs_val = 1
        try:
            _fs_val = int(config.get("performance", {}).get("frame_skip", 1) or 1)
        except Exception:
            _fs_val = 1
        _fs_val = max(1, _fs_val)
        if (frame_skip_counter % _fs_val != 0) and (not avoid_skip):
            try:
                state.performance_stats['camera_skips'] += 1
            except Exception:
                pass
            continue
        
        # Mirror the frame horizontally in the opposite direction
        # Using flip parameter 1 for horizontal flip (left-right)
        # Original code had flip(frame, 1), now we're not flipping to show original orientation
        # frame = cv2.flip(frame, 1)
        
        with state.lock:
            state.frame = frame
            # Track last frame time for permission/health monitoring
            try:
                state._last_frame_time = time.time()
            except Exception:
                pass
            state.performance_stats['frame_count'] += 1

        # Pipeline: push frames to YOLO and Depth queues for concurrent consumption
        try:
            if getattr(state, 'pipelines_async_enabled', False):
                if state.frame_queue_yolo.full():
                    _ = state.frame_queue_yolo.get_nowait()
                state.frame_queue_yolo.put_nowait(frame)
                if state.frame_queue_depth.full():
                    _ = state.frame_queue_depth.get_nowait()
                state.frame_queue_depth.put_nowait(frame)
        except Exception:
            pass

        # Lightweight preprocessing and deep buffer with backpressure
        try:
            if state.use_preproc_queue:
                _preproc_start = time.time()
                pf = preprocess_frame(frame)
                try:
                    state.performance_stats['camera_preproc_times'].append(time.time() - _preproc_start)
                except Exception:
                    pass
                if state.preproc_queue.full():
                    # Drop oldest to maintain freshness
                    _ = state.preproc_queue.get_nowait()
                state.preproc_queue.put_nowait(pf)
        except Exception:
            pass
        
        # Update performance stats
        update_performance_stats()
        adaptive_quality_control()
        
        # Strict frame rate control for 25 FPS
        elapsed = time.time() - frame_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif elapsed > frame_interval * 1.1:  # If we're running slow, opportunistically skip next frame
            frame_skip_counter += 1
    
    cap.release()


# --- Positioning providers and manager ---
def _gps_init_try():
    """Attempt to initialize CoreLocation; return (supported, ctor)"""
    try:
        import objc  # noqa: F401
        from CoreLocation import CLLocationManager, kCLLocationAccuracyBest
        return True, (CLLocationManager, kCLLocationAccuracyBest)
    except Exception:
        return False, (None, None)


def _recv_all(conn, n):
    data = bytearray()
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)

def rgbd_stream_thread():
    """Ingest RGB-D stream and feed runtime state.

    Protocol:
      - Header: 1 byte `frame_type` + 4 bytes big-endian payload length
      - frame_type 0: depth float32 HxW (meters)
      - frame_type 1: grayscale uint8 HxW

    Modes:
      - Server: listen on `rgbd.port` and accept a single connection
      - Client: connect to `rgbd.client_host`:`rgbd.client_port`

    Config:
      - `rgbd.port` (default 6000)
      - `rgbd.client_host` (optional, enables client mode)
      - `rgbd.client_port` (default 6000)
      - `logging.dataset_enabled` (bool, default False)
      - `logging.dataset_interval` (int, default 10)
      - Saves to `dataset/` when enabled
    """
    DEPTH_W, DEPTH_H = 256, 192
    # Prevent duplicate server starts in the same process
    try:
        with state.lock:
            if getattr(state, '_rgbd_server_started', False):
                logging.info('RGB-D stream server already running; skipping duplicate start')
                return
            else:
                state._rgbd_server_started = True
    except Exception:
        pass
    # Server-only fixed configuration: bind 0.0.0.0:6000
    HOST = '0.0.0.0'
    PORT = 6000
    client_host = ''
    client_port = PORT
    try:
        dataset_enabled = bool(config.get('logging', {}).get('dataset_enabled', False))
    except Exception:
        dataset_enabled = False
    try:
        save_every = int(config.get('logging', {}).get('dataset_interval', 10) or 10)
    except Exception:
        save_every = 10
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    if dataset_enabled:
        try:
            os.makedirs(dataset_dir, exist_ok=True)
        except Exception:
            dataset_enabled = False

    frame_count = 0
    depth_map = None
    depth_vis = None
    gray_img = None

    def _handle_payload(frame_type, payload):
        nonlocal depth_map, depth_vis, gray_img, frame_count
        if frame_type == 0:
            # Depth float32 meters DEPTH_H x DEPTH_W
            try:
                arr = np.frombuffer(payload, dtype=np.float32)
                if arr.size == DEPTH_W * DEPTH_H:
                    depth_map = arr.reshape(DEPTH_H, DEPTH_W)
                else:
                    w = int((arr.size * 4 / 3) ** 0.5)
                    h = max(1, arr.size // max(1, w))
                    depth_map = arr.reshape(h, w)
                try:
                    dv = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(dv, cv2.COLORMAP_JET)
                except Exception:
                    depth_vis = None
                now = time.time()
                with state.lock:
                    state._external_depth_m = depth_map
                    state._external_depth_time = now
                    if depth_vis is not None:
                        state.depth_vis = depth_vis
                    state._use_network_depth = True
            except Exception as e:
                try:
                    log_error('rgbd_stream.depth', e, level='debug', rate_limit_s=2.0)
                except Exception:
                    pass
        elif frame_type == 1:
            # Grayscale uint8 HxW -> BGR resized to CAMW x CAMH
            try:
                gray = np.frombuffer(payload, dtype=np.uint8)
                pixels = gray.size
                if pixels == 480 * 360:
                    gw, gh = 480, 360
                elif pixels == 960 * 720:
                    gw, gh = 960, 720
                elif pixels == 1280 * 720:
                    gw, gh = 1280, 720
                else:
                    gw = int((pixels * 4 / 3) ** 0.5)
                    gh = max(1, pixels // max(1, gw))
                gray_img = gray.reshape(gh, gw)
                bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
                try:
                    if (bgr.shape[1] != CAMW) or (bgr.shape[0] != CAMH):
                        bgr = cv2.resize(bgr, (CAMW, CAMH), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    pass
                now = time.time()
                with state.lock:
                    state._external_rgb = bgr
                    state._external_rgb_time = now
                try:
                    if bool(config.get('camera', {}).get('prefer_network_rgb', False)) or (str(config.get('depth', {}).get('source', 'model')).lower() == 'network'):
                        with state.lock:
                            state.frame = bgr
                            state._last_frame_time = now
                except Exception:
                    pass
            except Exception as e:
                try:
                    log_error('rgbd_stream.gray', e, level='debug', rate_limit_s=2.0)
                except Exception:
                    pass
        # Dataset logging every Nth pair
        try:
            if dataset_enabled and (frame_count % max(1, save_every) == 0):
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                if gray_img is not None:
                    cv2.imwrite(os.path.join(dataset_dir, f'gray_{ts}.png'), gray_img)
                if depth_map is not None:
                    np.save(os.path.join(dataset_dir, f'depth_m_{ts}.npy'), depth_map)
            frame_count += 1
        except Exception:
            pass
    
    # Client mode disabled: always run as server on port 6000

    # Server mode: accept a single connection
    # Preflight: if port 6000 already has a listener (likely our own server), skip binding
    try:
        probe_host = '127.0.0.1' if HOST == '0.0.0.0' else HOST
        _ps = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _ps.settimeout(0.5)
        in_use = (_ps.connect_ex((probe_host, PORT)) == 0)
        _ps.close()
        if in_use:
            logging.info(f"ℹ️ RGB-D server already active at {probe_host}:{PORT}; skipping duplicate bind")
            return
    except Exception:
        pass
    logging.info(f"📡 RGB-D stream server-only listening on {HOST}:{PORT}")
    s = None
    conn = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, PORT))
        except OSError as e:
            try:
                logging.warning(f"RGB-D TCP bind failed on {HOST}:{PORT}: {e}. If a legacy receiver is running, stop it or set 'rgbd.client_host' to connect.")
            except Exception:
                pass
            try:
                s.close()
            except Exception:
                pass
            return
        s.listen(1)
        s.settimeout(1.0)
        while state.running:
            try:
                if conn is None:
                    try:
                        conn, addr = s.accept()
                        conn.setblocking(True)
                        logging.info(f"✓ RGB-D stream connected from {addr}")
                        try:
                            print(f"RGB-D: accepted connection from {addr}")
                        except Exception:
                            pass
                        with state.lock:
                            state.telemetry['network_rgb'] = True
                            state.telemetry['network_depth'] = True
                    except socket.timeout:
                        continue
                header = _recv_all(conn, 5)
                if not header:
                    time.sleep(0.01)
                    continue
                frame_type = header[0]
                payload_len = struct.unpack('>I', header[1:5])[0]
                try:
                    print(f"RGB-D: header received type={frame_type} len={payload_len}")
                except Exception:
                    pass
                payload = _recv_all(conn, payload_len)
                if not payload:
                    continue
                try:
                    print(f"RGB-D: payload received bytes={len(payload)}")
                except Exception:
                    pass
                _handle_payload(frame_type, payload)
            except Exception:
                time.sleep(0.01)
            except KeyboardInterrupt:
                break
            except Exception as e:
                try:
                    log_error('rgbd_stream.loop', e, level='warning', rate_limit_s=2.0)
                except Exception:
                    pass
                # On connection drop, reset and continue listening
                try:
                    if conn:
                        conn.close()
                except Exception:
                    pass
                conn = None
        # Cleanup
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass
        try:
            if s:
                s.close()
        except Exception:
            pass
        logging.info('RGB-D stream thread closed cleanly')

def _ble_wifi_init_try():
    """Attempt to initialize CoreBluetooth scanning; return supported flag."""
    try:
        import objc  # noqa: F401
        from CoreBluetooth import CBCentralManager  # noqa: F401
        return True
    except Exception:
        return False


def _process_external_position_updates():
    """Drain external indoor updates pushed by other modules (BLE/Wi-Fi)."""
    updated = False
    now = time.time()
    try:
        while not state._position_updates.empty():
            upd = state._position_updates.get_nowait()
            # Expected schema: {'provider': 'ble'|'wifi', 'x': float, 'y': float, 'accuracy_m': float}
            if isinstance(upd, dict) and upd.get('provider') in ('ble', 'wifi'):
                with state.lock:
                    state.positioning['provider'] = upd.get('provider')
                    state.positioning['x'] = float(upd.get('x', 0.0) or 0.0)
                    state.positioning['y'] = float(upd.get('y', 0.0) or 0.0)
                    state.positioning['accuracy_m'] = float(upd.get('accuracy_m', 5.0) or 5.0)
                    state.positioning['indoor'] = True
                    state.positioning['last_update_ts'] = now
                updated = True
    except Exception:
        pass
    return updated


def _update_cv_odometry():
    """Compute relative indoor displacement via optical flow when enabled."""
    if not state.positioning_config.get('cv_enabled', True):
        return False
    frame = None
    with state.lock:
        if state.frame is not None:
            frame = state.frame.copy()
    if frame is None:
        return False
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return False
    updated = False
    try:
        if state._cv_prev_gray is None:
            state._cv_prev_gray = gray
            return False
        # Detect and track features
        prev = state._cv_prev_gray
        feat_params = dict(maxCorners=state.positioning_config['cv_feature_count'],
                           qualityLevel=state.positioning_config['cv_min_quality'],
                           minDistance=7,
                           blockSize=7)
        p0 = cv2.goodFeaturesToTrack(prev, mask=None, **feat_params)
        if p0 is None or len(p0) < state.positioning_config['cv_min_points']:
            state._cv_prev_gray = gray
            return False
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None)
        if p1 is None or st is None:
            state._cv_prev_gray = gray
            return False
        good_old = p0[st == 1]
        good_new = p1[st == 1]
        if len(good_old) < state.positioning_config['cv_min_points']:
            state._cv_prev_gray = gray
            return False
        # Compute median flow
        flow = (good_new - good_old).reshape(-1, 2)
        dx = np.median(flow[:, 0])
        dy = np.median(flow[:, 1])
        # Normalize by diagonal for scale invariance
        h, w = gray.shape[:2]
        diag = (h ** 2 + w ** 2) ** 0.5
        norm = ((dx ** 2 + dy ** 2) ** 0.5) / max(1e-6, diag)
        # Convert to local displacement (rough scale from config)
        local_dx = (dx / max(1.0, w)) * state.positioning_config.get('pixel_to_meter_scale', 1.0)
        local_dy = (-dy / max(1.0, h)) * state.positioning_config.get('pixel_to_meter_scale', 1.0)
        with state.lock:
            state._cv_pose['x'] += local_dx
            state._cv_pose['y'] += local_dy
            state.positioning['provider'] = 'cv'
            state.positioning['x'] = state._cv_pose['x']
            state.positioning['y'] = state._cv_pose['y']
            # Heuristic accuracy: better when more motion observed
            state.positioning['accuracy_m'] = max(3.0, 1.0 / max(1e-6, norm))
            state.positioning['indoor'] = True
            state.positioning['last_update_ts'] = time.time()
        updated = True
    except Exception:
        pass
    finally:
        state._cv_prev_gray = gray
    return updated


def _positioning_provider_switch():
    """Decide active provider based on GPS accuracy/freshness and indoor signals."""
    now = time.time()
    with state.lock:
        gps_fresh = (now - state._last_gps_update_ts) <= state.positioning_config['gps_stale_s']
        gps_ok_acc = (state.positioning.get('accuracy_m') or 1e9) <= state.positioning_config['gps_accuracy_threshold_m']
        gps_viable = state._gps_supported and gps_fresh and gps_ok_acc
    if gps_viable:
        with state.lock:
            state.positioning['provider'] = 'gps'
            state.positioning['indoor'] = False
        return 'gps'
    # Prefer external BLE/Wi-Fi indoor if any recent update
    if _process_external_position_updates():
        return state.positioning['provider']
    # Fallback to CV odometry indoors
    if _update_cv_odometry():
        return 'cv'
    return state.positioning.get('provider') or None


def positioning_thread():
    """Background positioning manager: integrates GPS + indoor providers with seamless switching."""
    # Initialize provider capabilities
    gps_supported, gps_ctor = _gps_init_try()
    ble_wifi_supported = _ble_wifi_init_try()
    with state.lock:
        state._gps_supported = gps_supported
        state._ble_wifi_supported = ble_wifi_supported
    # Try to instantiate CoreLocation manager if supported
    cl_manager = None
    if gps_supported and gps_ctor[0] is not None:
        try:
            from CoreLocation import kCLLocationAccuracyBest
            cl_manager = gps_ctor[0].alloc().init()
            # Best effort accuracy; auth calls are no-ops in headless contexts
            try:
                cl_manager.desiredAccuracy = gps_ctor[1]
                cl_manager.startUpdatingLocation()
            except Exception:
                pass
        except Exception:
            cl_manager = None
    last_provider = None
    while state.running:
        try:
            # Poll GPS location if available
            if cl_manager is not None:
                try:
                    loc = cl_manager.location()
                    if loc is not None:
                        coord = loc.coordinate()
                        lat = float(getattr(coord, 'latitude', None) or 0.0)
                        lon = float(getattr(coord, 'longitude', None) or 0.0)
                        acc = float(getattr(loc, 'horizontalAccuracy', None) or 50.0)
                        with state.lock:
                            state.positioning['lat'] = lat
                            state.positioning['lon'] = lon
                            state.positioning['accuracy_m'] = acc
                            state._last_gps_update_ts = time.time()
                except Exception:
                    pass
            provider = _positioning_provider_switch()
            if provider != last_provider and provider is not None:
                logging.info(f"Positioning provider switched to: {provider}")
                last_provider = provider
            # Push lightweight telemetry marker
            try:
                state.telemetry['events'].append({
                    'time': time.time(),
                    'type': 'system',
                    'importance': 'low',
                    'message': f"positioning_provider={provider} accuracy_m={state.positioning.get('accuracy_m')} indoor={state.positioning.get('indoor')}"
                })
            except Exception:
                pass
            # Update environment context with positioning snapshot
            with state.lock:
                pos = state.positioning.copy()
                ec = state.environment_context
                ec['positioning'] = {
                    'provider': pos.get('provider'),
                    'indoor': bool(pos.get('indoor', False)),
                    'accuracy_m': pos.get('accuracy_m'),
                    'lat': pos.get('lat'), 'lon': pos.get('lon'),
                    'x': pos.get('x'), 'y': pos.get('y'),
                    'last_update_ts': pos.get('last_update_ts')
                }
        except Exception:
            pass
        time.sleep(0.2)

def depth_thread(depth_pipe):
    """Depth processing worker thread.

    Acquires frames or external depth, runs DepthAnythingV2 or external resize,
    performs postprocessing (temporal smoothing, clamping, center ROI probe), and
    updates shared state and performance telemetry. Uses `safe_execute()` and
    `log_error()` to ensure robustness and rate-limited error logs.

    Args:
        depth_pipe: Callable that accepts a frame and returns a dict with depth
            outputs (e.g., `{'depth': np.ndarray, 'low_res': np.ndarray}`), or an
            adapter over external depth sources.

    Returns:
        None. This function runs a loop while `state.running` is True.
    """
    """
    Depth processing loop that computes a robust, smoothed depth map in meters.

    The thread supports external LiDAR/network depth sources with freshness checks
    and falls back to local model inference when necessary. It applies temporal
    smoothing, probes center depth for hazard detection, and updates shared state
    for visualization, TTC, and path planning.

    Args:
        depth_pipe: Callable or pipeline that accepts a PIL Image and returns a
            dict with a `'depth'` array. May be `None` when using external depth.

    Behavior:
        - Skips frames based on `performance.depth_skip` and dynamic rate limits.
        - Honors hazard boost by increasing resolution and disabling rate limits.
        - Resizes external depth to current frame size using area interpolation.
        - Records inference and total times in `state.performance_stats`.
        - Updates `state.depth_vis`, `state.depth_map_m`, and `state.center_depth_m`.

    Resiliency:
        - Wraps preprocessing and postprocessing in try/except, using safe
          defaults on failure (blank RGB frame, default meters map).
        - Uses `state._last_depth_time` to cooperate with the watchdog.
    """
    prev_depth = None
    depth_skip_counter = 0
    
    while state.running and state.depth_ready and (not state.restart_request.get('depth', False)):
        frame = None
        # Prefer pipeline queue when enabled; else use shared frame
        if getattr(state, 'pipelines_async_enabled', False):
            try:
                # Get latest frame for depth
                item = _queue_get_latest(state.frame_queue_depth)
                if item is not None:
                    frame = item
            except Exception:
                frame = None
        if frame is None:
            # Fallback to shared state frame when pipeline is disabled or empty
            with state.lock:
                if state.frame is not None:
                    frame = state.frame
        if frame is None:
            time.sleep(0.01)
            continue
        
        # Frame skipping for depth processing (bypass when hazard boost or external depth is fresh)
        depth_skip_counter += 1
        now = time.time()
        boost_active = now < getattr(state, 'hazard_boost_active_until', 0.0)
        avoid_skip_active = now < getattr(state, '_avoid_frame_skip_until', 0.0)
        # Detect fresh external depth to process every frame without compute cost
        use_network = bool(getattr(state, '_use_network_depth', False))
        ext_fresh = False
        if use_network and (getattr(state, '_external_depth_m', None) is not None):
            try:
                mode_upper = str(getattr(state, 'travel_mode', 'WALKING')).upper()
            except Exception:
                mode_upper = 'WALKING'
            fresh_default = float(config.get('network_depth_freshness_s', 0.8))
            fresh_s = float(config.get('network_depth_freshness_s_per_mode', {}).get(mode_upper, fresh_default))
            ext_fresh = (now - float(getattr(state, '_external_depth_time', 0.0) or 0.0)) <= fresh_s
        # Guard against invalid config values for depth_skip to avoid modulo errors
        _ds_val = 1
        try:
            _ds_val = int(config.get("performance", {}).get("depth_skip", 1) or 1)
        except Exception:
            _ds_val = 1
        _ds_val = max(1, _ds_val)
        if (depth_skip_counter % _ds_val != 0) and not boost_active and not avoid_skip_active and not ext_fresh:
            try:
                state.performance_stats['depth_skips'] += 1
            except Exception:
                pass
            time.sleep(0.01)
            continue
        
        # Rate-limit depth inference for 25 FPS
        effective_min_interval = state.depth_min_interval
        if boost_active:
            effective_min_interval = min(effective_min_interval, float(config["hazard"]["depth_min_interval_s"]))
        # Do not rate-limit when external depth is fresh or skip avoidance is active
        if ((now - state._last_depth_time) < effective_min_interval) and not ext_fresh and not avoid_skip_active and not boost_active:
            try:
                state.performance_stats['depth_skips'] += 1
            except Exception:
                pass
            time.sleep(0.005)
            continue
        state._last_depth_time = now
        
        # Optimized resolution for real-time performance (dynamic)
        target_w = int(getattr(state, 'dynamic_depth_resolution', config["depth"]["resolution"]))
        # During hazard boost, raise minimum resolution to improve depth accuracy
        try:
            if boost_active:
                boost_min = int(config.get('hazard', {}).get('depth_resolution_boost_min', 320))
                target_w = max(target_w, boost_min)
        except Exception:
            pass
        
        # Use optimized processor for faster depth frame preparation
        rgb = None
        global optimized_processor
        with profile_scope('depth.preprocess'):
            if optimized_processor is not None:
                try:
                    # Use optimized processor for depth frame preparation
                    _, depth_frame = optimized_processor.process_frame_unified(frame, depth_size=target_w)
                    if depth_frame is not None:
                        rgb = depth_frame
                except Exception as e:
                    try:
                        log_error('Depth preprocess optimized', e, level='debug', rate_limit_s=2.0)
                    except Exception:
                        pass
            
            # Fallback to original processing if optimized processor failed
            if rgb is None:
                try:
                    small_h = int(frame.shape[0]*target_w/frame.shape[1])
                    small = cv2.resize(frame, (target_w, max(1, small_h)))
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    try:
                        log_error('Depth preprocess cv2', e, level='debug', rate_limit_s=2.0)
                    except Exception:
                        pass
                    # Fallback to a blank RGB frame to keep pipeline alive
                    rgb = np.zeros((max(1, int(frame.shape[0]*target_w/max(1, frame.shape[1]))), target_w, 3), dtype=np.uint8)
        
        try:
            _depth_total_start = time.time()
            _depth_inf_start = time.time()
            depth_raw = None
            use_network = bool(getattr(state, '_use_network_depth', False))
            # Prefer external LiDAR depth when configured and fresh
            # Per-mode freshness override
            fresh_default = float(config.get('network_depth_freshness_s', 0.8))
            try:
                mode_upper = str(getattr(state, 'travel_mode', 'WALKING')).upper()
            except Exception:
                mode_upper = 'WALKING'
            fresh_s = float(config.get('network_depth_freshness_s_per_mode', {}).get(mode_upper, fresh_default))
            with profile_scope('depth.inference'):
                if use_network and (state._external_depth_m is not None) and ((now - state._external_depth_time) <= fresh_s):
                    try:
                        # Resize external depth to current frame size
                        h, w = frame.shape[:2]
                        ext = state._external_depth_m
                        dh, dw = ext.shape[:2]
                        if (dh != h) or (dw != w):
                            # Resize with area interpolation for smooth depth
                            depth_raw = cv2.resize(ext, (w, h), interpolation=cv2.INTER_AREA)
                        else:
                            depth_raw = ext
                        # Count this as an inference bypass
                        try:
                            state.performance_stats['depth_inference_times'].append(0.0)
                        except Exception:
                            pass
                    except Exception as re:
                        try:
                            log_error('Network depth resize', re, level='warning', rate_limit_s=5.0)
                        except Exception:
                            pass
                        depth_raw = None
            # Fallback to local/HF model pipeline
            if depth_raw is None:
                # depth_pipe may be None when source is network-only; guard call
                if depth_pipe is None:
                    # No depth available; skip this cycle lightly
                    time.sleep(0.005)
                    continue
                # Execute model inference with standardized error handling and zeros fallback
                def _depth_infer():
                    return depth_pipe(Image.fromarray(rgb))
                def _depth_fallback():
                    h0, w0 = rgb.shape[:2]
                    return {'depth': np.zeros((h0, w0), dtype=np.float32)}
                out = safe_execute('Depth model inference', _depth_infer, fallback=_depth_fallback, level='warning', rate_limit_s=5.0)
                try:
                    state.performance_stats['depth_inference_times'].append(time.time() - _depth_inf_start)
                except Exception:
                    pass
                depth_raw = np.array(out['depth']) if out is not None and 'depth' in out else np.zeros(rgb.shape[:2], dtype=np.float32)
            _post_start = time.time()
            # Postprocess depth with standardized error handling and safe zeros fallback
            def _post_fn():
                return postprocess_depth(depth_raw, frame.shape, rgb_frame=frame)
            def _post_fallback():
                h, w = frame.shape[:2]
                return (np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w), dtype=np.float32))
            with profile_scope('depth.postprocess'):
                pp = safe_execute('Depth postprocess', _post_fn, fallback=_post_fallback, level='warning', rate_limit_s=5.0)
            depth_vis, depth_m = pp if pp is not None else _post_fallback()
            try:
                state.performance_stats['postprocess_depth_times'].append(time.time() - _post_start)
            except Exception:
                pass
            
            # Optimized temporal smoothing
            base_alpha = float(config["depth"]["alpha"]) 
            # Dynamic adaptation: adjust smoothing in low light/outdoors
            _bright = _compute_brightness(frame)
            alpha = base_alpha
            try:
                if _bright < 60.0:  # low light: slightly stronger smoothing
                    alpha = min(0.8, base_alpha + 0.1)
                elif _bright > 180.0:  # bright outdoors: keep responsive ~0.6
                    alpha = max(0.5, min(0.65, base_alpha))
            except Exception:
                alpha = base_alpha
            _smooth_start = time.time()
            with profile_scope('depth.smooth'):
                depth_m = temporal_smooth(prev_depth, depth_m, alpha=alpha)
            try:
                state.performance_stats['temporal_smooth_times'].append(time.time() - _smooth_start)
            except Exception:
                pass
            prev_depth = depth_m

            # Clamp to reliable range to suppress extreme shallows/depths
            try:
                mn = float(getattr(state, 'min_m', 0.2))
                mx_rel = float(getattr(state, 'max_m', 8.0))  # honor config/state max depth (e.g., 5m)
                depth_m = np.clip(depth_m.astype(np.float32), mn, mx_rel).astype(np.float32)
            except Exception:
                pass
            
            # Compute fast center ROI probe for looming detection
            try:
                with profile_scope('depth.center_probe'):
                    h, w = depth_m.shape
                    wf = float(config["hazard"]["center_probe_width_frac"]) 
                    hf = float(config["hazard"]["center_probe_height_frac"]) 
                    cw = max(4, int(w * wf))
                    ch = max(4, int(h * hf))
                    cx = (w - cw) // 2
                    cy = (h - ch) // 2
                    patch = depth_m[cy:cy+ch, cx:cx+cw]
                    center_depth = float(np.median(patch[np.isfinite(patch)])) if np.isfinite(patch).any() else None
                    nowc = time.time()
                    if center_depth is not None and state._last_center_depth is not None and state._last_center_depth_time > 0:
                        dt = max(1e-3, nowc - state._last_center_depth_time)
                        dd = state._last_center_depth - center_depth  # approaching => positive
                        state.center_approach_mps = max(0.0, float(dd / dt))
                    state._last_center_depth = center_depth
                    state._last_center_depth_time = nowc
            except Exception:
                pass

            # Record total processing latency
            try:
                state.performance_stats['depth_total_times'].append(time.time() - _depth_total_start)
            except Exception:
                pass

            # Determine if depth inference produced valid finite values
            try:
                valid_depth = bool(np.isfinite(depth_m).any())
            except Exception:
                valid_depth = False

            with state.lock:
                state.depth_vis = depth_vis
                state.depth_map_m = depth_m
                state.center_depth_m = center_depth if 'center_depth' in locals() else state.center_depth_m
                try:
                    if valid_depth:
                        state.last_success_depth_ts = time.time()
                except Exception:
                    pass
            # Pipeline: push depth map to queue for downstream consumers
            try:
                if getattr(state, 'pipelines_async_enabled', False):
                    intel = _analyze_approach_and_clear(state, frame, depth_m)
                    payload = {
                        'ts': time.time(),
                        'depth': depth_m,
                        'depth_vis': depth_vis,
                        'center_depth_m': center_depth if 'center_depth' in locals() else None,
                        'approach_detected': bool(intel.get('approach_detected')), 
                        'approach_direction': intel.get('approach_direction'),
                        'estimated_distance': intel.get('estimated_distance'),
                        'path_clear': bool(intel.get('path_clear'))
                    }
                    if state.depth_map_queue.full():
                        _ = state.depth_map_queue.get_nowait()
                    state.depth_map_queue.put_nowait(payload)
            except Exception:
                pass
        except Exception as e:
            try:
                log_error('Depth processing', e, level='warning', rate_limit_s=5.0)
            except Exception:
                pass
            continue


def preprocess_frame(frame):
    """
    Prepare a frame for YOLO with optimized preprocessing when available.

    Attempts to use a shared `optimized_processor` to perform parallel
    preprocessing for both YOLO and depth paths. Falls back to returning the
    original frame to preserve stability if optimization fails.

    Args:
        frame: BGR image (`numpy.ndarray`, HxWx3) from the camera.

    Returns:
        Preprocessed frame optimized for YOLO, or the original frame when the
        optimized path is unavailable or fails.
    """
    global optimized_processor
    if optimized_processor is not None:
        try:
            # Use optimized processor to process YOLO and depth in parallel, priming both caches
            try:
                depth_w = int(getattr(state, 'dynamic_depth_resolution', config["depth"]["resolution"]))
            except Exception:
                depth_w = int(config["depth"]["resolution"])
            yolo_frame, _ = optimized_processor.process_frame_parallel(frame, yolo_size=YOLO_IMG_SIZE, depth_size=depth_w)
            if yolo_frame is not None:
                return yolo_frame
        except Exception as e:
            logging.debug(f"Optimized preprocessing failed, using fallback: {e}")
    
    # Fallback to original minimal preprocessing for stability
    return frame

def _to_rgb_safe(img):
    """Convert an input image to RGB robustly, handling BGR, BGRA, and GRAY.

    Returns the original image if conversion fails.
    """
    try:
        if img is None:
            return None
        if len(img.shape) == 3:
            ch = img.shape[2]
            if ch == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if ch == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    except Exception:
        pass
    return img

def _resize_square_safe(img, size):
    """Resize image to a square of given size, with safe fallback."""
    try:
        if img is None:
            return None
        return cv2.resize(img, (int(size), int(size)), interpolation=cv2.INTER_LINEAR)
    except Exception:
        return img

def compute_descriptor(frame, bbox):
    """
    Compute a lightweight HSV histogram descriptor for appearance matching.

    This descriptor helps track objects across frames by comparing color
    histograms in the cropped bounding box region.

    Args:
        frame: BGR image (`numpy.ndarray`, HxWx3).
        bbox: Tuple `(x1, y1, x2, y2)` describing the bounding box in pixels.

    Returns:
        A normalized `numpy.ndarray` of shape `(24,)` with 8 bins per channel
        for H, S, and V. Returns `None` when the crop is invalid or processing
        fails.
    """
    try:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        bins = 8
        hist_h = cv2.calcHist([hsv],[0],None,[bins],[0,180]).flatten()
        hist_s = cv2.calcHist([hsv],[1],None,[bins],[0,256]).flatten()
        hist_v = cv2.calcHist([hsv],[2],None,[bins],[0,256]).flatten()
        desc = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        norm = np.linalg.norm(desc) + 1e-6
        return (desc / norm)
    except Exception:
        return None

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two descriptors with numeric guards.

    Args:
        a: First descriptor vector (`numpy.ndarray` or list-like).
        b: Second descriptor vector.

    Returns:
        Cosine similarity in `[0, 1]` when inputs are valid; `0.0` on invalid
        inputs, zero norms, or exceptions.
    """
    try:
        if a is None or b is None:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
        if denom <= 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)
    except Exception:
        return 0.0

def update_encounter_memory_for_track(track_id):
    """Mark tracks as 'passed' to suppress immediate re-announcements.
    Trigger when object was near and depth starts increasing (moving away)."""
    try:
        data = state.object_tracks.get(track_id)
        if not data or len(data) < 3:
            return
        t2, d2, _ = data[-1]
        t1, d1, _ = data[-2]
        t0, d0, _ = data[-3]
        if (d0 is None) or (d1 is None) or (d2 is None):
            return
        # Was near, then depth increased notably
        was_near = (min(d0, d1) <= WARNING_DISTANCE_M)
        increasing = (d2 - d1) >= 0.5
        stable_before = abs(d1 - d0) <= 0.2
        if was_near and increasing and stable_before:
            state.encounter_memory[track_id] = {'last_pass': t2}
        # Prune old entries
        nowt = time.time()
        stale = [tid for tid, m in state.encounter_memory.items() if (nowt - float(m.get('last_pass', 0.0))) > 10.0]
        for tid in stale:
            try:
                del state.encounter_memory[tid]
            except Exception:
                pass
    except Exception:
        pass

def _get_center_corridor_rect(frame_shape, config):
    h, w = frame_shape[:2]
    cw = float(config.get('hazard', {}).get('center_probe_width_frac', 0.34))
    ch = float(config.get('hazard', {}).get('center_probe_height_frac', 0.34))
    rect_w = int(w * cw)
    rect_h = int(h * ch)
    x0 = (w - rect_w) // 2
    y0 = (h - rect_h) // 2
    return (x0, y0, rect_w, rect_h)

def roi_in_center_corridor(detection, frame_shape, config):
    try:
        x0, y0, rw, rh = _get_center_corridor_rect(frame_shape, config)
        x, y, bw, bh = detection['bbox']
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        return (x0 <= cx <= x0 + rw) and (y0 <= cy <= y0 + rh)
    except Exception:
        return False

def predict_crossing_to_center(track_history, frame_shape, config, horizon_s=0.8):
    try:
        if track_history is None or len(track_history) < 2:
            return False
        (t1, d1, b1), (t2, d2, b2) = track_history[-2], track_history[-1]
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        c1 = (x1 + w1 / 2.0, y1 + h1 / 2.0)
        c2 = (x2 + w2 / 2.0, y2 + h2 / 2.0)
        dt = max(0.01, t2 - t1)
        vx = (c2[0] - c1[0]) / dt
        vy = (c2[1] - c1[1]) / dt
        xf = c2[0] + vx * horizon_s
        yf = c2[1] + vy * horizon_s
        x0, y0, rw, rh = _get_center_corridor_rect(frame_shape, config)
        in_future = (x0 <= xf <= x0 + rw) and (y0 <= yf <= y0 + rh)
        if in_future:
            return True
        cx_rect = x0 + rw / 2.0
        cy_rect = y0 + rh / 2.0
        dir_to_rect = (cx_rect - c2[0], cy_rect - c2[1])
        dot = vx * dir_to_rect[0] + vy * dir_to_rect[1]
        return dot > 0
    except Exception:
        return False


def compute_bearing_side(detection, frame_shape, config):
    """Return 'ahead' if object center lies within the center corridor; otherwise 'left' or 'right'."""
    try:
        # Use corridor rectangle for robust center-aware classification
        x0, y0, rw, rh = _get_center_corridor_rect(frame_shape, config)
        x, y, bw, bh = detection.get('bbox')
        cx = x + bw / 2.0
        # Inside corridor => ahead
        if (x0 <= cx <= (x0 + rw)):
            side = 'ahead'
            # Update pan hint based on bbox
            try:
                ang_deg, pan_val = compute_angle_pan_from_detection(detection, frame_shape, config)
                _set_spatial_pan_hint(ang_deg, pan_val)
            except Exception:
                pass
            return side
        # Outside corridor: decide side based on corridor boundaries
        if cx < x0:
            side = 'left'
        elif cx > (x0 + rw):
            side = 'right'
        else:
            side = 'ahead'
        try:
            ang_deg, pan_val = compute_angle_pan_from_detection(detection, frame_shape, config)
            _set_spatial_pan_hint(ang_deg, pan_val)
        except Exception:
            pass
        return side
    except Exception:
        # Fallback to coarse width fractions if bbox/shape missing
        try:
            w = frame_shape[1]
            cx = detection.get('center', (w * 0.5, 0))[0]
            if cx < w * 0.4:
                side = 'left'
            elif cx > w * 0.6:
                side = 'right'
            else:
                side = 'ahead'
            try:
                ang_deg, pan_val = compute_angle_pan_from_detection(detection, frame_shape, config)
                _set_spatial_pan_hint(ang_deg, pan_val)
            except Exception:
                pass
            return side
        except Exception:
            return 'ahead'


def apply_ema_smoothing(detection, track_id, alpha=0.7):
    """Apply EMA smoothing to bounding boxes for stability"""
    if track_id not in state.ema_boxes:
        state.ema_boxes[track_id] = {
            'bbox': detection['bbox'],
            'confidence': detection['conf'],
            'label': detection['label']
        }
        return detection
    
    # Smooth bounding box coordinates
    old_bbox = state.ema_boxes[track_id]['bbox']
    new_bbox = detection['bbox']
    
    smoothed_bbox = (
        int(alpha * new_bbox[0] + (1 - alpha) * old_bbox[0]),
        int(alpha * new_bbox[1] + (1 - alpha) * old_bbox[1]),
        int(alpha * new_bbox[2] + (1 - alpha) * old_bbox[2]),
        int(alpha * new_bbox[3] + (1 - alpha) * old_bbox[3])
    )
    
    # Smooth confidence
    smoothed_conf = alpha * detection['conf'] + (1 - alpha) * state.ema_boxes[track_id]['confidence']
    
    # Update EMA state
    state.ema_boxes[track_id] = {
        'bbox': smoothed_bbox,
        'confidence': smoothed_conf,
        'label': detection['label']
    }
    
    # Return smoothed detection
    smoothed_detection = detection.copy()
    smoothed_detection['bbox'] = smoothed_bbox
    smoothed_detection['conf'] = smoothed_conf
    
    return smoothed_detection

def compute_bearing_fine_side(detection, frame_shape, config):
    """Return fine-grained side: 'center', 'slight_left', 'left', 'slight_right', or 'right'.
    Uses the center corridor with a margin to avoid false strong left/right classifications.
    """
    try:
        x0, y0, rw, rh = _get_center_corridor_rect(frame_shape, config)
        x, y, bw, bh = detection.get('bbox')
        cx = x + bw / 2.0
        w = frame_shape[1]
        # Margin near corridor edges counts as slight deviations
        margin_frac = float(config.get('free_space', {}).get('slight_margin_frac', 0.05))
        margin = max(2, int(w * margin_frac))
        if x0 <= cx <= (x0 + rw):
            try:
                ang_deg, pan_val = compute_angle_pan_from_detection(detection, frame_shape, config)
                _set_spatial_pan_hint(ang_deg, pan_val)
            except Exception:
                pass
            return 'center'
        if (x0 - margin) <= cx < x0:
            try:
                ang_deg, pan_val = compute_angle_pan_from_detection(detection, frame_shape, config)
                _set_spatial_pan_hint(ang_deg, pan_val)
            except Exception:
                pass
            return 'slight_left'
        if (x0 + rw) < cx <= (x0 + rw + margin):
            try:
                ang_deg, pan_val = compute_angle_pan_from_detection(detection, frame_shape, config)
                _set_spatial_pan_hint(ang_deg, pan_val)
            except Exception:
                pass
            return 'slight_right'
        if cx < (x0 - margin):
            try:
                ang_deg, pan_val = compute_angle_pan_from_detection(detection, frame_shape, config)
                _set_spatial_pan_hint(ang_deg, pan_val)
            except Exception:
                pass
            return 'left'
        try:
            ang_deg, pan_val = compute_angle_pan_from_detection(detection, frame_shape, config)
            _set_spatial_pan_hint(ang_deg, pan_val)
        except Exception:
            pass
        return 'right'
    except Exception:
        # Fallback to coarse fractions
        try:
            w = frame_shape[1]
            cx = detection.get('center', (w * 0.5, 0))[0]
            if cx < w * 0.35:
                side = 'left'
            elif cx < w * 0.45:
                side = 'slight_left'
            elif cx > w * 0.65:
                side = 'right'
            elif cx > w * 0.55:
                side = 'slight_right'
            else:
                side = 'center'
            try:
                ang_deg, pan_val = compute_angle_pan_from_detection(detection, frame_shape, config)
                _set_spatial_pan_hint(ang_deg, pan_val)
            except Exception:
                pass
            return side
        except Exception:
            return 'center'

def compute_angle_pan_from_detection(detection, frame_shape, config):
    """Compute horizontal angle in degrees and stereo pan (-1..1) from a detection bbox.
    Angle is estimated from bbox center relative to frame center using an approximate horizontal FOV.
    Pan is proportional to normalized horizontal offset, with a deadzone inside the center corridor.
    """
    try:
        sa = config.get('spatial_audio', {})
        fov_deg = float(sa.get('horizontal_fov_deg', 60.0))
        pan_gain = float(sa.get('pan_gain', 1.0))
        deadzone_gain = float(sa.get('deadzone_pan_gain', 0.2))
        w = int(frame_shape[1])
        # Center X from bbox or detection center fallback
        if 'bbox' in detection and detection['bbox'] is not None:
            x, y, bw, bh = detection.get('bbox')
            cx = float(x) + float(bw) / 2.0
        else:
            cx = float(detection.get('center', (w * 0.5, 0))[0])
        # Normalize [-1,1] where -1 is far left, +1 is far right
        nx = 0.0
        if w > 0:
            nx = (cx - (w / 2.0)) / (w / 2.0)
        nx = max(-1.0, min(1.0, nx))
        angle_deg = float(nx * (fov_deg / 2.0))
        # Deadzone: soften pan inside the center corridor
        try:
            x0, y0, rw, rh = _get_center_corridor_rect(frame_shape, config)
            in_corridor = (cx >= x0) and (cx <= (x0 + rw))
        except Exception:
            in_corridor = abs(nx) < 0.15
        if in_corridor:
            pan = float(nx * deadzone_gain)
        else:
            pan = float(nx * pan_gain)
        pan = max(-1.0, min(1.0, pan))
        return angle_deg, pan
    except Exception:
        return 0.0, 0.0

def _set_spatial_pan_hint(angle_deg: float, pan: float):
    """Store a short-lived pan/angle hint in state to be used by spatial earcon playback."""
    try:
        with state.lock:
            state._last_spatial_angle_deg = float(angle_deg)
            state._last_spatial_pan = float(pan)
            state._last_spatial_pan_ts = time.time()
    except Exception:
        # Best-effort; ignore failures
        pass

def ensemble_detect(yolo, ensemble_models, frame, device):
    """Run ensemble detection for better accuracy"""
    all_detections = []
    
    # Primary model
    try:
        target_sz = int(getattr(state, '_yolo_img_override', None) or YOLO_IMG_SIZE)
        rgb_frame = preprocess_frame_for_yolo(frame, target_size=target_sz)
    except Exception:
        rgb_frame = frame
        res = safe_execute(
            'YOLO Ultralytics predict (ensemble primary)',
            lambda: yolo.predict(
                rgb_frame,
                imgsz=int(target_sz),
                conf=float(YOLO_CONF),
                iou=float(YOLO_IOU),
                max_det=int(MAX_DET),
                device=device,
                verbose=False,
                half=True if device.type == 'cuda' else False
            ),
            fallback=[],
            level='warning',
            rate_limit_s=5.0
        )
    if res and res[0].boxes is not None:
        all_detections.append(res[0])
    
    # Ensemble models
    for model in ensemble_models:
        res = safe_execute(
            'YOLO Ultralytics predict (ensemble secondary)',
            lambda: model.predict(
                rgb_frame,
                imgsz=int(target_sz),
                conf=float(YOLO_CONF),
                iou=float(YOLO_IOU),
                max_det=int(MAX_DET),
                device=device,
                verbose=False,
                half=True if device.type == 'cuda' else False
            ),
            fallback=[],
            level='warning',
            rate_limit_s=5.0
        )
        if res and res[0].boxes is not None:
            all_detections.append(res[0])
    
    # Combine detections using weighted voting
    if len(all_detections) > 1:
        return combine_detections(all_detections)
    elif all_detections:
        return all_detections[0]
    else:
        return None


def combine_detections(detections):
    """Fuse ensemble detections by selecting the highest-confidence result.

    Given a list of detection results (Ultralytics-like objects with `.boxes.conf`),
    chooses the element with the maximum confidence. Empty/None entries are
    ignored.

    Args:
        detections: List of detection result objects; may include None.

    Returns:
        The selected detection result, or None when no valid entries exist.
    """
    """Combine multiple detection results using weighted voting"""
    if not detections:
        return None
    
    # Simple combination: take the detection with highest confidence
    best_detection = None
    best_score = 0
    
    for det in detections:
        if det.boxes is not None and len(det.boxes) > 0:
            max_conf = float(det.boxes.conf.max())
            if max_conf > best_score:
                best_score = max_conf
                best_detection = det
    
    return best_detection


def yolo_thread(yolo, device, ensemble_models=None):
    """YOLO detection worker thread.

    Runs Ultralytics or Core ML YOLO inference at a configured cadence, with
    ROI-focused frames and optional ensemble fusion (`combine_detections`).
    Updates shared state on liveness, performance stats, and passes detections to
    downstream announcement logic. Errors are wrapped with `safe_execute()`.

    Args:
        yolo: Ultralytics model instance.
        device: Torch device or runtime hint for model execution.
        ensemble_models: Optional list of models for ensemble detection.

    Returns:
        None. Runs a loop while `state.running` is True.
    """
    """
    YOLO detection loop with ROI-based acceleration and hazard-aware cadence.

    Runs object detection either via Core ML (when configured) or Ultralytics
    YOLO. Applies an ROI-first strategy to focus inference on the center region
    during most cycles, with adaptive ROI shifting toward the nearest hazard
    lane. Balances performance and responsiveness using skip counters and
    dynamic intervals.

    Args:
        yolo: Ultralytics YOLO model instance.
        device: Torch device or backend identifier.
        ensemble_models: Optional list of auxiliary detectors for ensembling.

    Behavior:
        - Honors hazard boost and critical proximity to reduce interval times.
        - Prefers Core ML when `config['yolo']['use_coreml']` is true and a
          compatible model/package is available.
        - Records inference times and processing latency into performance stats.
        - Updates `state.last_yolo_time` and produces bounded detections.

    Resiliency:
        - Wraps inference in try/except and falls back from Core ML to Ultralytics.
        - Clamps detection coordinates to frame bounds and validates boxes.
        - Safely reads confidence/class/labels and depth per bbox.
    """
    yolo_skip_counter = 0
    roi_counter = 0
    
    while state.running and (not state.restart_request.get('yolo', False)):
        current_time = time.time()
        boost_active = current_time < state.hazard_boost_active_until
        # Compute hazard proximity using center depth and lane slices
        critical_near = False
        try:
            mode = getattr(state, 'travel_mode', 'WALKING')
            factor = 1.0
            if str(mode).upper() in ('DRIVING', 'CYCLING'):
                factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
            critical_m = CRITICAL_DISTANCE_M * factor
            cd = getattr(state, 'center_depth_m', None)
            # Debug center depth before proximity condition
            try:
                logging.info(f"Hazard proximity check: center_depth_m={cd} critical_m={critical_m}")
            except Exception:
                pass
            if (cd is not None) and (cd <= critical_m):
                critical_near = True
            else:
                lanes = getattr(state, 'environment_context', {}).get('lanes', [])
                if lanes:
                    try:
                        # Closest lane by average depth
                        nearest = min(lanes, key=lambda L: float(L.get('avg_depth_m', 99.0)))
                        try:
                            logging.info(f"Hazard proximity lanes: nearest_avg_depth_m={float(nearest.get('avg_depth_m', 99.0))} critical_m={critical_m}")
                        except Exception:
                            pass
                        if float(nearest.get('avg_depth_m', 99.0)) <= critical_m:
                            critical_near = True
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Frame skipping for YOLO processing (guard against invalid config values)
        yolo_skip_counter += 1
        _ys_val = 1
        try:
            _ys_val = int(config.get("performance", {}).get("yolo_skip", 1) or 1)
        except Exception:
            _ys_val = 1
        _ys_val = max(1, _ys_val)
        if (yolo_skip_counter % _ys_val != 0) and not boost_active:
            try:
                state.performance_stats['yolo_skips'] += 1
            except Exception:
                pass
            time.sleep(0.01)
            continue
        
        # Only run YOLO at reduced rate for stability (use faster interval on hazard)
        effective_interval = state.yolo_interval
        if critical_near:
            effective_interval = float(config["hazard"].get("yolo_interval_critical_s", config["hazard"]["yolo_interval_s"]))
        elif boost_active:
            effective_interval = float(config["hazard"]["yolo_interval_s"])
        if current_time - state.last_yolo_time < effective_interval:
            try:
                state.performance_stats['yolo_skips'] += 1
            except Exception:
                pass
            time.sleep(0.01)
            continue
        
        frame = None
        # Prefer async pipeline frames for YOLO when enabled
        try:
            if getattr(state, 'pipelines_async_enabled', False):
                item = _queue_get_latest(state.frame_queue_yolo)
                if item is not None:
                    frame = item
        except Exception:
            frame = None
        # Fallback to preprocessed buffer, then shared state
        if frame is None:
            try:
                if state.use_preproc_queue and (state.preproc_queue.qsize() > 0):
                    frame = state.preproc_queue.get_nowait()
            except Exception:
                frame = None
        if frame is None:
            with state.lock:
                if state.frame is not None:
                    # Avoid copy to improve tensor sharing and cache hits
                    frame = state.frame
        if frame is None:
            time.sleep(0.01)
            continue
        
        # Performance timing
        processing_start = time.time()
        
        # Minimal preprocessing for stability
        _yolo_preproc_start = time.time()
        with profile_scope('yolo.preprocess'):
            processed_frame = preprocess_frame(frame)
        try:
            state.performance_stats['yolo_preproc_times'].append(time.time() - _yolo_preproc_start)
        except Exception:
            pass
        
        # ROI-first detection: crop central region most of the time
        use_full = True
        roi_offset = (0, 0)
        roi_counter += 1
        if state.roi_enabled:
            use_full = (roi_counter % state.roi_full_every_n == 0)
            # Under critical hazard, prioritize ROI more aggressively
            if critical_near:
                use_full = False
        inference_frame = processed_frame
        roi_offset = (0, 0)
        _roi_start = time.time()
        with profile_scope('yolo.roi'):
            if not use_full:
                h, w = processed_frame.shape[:2]
                rw = int(w * state.roi_ratio)
                rh = int(h * state.roi_ratio)
                rx = (w - rw) // 2
                ry = (h - rh) // 2
                # Adaptive ROI shift toward nearest hazard lane for faster detection
                try:
                    if bool(config.get('hazard', {}).get('roi_shift_on_hazard', True)) and critical_near:
                        lanes = getattr(state, 'environment_context', {}).get('lanes', [])
                        if lanes:
                            nearest = min(lanes, key=lambda L: float(L.get('avg_depth_m', 99.0)))
                            cx_lane = (float(nearest.get('start', 0)) + float(nearest.get('end', w))) / 2.0
                            rx = int(max(0, min(w - rw, cx_lane - (rw / 2.0))))
                except Exception:
                    pass
                roi_offset = (rx, ry)
                inference_frame = processed_frame[ry:ry+rh, rx:rx+rw]
        try:
            state.performance_stats['yolo_roi_times'].append(time.time() - _roi_start)
        except Exception:
            pass
        
        # Run YOLO detection with optimized settings (Core ML preferred if available)
        res = None
        used_coreml = False
        used_ultra = False
        yolo_inf_start = time.time()
        # Track ROI shapes and offsets for accurate box remapping
        roi_shapes = []
        roi_offsets = []
        use_full_list = []
        try:
            with profile_scope('yolo.inference'):
                if state.yolo_coreml is not None and coreml_is_enabled(config):
                    # Attempt Core ML prediction
                    try:
                        # Try common input names; ensure RGB color space for Core ML
                        target_sz = int(state._yolo_img_override or YOLO_IMG_SIZE)
                        lb_img, lb_ratio, lb_pad = letterbox_with_meta(inference_frame, target_size=target_sz)
                        # Record letterbox metadata for accurate post-process scaling
                        lb_ratio_sel, lb_pad_sel = lb_ratio, lb_pad
                        h_roi_sel, w_roi_sel = inference_frame.shape[:2]
                        # Save debug metadata for overlay
                        try:
                            with state.lock:
                                state.debug_meta = {
                                    'img0_shape': (int(h_roi_sel), int(w_roi_sel)),
                                    'img1_shape': (int(target_sz), int(target_sz)),
                                    'lb_ratio': float(lb_ratio_sel) if not isinstance(lb_ratio_sel, (tuple, list)) else lb_ratio_sel,
                                    'lb_pad': lb_pad_sel
                                }
                        except Exception:
                            pass
                        coreml_input = {'image': Image.fromarray(lb_img)}
                        out = safe_execute('YOLO CoreML predict', lambda: state.yolo_coreml.predict(coreml_input), fallback=None, level='warning', rate_limit_s=5.0)
                    except Exception:
                        target_sz = int(state._yolo_img_override or YOLO_IMG_SIZE)
                        lb_img, lb_ratio, lb_pad = letterbox_with_meta(inference_frame, target_size=target_sz)
                        # Record letterbox metadata for accurate post-process scaling
                        lb_ratio_sel, lb_pad_sel = lb_ratio, lb_pad
                        h_roi_sel, w_roi_sel = inference_frame.shape[:2]
                        # Save debug metadata for overlay
                        try:
                            with state.lock:
                                state.debug_meta = {
                                    'img0_shape': (int(h_roi_sel), int(w_roi_sel)),
                                    'img1_shape': (int(target_sz), int(target_sz)),
                                    'lb_ratio': float(lb_ratio_sel) if not isinstance(lb_ratio_sel, (tuple, list)) else lb_ratio_sel,
                                    'lb_pad': lb_pad_sel
                                }
                        except Exception:
                            pass
                        coreml_input = {'input': Image.fromarray(lb_img)}
                        out = safe_execute('YOLO CoreML predict', lambda: state.yolo_coreml.predict(coreml_input), fallback=None, level='warning', rate_limit_s=5.0)
                    # Expect keys like 'boxes', 'scores', 'labels'; fallback tolerant
                    boxes = out.get('boxes') or out.get('coordinates') if out is not None else None
                    scores = out.get('scores') or out.get('confidence') if out is not None else None
                    labels = out.get('labels') or out.get('class') if out is not None else None
                    if boxes is not None and scores is not None:
                        # Create a lightweight result structure similar to Ultralytics
                        # Always use the model's native class names; custom_classes are for filtering only
                        try:
                            names_map = getattr(yolo, 'names', {})
                            # Build reverse map to resolve string labels -> class indices when needed
                            rev_map = {str(v).lower(): int(k) for k, v in (names_map.items() if isinstance(names_map, dict) else [])}
                        except Exception:
                            names_map = {}
                            rev_map = {}

                        class Box:
                            def __init__(self, b, s, c):
                                # Ultralytics iteration often expects `box.xyxy[0]`, so keep a list-like structure
                                try:
                                    arr = np.array(b, dtype=np.float32)
                                except Exception:
                                    arr = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                                self.xyxy = [arr]
                                # Keep conf/cls indexable so downstream `box.conf[0]` works
                                try:
                                    self.conf = [float(s)]
                                except Exception:
                                    self.conf = [0.0]
                                try:
                                    self.cls = [int(c)]
                                except Exception:
                                    self.cls = [0]

                        class Result:
                            def __init__(self, boxes):
                                self.boxes = boxes

                        # Normalize Core ML output shapes and types
                        bxs = []
                        try:
                            n = min(len(boxes), len(scores))
                            for i in range(n):
                                b = boxes[i]
                                # Convert to plain list [x1,y1,x2,y2]
                                if isinstance(b, (np.ndarray, torch.Tensor)):
                                    b = b.tolist()
                                # Some Core ML outputs nest coordinates; flatten conservatively
                                if isinstance(b, (list, tuple)) and len(b) >= 4:
                                    bb = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                                else:
                                    bb = [0.0, 0.0, 0.0, 0.0]
                                # Score
                                try:
                                    sc = float(scores[i])
                                except Exception:
                                    sc = 0.0
                                # Class label may be int or string; resolve robustly
                                ci = 0
                                try:
                                    ci = int(labels[i]) if labels is not None else 0
                                except Exception:
                                    try:
                                        # Map string label to index via reverse names_map
                                        ci = int(rev_map.get(str(labels[i]).lower(), 0))
                                    except Exception:
                                        ci = 0
                                bxs.append(Box(bb, sc, ci))
                            res = Result(bxs)
                            used_coreml = True
                        except Exception:
                            res = None
                    else:
                        res = None
                if res is None:
                    # Fallback to Ultralytics YOLO (supports micro-batching)
                    # Select confidence threshold based on hazard severity
                    conf_used = YOLO_CONF
                    try:
                        if critical_near:
                            conf_used = min(YOLO_CONF, float(config['hazard'].get('yolo_conf_critical_floor', config['hazard'].get('yolo_conf_floor', YOLO_CONF))))
                        elif boost_active:
                            conf_used = min(YOLO_CONF, float(config['hazard'].get('yolo_conf_floor', YOLO_CONF)))
                    except Exception:
                        conf_used = YOLO_CONF
                # Build a micro-batch when enabled and beneficial
                batch_enabled = bool(config.get('performance', {}).get('yolo_batch_enabled', True))
                batch_size = int(config.get('performance', {}).get('yolo_batch_size', 1))
                batch_delay_ms = int(config.get('performance', {}).get('yolo_batch_delay_ms', 0))
                inference_batch = [inference_frame]
                roi_offsets = [roi_offset]
                use_full_list = [use_full]
                roi_shapes = [inference_frame.shape[:2]]
                if batch_enabled and batch_size > 1 and not critical_near:
                    try:
                        # Pull additional frames from preproc queue without waiting
                        if getattr(state, 'use_preproc_queue', False):
                            while len(inference_batch) < batch_size and state.preproc_queue.qsize() > 0:
                                try:
                                    nxt = state.preproc_queue.get_nowait()
                                except Exception:
                                    break
                                if nxt is None:
                                    break
                                # Preprocess and ROI-crop like the primary frame
                                pf = preprocess_frame(nxt)
                                inf_f = pf
                                rx, ry = 0, 0
                                is_full = True
                                if state.roi_enabled:
                                    h2, w2 = pf.shape[:2]
                                    rw2 = int(w2 * state.roi_ratio)
                                    rh2 = int(h2 * state.roi_ratio)
                                    rx = (w2 - rw2) // 2
                                    ry = (h2 - rh2) // 2
                                    # Keep ROI selection similar; avoid hazard shift for queued frames
                                    inf_f = pf[ry:ry+rh2, rx:rx+rw2]
                                    is_full = False
                                inference_batch.append(inf_f)
                                roi_offsets.append((rx, ry))
                                use_full_list.append(is_full)
                                roi_shapes.append(inf_f.shape[:2])
                        # Optional tiny delay to allow another frame to arrive
                        if batch_delay_ms > 0 and len(inference_batch) < batch_size and not boost_active:
                            time.sleep(float(batch_delay_ms) / 1000.0)
                    except Exception:
                        pass

                # Predict on single frame or micro-batch: pass raw ROI frames to Ultralytics
                # Avoid pre-letterboxing here to let Ultralytics handle transforms and scale boxes back to the input image.
                try:
                    target_sz_used = int(state._yolo_img_override or YOLO_IMG_SIZE)
                except Exception:
                    target_sz_used = YOLO_IMG_SIZE
                predict_input = inference_batch if len(inference_batch) > 1 else inference_frame
                # Identity lb metadata since we are not letterboxing for Ultralytics path
                lb_meta_list = [(1.0, (0.0, 0.0))] * (len(inference_batch) if len(inference_batch) > 1 else 1)

                # Resolve device param for Ultralytics predict API
                try:
                    device_param = device.type if isinstance(device, torch.device) else device
                except Exception:
                    device_param = device

                reslist = safe_execute(
                    'YOLO Ultralytics predict',
                    lambda: yolo.predict(
                        predict_input,
                        imgsz=int(state._yolo_img_override or YOLO_IMG_SIZE),
                        conf=float(conf_used),
                        iou=float(YOLO_IOU),
                        max_det=int(MAX_DET),
                        agnostic=bool(config.get('yolo', {}).get('agnostic_nms', False)),
                        device=device_param,
                        verbose=False,
                        half=True if isinstance(device, torch.device) and device.type == 'cuda' else False,
                        save=False,
                        show=False,
                        augment=False
                    ),
                    fallback=[],
                    level='warning',
                    rate_limit_s=5.0
                )
                try:
                    dets = reslist[0].boxes
                except Exception:
                    pass
                if reslist:
                    # If batched, take the last (newest) result
                    res = reslist[-1] if isinstance(reslist, (list, tuple)) else reslist[0]
                    used_ultra = True
                    # Select corresponding ROI mapping for the chosen result
                    try:
                        roi_offset = roi_offsets[-1]
                        use_full = use_full_list[-1]
                        h_roi_sel, w_roi_sel = roi_shapes[-1]
                        lb_ratio_sel, lb_pad_sel = lb_meta_list[-1]
                    except Exception:
                        h_roi_sel, w_roi_sel = inference_frame.shape[:2]
                        roi_offset = roi_offsets[0] if roi_offsets else (0, 0)
                        use_full = use_full_list[0] if use_full_list else True
                        lb_ratio_sel, lb_pad_sel = (1.0, (0.0, 0.0))
                    try:
                        # Basic detection diagnostics: count boxes and top confidence
                        bx = res.boxes
                        if bx is not None:
                            num = int(len(bx))
                            topc = float(torch.max(bx.conf).item()) if hasattr(bx, 'conf') and len(bx) > 0 else 0.0
                            ts = time.time()
                            state.performance_stats.setdefault('yolo_det_counts', []).append({'time': ts, 'count': num, 'top_conf': topc})
                            # Trim history
                            if len(state.performance_stats['yolo_det_counts']) > 50:
                                state.performance_stats['yolo_det_counts'].pop(0)
                            last_log = state.performance_stats.get('last_det_log', 0.0)
                            if (ts - last_log) >= 5.0:
                                logging.info(f"YOLO det: count {num}, top_conf {topc:.2f}, conf_used {conf_used:.2f}, max_det {MAX_DET}, imgsz {int(state._yolo_img_override or YOLO_IMG_SIZE)}")
                                state.performance_stats['last_det_log'] = ts
                    except Exception:
                        pass
        except Exception as e:
            logging.warning(f"YOLO error: {e}")
            res = None

        # Record YOLO inference latency
        try:
            state.performance_stats['yolo_inference_times'].append(time.time() - yolo_inf_start)
            if used_coreml:
                state.performance_stats['yolo_coreml_calls'] += 1
                state.performance_stats['yolo_coreml_times'].append(state.performance_stats['yolo_inference_times'][-1])
            if used_ultra:
                state.performance_stats['yolo_ultra_calls'] += 1
                state.performance_stats['yolo_ultra_times'].append(state.performance_stats['yolo_inference_times'][-1])
        except Exception:
            pass
        
        # Update processing time
        processing_time = time.time() - processing_start
        state.performance_stats['processing_times'].append(processing_time)
        if len(state.performance_stats['processing_times']) > 30:
            state.performance_stats['processing_times'].pop(0)
        
        state.last_yolo_time = current_time
        # Mark successful YOLO inference only when results are present
        try:
            if res is not None:
                state.last_success_yolo_ts = time.time()
        except Exception:
            pass
        dets = []
        if res is not None and res.boxes is not None:
            with profile_scope('yolo.postprocess'):
                h_full, w_full = frame.shape[:2]
                # Determine scaling from model input back to ROI/original using ratio/pad
                try:
                    h_roi_sel, w_roi_sel
                except Exception:
                    h_roi_sel, w_roi_sel = inference_frame.shape[:2]

                # Choose scaling strategy: Ultralytics predictor scales to ROI; Core ML needs manual un-letterbox
                try:
                    coreml_active = bool(used_coreml)
                except Exception:
                    coreml_active = False

                if coreml_active:
                    # Core ML path used explicit letterbox; recover ratio/pad from debug_meta
                    try:
                        meta = getattr(state, 'debug_meta', {}) or {}
                        lb_ratio_sel = meta.get('lb_ratio', 1.0)
                        lb_pad_sel = meta.get('lb_pad', (0.0, 0.0))
                    except Exception:
                        lb_ratio_sel, lb_pad_sel = (1.0, (0.0, 0.0))
                    target_sz_sel = int(getattr(state, '_yolo_img_override', None) or YOLO_IMG_SIZE)
                    img1_shape = (int(target_sz_sel), int(target_sz_sel))
                    img0_shape = (int(h_roi_sel), int(w_roi_sel))
                else:
                    # Ultralytics path: identity mapping; boxes already in ROI space
                    img1_shape = (int(h_roi_sel), int(w_roi_sel))
                    img0_shape = (int(h_roi_sel), int(w_roi_sel))
                    lb_ratio_sel, lb_pad_sel = (1.0, (0.0, 0.0))

                # Record debug metadata for overlay
                try:
                    with state.lock:
                        state.debug_meta = {
                            'img0_shape': img0_shape,
                            'img1_shape': img1_shape,
                            'lb_ratio': lb_ratio_sel,
                            'lb_pad': lb_pad_sel
                        }
                except Exception:
                    pass

                for box in res.boxes:
                    xy = box.xyxy[0]
                    if hasattr(xy, 'cpu'):
                        coords = xy.cpu().numpy().astype(np.float32)
                    else:
                        coords = np.array(xy, dtype=np.float32)

                    if coreml_active:
                        # Un-letterbox Core ML boxes back to ROI/original image space
                        try:
                            from ultralytics.utils import ops as _ops
                            arr = np.expand_dims(coords, axis=0)  # shape (1,4)
                            # scale_boxes expects ratio_pad=( (gain_x, gain_y), (pad_x, pad_y) )
                            try:
                                gain = float(lb_ratio_sel) if isinstance(lb_ratio_sel, (int, float)) else float(lb_ratio_sel[0])
                            except Exception:
                                gain = 1.0
                            rp = ((gain, gain), lb_pad_sel)
                            _ops.scale_boxes(img1_shape, arr, img0_shape, ratio_pad=rp, padding=True, xywh=False)
                            coords = arr[0]
                        except Exception:
                            # Fallback to manual reverse of ratio/pad
                            try:
                                r = float(lb_ratio_sel) if not isinstance(lb_ratio_sel, (tuple, list)) else float(lb_ratio_sel[0])
                            except Exception:
                                r = 1.0
                            try:
                                dw, dh = lb_pad_sel
                            except Exception:
                                dw, dh = (0.0, 0.0)
                            coords = np.array([
                                (coords[0] - float(dw)) / max(1e-6, r),
                                (coords[1] - float(dh)) / max(1e-6, r),
                                (coords[2] - float(dw)) / max(1e-6, r),
                                (coords[3] - float(dh)) / max(1e-6, r),
                            ], dtype=np.float32)

                    # Ultralytics boxes are in ROI coordinates already; Core ML now remapped
                    x1 = float(coords[0])
                    y1 = float(coords[1])
                    x2 = float(coords[2])
                    y2 = float(coords[3])

                    # Map ROI coords back to full-frame if ROI was used
                    if not use_full:
                        try:
                            off = roi_offset
                        except Exception:
                            off = (0, 0)
                        x1 += off[0]
                        x2 += off[0]
                        y1 += off[1]
                        y2 += off[1]

                    # Clamp to frame bounds defensively
                    x1 = max(0, min(w_full - 1, int(x1)))
                    y1 = max(0, min(h_full - 1, int(y1)))
                    x2 = max(0, min(w_full - 1, int(x2)))
                    y2 = max(0, min(h_full - 1, int(y2)))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Confidence and class guards
                    conf = 0.0
                    try:
                        conf = float(box.conf[0])
                    except Exception:
                        pass
                    cls = 0
                    try:
                        cls = int(box.cls[0])
                    except Exception:
                        pass

                    # Label fallback
                    name_map = getattr(yolo, 'names', {})
                    label = str(cls)
                    if isinstance(name_map, dict):
                        label = name_map.get(cls, label) or label

                    # Restrict detections to configured custom classes when provided
                    custom = [c.lower() for c in config.get("yolo", {}).get("custom_classes", [])]
                    if custom and label.lower() not in custom:
                        continue

                    # Apply per-class confidence overrides (post-filter) to balance recall vs precision
                    try:
                        overrides = config.get("yolo", {}).get("class_conf_overrides", {}) or {}
                        min_conf_map = {str(k).lower(): float(v) for k, v in overrides.items()}
                    except Exception:
                        min_conf_map = {}

                    # Use override for this class if present; otherwise no extra floor
                    min_class_conf = min_conf_map.get(label.lower())
                    if isinstance(min_class_conf, (int, float)):
                        if conf < float(min_class_conf):
                            # Skip low-confidence detection for this class
                            continue

                    # Center computation
                    cx = float(x1 + x2) / 2.0
                    cy = float(y1 + y2) / 2.0
                    depth_m = None
                    near_depth_m = None
                    depth_valid_ratio = 0.0
                    depth_valid_pixels = 0
                    with state.lock:
                        dm_local = state.depth_map_m
                    if dm_local is not None:
                        try:
                            stats = get_bbox_depth_stats((x1, y1, x2, y2), dm_local)
                            depth_m = stats.get('median_m')
                            near_depth_m = stats.get('near_m')
                            depth_valid_ratio = float(stats.get('valid_ratio', 0.0) or 0.0)
                            depth_valid_pixels = int(stats.get('valid_pixels', 0) or 0)
                        except Exception:
                            depth_m = get_bbox_median_depth((x1, y1, x2, y2), dm_local)

                    # Compute bbox area fraction for proximity heuristics
                    try:
                        frame_area = float(max(1, int(w_full) * int(h_full)))
                        box_w = float(max(1.0, x2 - x1))
                        box_h = float(max(1.0, y2 - y1))
                        area_frac = float((box_w * box_h) / frame_area)
                    except Exception:
                        area_frac = 0.0

                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'conf': max(0.0, min(1.0, conf)),
                        'label': label,
                        'center': (int(cx), int(cy)),
                        'depth_m': depth_m,
                        'near_depth_m': near_depth_m,
                        'depth_valid_ratio': depth_valid_ratio,
                        'depth_valid_pixels': depth_valid_pixels,
                        'time': time.time(),
                        'area_frac': area_frac
                    }

                    # Debug: print detected object distance and sanity-check range
                    try:
                        if depth_m is None and near_depth_m is None:
                            logging.info(f"Det {label} conf={conf:.2f} depth=unknown bbox=({x1},{y1},{x2},{y2})")
                        else:
                            dbg_d = float((near_depth_m if near_depth_m is not None else depth_m) or 0.0)
                            logging.info(f"Det {label} conf={conf:.2f} depth={dbg_d:.2f}m bbox=({x1},{y1},{x2},{y2}) valid={depth_valid_ratio:.2f}")
                            if not (0.5 <= dbg_d <= 10.0):
                                logging.warning(f"Det {label}: depth out of expected range: {dbg_d:.2f}m")
                    except Exception:
                        pass

                    # Compute appearance descriptor (persons only)
                    try:
                        if detection['label'].lower() == 'person':
                            detection['descriptor'] = compute_descriptor(frame, (int(x1), int(y1), int(x2), int(y2)))
                        else:
                            detection['descriptor'] = None
                    except Exception:
                        detection['descriptor'] = None

                    # Assign track ID and update tracking
                    track_id = assign_track_id(detection)
                    detection['track_id'] = track_id

                    # Apply EMA smoothing for stable boxes
                    if config["yolo"]["ema_smoothing"]:
                        detection = apply_ema_smoothing(detection, track_id)

                    with state.lock:
                        # Per-track Kalman filter for depth
                        kf_depth = depth_m
                        if depth_m is not None:
                            try:
                                filt_type = str(config.get('speed_detection', {}).get('filter_type', 'kalman')).lower()
                                if filt_type == 'alpha_beta':
                                    ab = state.alpha_beta_states.get(track_id)
                                    nowt = detection['time']
                                    # Estimate dt from track history
                                    dt = 0.1
                                    if track_id in state.object_tracks and len(state.object_tracks[track_id]) >= 1:
                                        dt = max(0.01, nowt - state.object_tracks[track_id][-1][0])
                                    alpha = float(config.get('speed_detection', {}).get('alpha', 0.65))
                                    beta = float(config.get('speed_detection', {}).get('beta', 0.25))
                                    if ab is None:
                                        ab = {'x': float(depth_m), 'v': 0.0}
                                    # Predict
                                    x_pred = ab['x'] + ab['v'] * dt
                                    # Update with measurement
                                    r = float(depth_m) - x_pred
                                    x_upd = x_pred + alpha * r
                                    v_upd = ab['v'] + beta * (r / dt)
                                    state.alpha_beta_states[track_id] = {'x': x_upd, 'v': v_upd}
                                    kf_depth = float(x_upd)
                                else:
                                    st = state.kalman_states.get(track_id)
                                    if st is None:
                                        # Initialize with current measurement
                                        st = {'x': float(depth_m), 'p': 0.25}
                                    q = float(config.get('speed_detection', {}).get('kalman_q', 0.04))
                                    r = float(config.get('speed_detection', {}).get('kalman_r', 0.10))
                                    # Predict
                                    p_pred = st['p'] + q
                                    # Update
                                    k = p_pred / (p_pred + r)
                                    x_upd = st['x'] + k * (float(depth_m) - st['x'])
                                    p_upd = (1.0 - k) * p_pred
                                    state.kalman_states[track_id] = {'x': x_upd, 'p': p_upd}
                                    kf_depth = float(x_upd)
                            except Exception:
                                kf_depth = depth_m

                    if track_id not in state.object_tracks:
                        state.object_tracks[track_id] = []
                    # Store smoothed depth for speed calculation stability
                    state.object_tracks[track_id].append((detection['time'], kf_depth, detection['center']))

                    # Maintain per-track area history
                    try:
                        w = float(max(1.0, x2 - x1))
                        h = float(max(1.0, y2 - y1))
                        area = float(w * h)
                        if track_id not in state.track_areas:
                            state.track_areas[track_id] = []
                        state.track_areas[track_id].append((detection['time'], area))
                    except Exception:
                        pass

                    # Keep only recent tracking data
                    cutoff_time = time.time() - 10.0  # Keep 10 seconds of history
                    state.object_tracks[track_id] = [(t, d, c) for t, d, c in state.object_tracks[track_id] if t >= cutoff_time]
                    try:
                        state.track_areas[track_id] = [(t, a) for t, a in state.track_areas.get(track_id, []) if t >= cutoff_time]
                    except Exception:
                        pass

                    # Update appearance gallery for persons
                    try:
                        if detection['label'].lower() == 'person' and detection.get('descriptor') is not None:
                            state.appearance_gallery[track_id] = {
                                'desc': detection['descriptor'],
                                'last_seen': detection['time'],
                                'last_center': detection['center']
                            }
                            # Prune stale gallery entries
                            stale = [tid for tid, g in state.appearance_gallery.items() if (current_time - float(g.get('last_seen', 0.0))) > 12.0]
                            for tid in stale:
                                try:
                                    del state.appearance_gallery[tid]
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # Encounter memory update (passed objects suppression)
                    try:
                        update_encounter_memory_for_track(track_id)
                    except Exception:
                        pass

                    # Append detection for downstream consumers
                    # Compute approach speed and proximity confidence
                    try:
                        approach_speed = 0.0
                        if track_id in state.object_tracks:
                            approach_speed = float(calculate_object_speed(state.object_tracks[track_id], detection['time']))
                        detection['approach_speed_mps'] = approach_speed
                    except Exception:
                        detection['approach_speed_mps'] = 0.0

                    # Proximity confidence: blend closeness, approach speed, and size
                    try:
                        # Thresholds
                        critical_m = float(config.get('distances', {}).get('critical', 1.0))
                        warning_m = float(config.get('distances', {}).get('warning', max(critical_m * 2.0, 2.0)))
                        # Prefer near-depth when available
                        base_depth = detection.get('near_depth_m', detection.get('depth_m'))
                        # Normalize depth: closer -> higher
                        if base_depth is None:
                            depth_score = 0.0
                        else:
                            d = float(base_depth)
                            if d <= critical_m:
                                depth_score = 1.0
                            elif d >= warning_m:
                                depth_score = 0.0
                            else:
                                # Linear ramp between warning and critical
                                depth_score = float((warning_m - d) / max(1e-6, (warning_m - critical_m)))
                        # Speed score: cap and normalize typical approach speeds
                        sp = float(detection.get('approach_speed_mps', 0.0))
                        max_sp = float(config.get('proximity', {}).get('max_speed_for_conf_mps', 2.0))
                        speed_score = float(max(0.0, min(1.0, sp / max(1e-6, max_sp))))
                        # Area score: large boxes imply near-field
                        af = float(detection.get('area_frac', 0.0))
                        max_af = float(config.get('proximity', {}).get('area_frac_max_for_conf', 0.30))
                        area_score = float(max(0.0, min(1.0, af / max(1e-6, max_af))))
                        # Depth reliability bonus: boost confidence when ROI has enough valid mass
                        try:
                            vr = float(detection.get('depth_valid_ratio', 0.0) or 0.0)
                            vp = int(detection.get('depth_valid_pixels', 0) or 0)
                            min_ratio = float(config.get('hazard', {}).get('min_depth_valid_ratio', 0.02))
                            min_pixels = int(config.get('hazard', {}).get('min_roi_valid_pixels', 50))
                            reliability = 1.0 if (vr >= min_ratio and vp >= min_pixels) else 0.6
                        except Exception:
                            reliability = 1.0
                        # Weights
                        w_depth = float(config.get('proximity', {}).get('weight_depth', 0.50))
                        w_speed = float(config.get('proximity', {}).get('weight_speed', 0.30))
                        w_area = float(config.get('proximity', {}).get('weight_area', 0.20))
                        prox_raw = float(max(0.0, min(1.0, (w_depth * depth_score) + (w_speed * speed_score) + (w_area * area_score))))
                        prox_conf = float(max(0.0, min(1.0, prox_raw * reliability)))
                        detection['proximity_confidence'] = prox_conf
                    except Exception:
                        detection['proximity_confidence'] = 0.0

                    dets.append(detection)
        
        # Update tracker detections for smooth display
        with state.lock:
            state.tracker_detections = dets
            state.detections = dets
        # Publish detections to async pipeline queue for downstream consumers
        try:
            if getattr(state, 'pipelines_async_enabled', False):
                payload = {
                    'ts': time.time(),
                    'detections': dets,
                }
                if state.detection_queue.full():
                    _ = state.detection_queue.get_nowait()
                state.detection_queue.put_nowait(payload)
        except Exception:
            pass


def speak(text):
    try:
        # Enforce a global minimum gap before speaking directly (urgent fallback)
        try:
            gap_s = float(config.get('audio', {}).get('global_min_speak_gap_s', 0.8))
        except Exception:
            gap_s = 0.8
        try:
            last_end = float(getattr(state, '_last_audio_work', 0.0) or 0.0)
            now_gap = time.time()
            remaining = gap_s - max(0.0, (now_gap - last_end))
            if gap_s > 0.0 and remaining > 0.0:
                time.sleep(min(gap_s, remaining))
        except Exception:
            pass
        # Use blocking speak to avoid overlapping with queued audio
        tts_speak(text, block=True)
    except Exception:
        pass


# Non-blocking audio queue (FIFO) with de-duplication
# Using FIFO avoids heap comparison errors when legacy dict items appear.
audio_queue = queue.Queue()
_last_enqueued_time = 0.0

# macOS AVFoundation fallback for better audio handling (ducking/serialization)
try:
    from AVFoundation import AVSpeechSynthesizer, AVSpeechUtterance
    HAVE_AVFOUNDATION = True
except Exception:
    HAVE_AVFOUNDATION = False
try:
    from providers import get_tts_provider, get_phrasing_provider
except Exception:
    get_tts_provider = None
    get_phrasing_provider = None

# Lightweight helpers for provider health, circuit breaker, and caching
import hashlib
import os

def _ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def _tts_cache_key(text: str, provider_name: str, voice_id: str) -> str:
    try:
        norm = f"{provider_name}\n{voice_id}\n{str(text).strip()}".encode('utf-8')
        return hashlib.sha1(norm).hexdigest()
    except Exception:
        return hashlib.sha1(str(text).encode('utf-8')).hexdigest()

def _tts_cache_path(key: str) -> str:
    try:
        base = getattr(state, '_tts_cache_dir', os.path.join(os.getcwd(), 'cache', 'tts'))
    except Exception:
        base = os.path.join(os.getcwd(), 'cache', 'tts')
    _ensure_dir(base)
    return os.path.join(base, f"{key}.mp3")

def _prune_tts_cache(max_items: int):
    try:
        base = getattr(state, '_tts_cache_dir', os.path.join(os.getcwd(), 'cache', 'tts'))
        if not os.path.isdir(base):
            return
        files = [os.path.join(base, f) for f in os.listdir(base) if f.endswith('.mp3')]
        if len(files) <= max_items:
            return
        files.sort(key=lambda p: os.path.getmtime(p))
        to_delete = files[:-max_items]
        for p in to_delete:
            try:
                os.remove(p)
            except Exception:
                pass
    except Exception:
        pass

def init_tts_backend():
    # Initialize TTS backend and circuit breakers; offline/local only
    try:
        with state.lock:
            state._ext_tts = None
        # Phrasing provider may remain as configured; leave as-is
        if get_phrasing_provider:
            with state.lock:
                state._phrasing_provider = get_phrasing_provider(config)
    except Exception:
        pass
    # Initialize circuit breaker states and cache configuration
    try:
        with state.lock:
            # TTS circuit breaker
            state._tts_cb = state._tts_cb if hasattr(state, '_tts_cb') else {}
            state._tts_cb.setdefault('fail_count', 0)
            state._tts_cb.setdefault('open', False)
            state._tts_cb.setdefault('opened_at', 0.0)
            state._tts_cb.setdefault('last_announce', 0.0)
            state._tts_cb['threshold'] = int(config.get('audio', {}).get('tts_cb_threshold', 50))  # Significantly increased threshold
            state._tts_cb['cooldown_s'] = float(config.get('audio', {}).get('tts_cb_cooldown_s', 0.5))  # Drastically reduced cooldown
            # Phrasing circuit breaker
            state._phr_cb = state._phr_cb if hasattr(state, '_phr_cb') else {}
            state._phr_cb.setdefault('fail_count', 0)
            state._phr_cb.setdefault('open', False)
            state._phr_cb.setdefault('opened_at', 0.0)
            state._phr_cb.setdefault('last_announce', 0.0)
            ann = config.get('announcements', {})
            state._phr_cb['threshold'] = int(ann.get('phrasing_cb_threshold', 50))  # Significantly increased threshold
            state._phr_cb['cooldown_s'] = float(ann.get('phrasing_cb_cooldown_s', 0.5))  # Drastically reduced cooldown
            # TTS cache configuration
            state._tts_cache_dir = state._tts_cache_dir if hasattr(state, '_tts_cache_dir') else os.path.join(os.getcwd(), 'cache', 'tts')
            _ensure_dir(state._tts_cache_dir)
            state._tts_cache_max_items = int(config.get('audio', {}).get('tts_cache_max_items', 100))
    except Exception:
        pass
    # Select speech backend based on config; default local
    try:
        provider = str(config.get('audio', {}).get('tts_provider', 'local') or 'local').strip().lower()
    except Exception:
        provider = 'local'
    try:
        with state.lock:
            # Use local system voice (say/AVFoundation) by default
            state._speech_backend = getattr(state, '_speech_backend', 'say')
    except Exception:
        pass
    if HAVE_AVFOUNDATION:
        try:
            synth = AVSpeechSynthesizer.alloc().init()
            with state.lock:
                state._av_synth = synth
                state._speech_backend = 'avfoundation'
        except Exception:
            with state.lock:
                state._speech_backend = 'say'
    else:
        with state.lock:
            state._speech_backend = 'say'
    # Enforce local-only speech backend regardless of provider value
    try:
        with state.lock:
            if getattr(state, '_speech_backend', 'say') not in ('avfoundation', 'say'):
                state._speech_backend = 'say'
    except Exception:
        pass

def _get_mac_voice() -> str:
    try:
        v = str(config.get('audio', {}).get('mac_voice', '') or '').strip()
        return v
    except Exception:
        return ''

def _map_rate_to_wpm(rate: float) -> int:
    try:
        r = float(rate)
    except Exception:
        r = 0.5
    # Clamp 0..1 and map to reasonable WPM range
    r = max(0.0, min(1.0, r))
    try:
        base = int(config.get('audio', {}).get('mac_wpm_base', 160))
        span = int(config.get('audio', {}).get('mac_wpm_span', 100))
    except Exception:
        base = 160
        span = 100
    wpm = int(base + span * r)
    return max(80, min(400, wpm))

def mac_say(text: str, voice: str = None, rate: float = None, block: bool = True):
    try:
        args = ['say']
        v = voice if isinstance(voice, str) and voice.strip() else _get_mac_voice()
        if v:
            args += ['-v', v]
        if rate is not None:
            try:
                wpm = _map_rate_to_wpm(rate)
                args += ['-r', str(int(wpm))]
            except Exception:
                pass
        args += [str(text)]
        if block:
            subprocess.run(args)
        else:
            subprocess.Popen(args)
    except Exception:
        try:
            # Minimal fallback
            if block:
                subprocess.run(['say', str(text)])
            else:
                subprocess.Popen(['say', str(text)])
        except Exception:
            pass

def tts_speak(message, block=True):
    backend = getattr(state, '_speech_backend', 'say')
    # Defensive: if AVFoundation volume is near-zero, ensure urgent/warning remains audible
    try:
        vol = float(getattr(state, '_speech_volume', 1.0) or 1.0)
    except Exception:
        vol = 1.0
    if backend == 'avfoundation' and state._av_synth is not None:
        try:
            utter = AVSpeechUtterance.speechUtteranceWithString_(message)
            try:
                # Safe defaults
                utter.setRate_(state._speech_rate)
            except Exception:
                pass
            try:
                # Clamp volume to avoid accidental silence; keep minimum audible level
                v = float(getattr(state, '_speech_volume', 1.0) or 1.0)
                v = max(0.2, min(1.0, v))
                utter.setVolume_(v)
            except Exception:
                pass
            # Apply language/voice if configured
            try:
                from AVFoundation import AVSpeechSynthesisVoice
                lang = str(config.get('audio', {}).get('tts_language', '') or '').strip()
                voice_hint = str(config.get('audio', {}).get('voice_id', '') or '').strip()
                voice_obj = None
                if lang:
                    try:
                        voice_obj = AVSpeechSynthesisVoice.voiceWithLanguage_(lang)
                    except Exception:
                        voice_obj = None
                # If no language provided or voice lookup by id requested, try matching by name
                if (voice_obj is None) and voice_hint:
                    try:
                        # Iterate available voices to find name match
                        voices = AVSpeechSynthesisVoice.speechVoices()
                        for v in voices:
                            try:
                                name = str(getattr(v, 'name', '') or '')
                            except Exception:
                                name = ''
                            if name and (name.lower() == voice_hint.lower()):
                                voice_obj = v
                                break
                    except Exception:
                        voice_obj = None
                if voice_obj is not None:
                    try:
                        utter.setVoice_(voice_obj)
                    except Exception:
                        pass
            except Exception:
                pass
            # If internal volume is essentially muted, ensure emergencies still speak via 'say'
            try:
                ml = str(message).lower()
                is_emergency = ('urgent' in ml) or ('warning' in ml) or ('caution' in ml) or ('alert' in ml) or ('critical' in ml)
            except Exception:
                is_emergency = False
            if vol <= 0.05 and is_emergency:
                try:
                    mac_say(message, voice=_get_mac_voice(), rate=getattr(state, '_speech_rate', None), block=block)
                    # Telemetry: emergency say fallback invoked
                    try:
                        with state.lock:
                            state.telemetry['events'].append({'time': time.time(), 'type': 'tts_emergency_fallback_say'})
                    except Exception:
                        pass
                    return
                except Exception:
                    pass
            state._av_synth.speakUtterance_(utter)
            if block:
                # Serialize until speaking finishes
                while state._av_synth.isSpeaking() and state.running:
                    time.sleep(0.05)
        except Exception:
            # Fallback to 'say'
            try:
                mac_say(message, voice=_get_mac_voice(), rate=getattr(state, '_speech_rate', None), block=block)
            except Exception:
                pass
    else:
        try:
            mac_say(message, voice=_get_mac_voice(), rate=getattr(state, '_speech_rate', None), block=block)
        except Exception:
            pass

def _provider_monitor_worker():
    """Periodically check external providers and auto-close circuit breaker after cooldown."""
    interval = 30.0
    while state.running:
        try:
            # TTS provider health and half-open probe
            try:
                prov = getattr(state, '_ext_tts', None)
                cb = getattr(state, '_tts_cb', {'open': False, 'cooldown_s': 60.0})
                open_flag = bool(cb.get('open', False))
                cooldown_s = float(cb.get('cooldown_s', 60.0))
                opened_at = float(cb.get('opened_at', 0.0) or 0.0)
                if prov is not None:
                    # If breaker open and cooldown passed, try half-open test
                    if open_flag and (time.time() - opened_at) >= cooldown_s:
                        try:
                            test = prov.synthesize('ok')
                            if test:
                                # Close breaker and announce restored
                                cb['open'] = False
                                cb['fail_count'] = 0
                                cb['opened_at'] = 0.0
                                last_announce = float(cb.get('last_announce', 0.0) or 0.0)
                                cb['last_announce'] = time.time()
                                state._tts_cb = cb
                                try:
                                    speak_status('External TTS restored')
                                except Exception:
                                    pass
                            else:
                                # Extend cooldown window
                                cb['opened_at'] = time.time()
                                state._tts_cb = cb
                        except Exception:
                            cb['opened_at'] = time.time()
                            state._tts_cb = cb
            except Exception:
                pass

            # Phrasing provider health and half-open probe
            try:
                phr = getattr(state, '_phrasing_provider', None)
                pcb = getattr(state, '_phr_cb', {'open': False, 'cooldown_s': 60.0})
                open_flag = bool(pcb.get('open', False))
                cooldown_s = float(pcb.get('cooldown_s', 60.0))
                opened_at = float(pcb.get('opened_at', 0.0) or 0.0)
                if phr is not None:
                    if open_flag and (time.time() - opened_at) >= cooldown_s:
                        try:
                            test = phr.phrase({'raw': 'ok', 'urgency': 'info'})
                            if isinstance(test, str) and len(test.strip()) > 0:
                                pcb['open'] = False
                                pcb['fail_count'] = 0
                                pcb['opened_at'] = 0.0
                                last_announce = float(pcb.get('last_announce', 0.0) or 0.0)
                                pcb['last_announce'] = time.time()
                                state._phr_cb = pcb
                                try:
                                    speak_status('AI phrasing restored')
                                except Exception:
                                    pass
                            else:
                                pcb['opened_at'] = time.time()
                                state._phr_cb = pcb
                        except Exception:
                            pcb['opened_at'] = time.time()
                            state._phr_cb = pcb
            except Exception:
                pass
        except Exception:
            pass
        time.sleep(interval)

# Phrase variants and helpers
PHRASE_VARIANTS = {
    'clear_path': [
        'Clear ahead.',
        'Path looks clear.',
        'You have a clear path.',
        'Way is clear.'
    ],
    'area_clear': [
        'Area looks clear.',
        'All clear ahead.',
        'No obstacles ahead.',
        "You're good to go."
    ],
    'provider_switch_google': [
        'Google Maps selected.',
        'Using Google Maps.',
        'Switched to Google Maps.'
    ],
    'provider_switch_apple': [
        'Apple Maps selected.',
        'Using Apple Maps.',
        'Switched to Apple Maps.'
    ],
    'nav_mode_walk': [
        'Walking mode set.',
        'Navigation set to walking.',
        'Walking mode enabled.'
    ],
    'nav_mode_drive': [
        'Driving mode set.',
        'Navigation set to driving.',
        'Driving mode enabled.'
    ],
    'crowd_on': [
        'Crowd mode on.',
        'Silent crowd engaged.',
        'Crowd awareness enabled.'
    ],
    'crowd_off': [
        'Crowd mode off.',
        'Silent crowd disengaged.',
        'Crowd awareness disabled.'
    ],
    'focus_center': [
        'Focusing center.',
        'Center focus on.',
        'Center prioritized.'
    ],
    'quieter_ack': [
        'Okay, quieter.',
        "I'll lower my volume.",
        'I will be quieter.',
        'Quieter mode on.'
    ],
    'moderate_approach': [
        'Caution.',
        'Careful.',
        'Watch out.',
        'Attention.'
    ],
    'slow_approach': [
        'Heads up.',
        'Careful.',
        'Note.',
        'Attention.'
    ],
    'warning_approach': [
        'Warning!',
        'Caution!',
        'Alert!',
        'Be careful!'
    ],
    'urgent_approach': [
        'Urgent!',
        'Immediate danger!',
        'Danger!',
        'Act now!'
    ],
    'caution': [
        'Caution.',
        'Heads up.',
        'Careful.',
        'Watch out.',
        'Attention.',
        'Note.'
    ],
    'warning_critical': [
        'Warning!',
        'Alert!',
        'Caution!'
    ],
    'urgent_critical': [
        'Urgent!',
        'Immediate danger!',
        'Danger!'
    ],
    'nearby': [
        'Nearby.',
        'Close by.',
        'In proximity.'
    ]
}

_last_phrase_variant = {}


def _apply_phrase_variant(message):
    """Return a possibly varied phrasing for common status lines.

    For approach-tier messages with dynamic content, vary the leading
    prefix (e.g., "Caution.", "Heads up.") while preserving the tail.
    """
    try:
        msg_l = str(message).lower().strip()
        key = None
        if 'clear path' in msg_l:
            key = 'clear_path'
        elif 'area clear' in msg_l:
            key = 'area_clear'
        elif 'switched to google maps' in msg_l:
            key = 'provider_switch_google'
        elif 'switched to apple maps' in msg_l:
            key = 'provider_switch_apple'
        elif 'navigation mode set to walking' in msg_l:
            key = 'nav_mode_walk'
        elif 'navigation mode set to driving' in msg_l:
            key = 'nav_mode_drive'
        elif 'crowd mode enabled' in msg_l:
            key = 'crowd_on'
        elif 'crowd mode disabled' in msg_l:
            key = 'crowd_off'
        elif 'focusing center' in msg_l:
            key = 'focus_center'
        elif 'be quieter' in msg_l or "i will be quieter" in msg_l or 'okay, i will be quieter' in msg_l:
            key = 'quieter_ack'
        # Detect approach tiers for variant prefixing
        elif ('heads up.' in msg_l) and ('slowly approaching' in msg_l):
            key = 'slow_approach'
        elif ('caution.' in msg_l) and ('approaching' in msg_l) and ('slowly' not in msg_l) and ('fast' not in msg_l) and ('urgent' not in msg_l):
            key = 'moderate_approach'
        elif ('warning!' in msg_l) and ('approaching' in msg_l):
            key = 'warning_approach'
        elif ('urgent!' in msg_l) and ('approaching' in msg_l):
            key = 'urgent_approach'
        elif ('warning!' in msg_l) and ('very close' in msg_l):
            key = 'warning_critical'
        elif ('urgent!' in msg_l) and ('very close' in msg_l):
            key = 'urgent_critical'
        elif ('nearby' in msg_l):
            key = 'nearby'
        # Caution lines: static or dynamic with a prefix and tail
        elif msg_l in ('caution.', 'caution!', 'careful.', 'careful!', 'watch out.', 'attention.', 'attention!', 'heads up.', 'note.'):
            key = 'caution'
        elif msg_l.startswith(('caution', 'heads up', 'attention', 'careful', 'watch out')):
            key = 'caution'

        if not key:
            return message

        variants = PHRASE_VARIANTS.get(key)
        if not variants:
            return message

        last = _last_phrase_variant.get(key)
        # Simple rotation to avoid repeating last variant
        if last is None:
            choice = variants[0]
        else:
            try:
                idx = variants.index(last)
                choice = variants[(idx + 1) % len(variants)]
            except ValueError:
                choice = variants[0]
        _last_phrase_variant[key] = choice

        # For dynamic approach or caution messages, vary the prefix and keep tail intact
        if key in ('moderate_approach', 'slow_approach', 'caution', 'warning_approach', 'urgent_approach', 'warning_critical', 'urgent_critical', 'nearby'):
            try:
                # Split at first punctuation to isolate the leading prefix
                sep_idx = None
                for ch in ('.', '!', '?'):
                    i = message.find(ch)
                    if i != -1:
                        sep_idx = i
                        break
                if sep_idx is not None:
                    head = message[:sep_idx+1]
                    tail = message[sep_idx+1:].lstrip()
                    tail_plain = tail.strip().strip('.!?:').lower()
                    head_plain = head.strip().strip('.!?:').lower()
                    caution_tails = (
                        'caution', 'heads up', 'attention', 'careful', 'watch out', 'note', 'use caution'
                    )
                    # If the tail is just a generic caution prompt, replace the tail with the variant
                    # and preserve the informative head (e.g., "Fast object approaching ahead.")
                    if any(tail_plain.startswith(ct) for ct in caution_tails):
                        return f"{head} {choice}"
                    # If the head itself is a generic caution phrase, replace the head and keep the tail
                    if any(head_plain.startswith(ct) for ct in caution_tails):
                        return f"{choice} {tail}"
                    # Otherwise, keep the message as-is to avoid losing context
                    return message
                else:
                    # No obvious prefix; just return the rotated choice
                    return choice
            except Exception:
                return message

        return choice
    except Exception:
        return message


def _infer_earcon_for_message(message):
    """Infer earcon level and side from message content; returns (level, side) or (None, None)."""
    try:
        msg_l = str(message).lower()
        # Avoid duplicating hazard earcons; those are handled in the announcer
        if 'urgent' in msg_l or 'warning!' in msg_l:
            return (None, None)
        level = None
        side = 'center'
        if 'left' in msg_l:
            side = 'left'
        elif 'right' in msg_l:
            side = 'right'
        if ('clear path' in msg_l) or ('area clear' in msg_l):
            level = 'success'
        elif ('switched to google maps' in msg_l) or ('switched to apple maps' in msg_l) or ('navigation mode set to' in msg_l):
            level = 'success'
        elif ('starting directions' in msg_l) or ('searching for' in msg_l):
            level = 'info'
        elif ('error' in msg_l) or ('failed' in msg_l):
            level = 'error'
        elif ('crowd mode enabled' in msg_l) or ('crowd mode disabled' in msg_l):
            level = 'info'
        elif ('voice mode on' in msg_l) or ('voice mode off' in msg_l):
            level = 'info'
        return (level, side)
    except Exception:
        return (None, None)

def _parse_depth_hint_from_message(message: str) -> float:
    """Parse an approximate depth hint in meters from a message string.
    Looks for patterns like 'around X meters', 'within X meters', or 'X m'.
    Returns 0.0 if not found or if unit isn't meters.
    """
    try:
        m = str(message or '').lower()
        import re
        # Prefer explicit meters units
        match = re.search(r"(\d+(?:\.\d+)?)\s*(meters|meter|m)\b", m)
        if match:
            val = float(match.group(1))
            unit = match.group(2)
            # Treat 'm' and 'meters' as meters; ignore if seconds
            if unit in ('m','meter','meters'):
                return max(0.0, val)
        return 0.0
    except Exception:
        return 0.0


def _format_with_tone_and_verbosity(text: str, priority_hint: int = 2) -> str:
    try:
        ann = config.get('announcements', {})
        tone = str(ann.get('tone_style', 'supportive') or 'neutral').lower()
        verbosity = str(ann.get('verbosity_level', 'standard') or 'standard').lower()
        msg = str(text or '').strip()
        # Apply verbosity: brief trims long informational messages; detailed leaves intact
        if verbosity == 'brief' and priority_hint >= 2:
            if len(msg) > 90:
                # Trim to ~90 chars at a natural boundary
                cut = msg[:90]
                # Prefer cutting at last comma or period
                for sep in ('.', ',', ';'):
                    idx = cut.rfind(sep)
                    if idx >= 40:
                        cut = cut[:idx+1]
                        break
                msg = cut.strip()
        # Apply supportive tone only for non-urgent/info messages
        if tone == 'supportive' and priority_hint >= 2:
            try:
                prefixes = list(ann.get('supportive_prefixes', [])) or []
                suffixes = list(ann.get('supportive_suffixes', [])) or []
                # Avoid double-prefix if message already starts with an alert keyword
                low = msg.lower()
                starts_alert = any(low.startswith(p) for p in ('urgent', 'warning', 'alert', 'caution', 'heads up'))
                if not starts_alert and prefixes:
                    pref = random.choice(prefixes)
                    msg = f"{pref} {msg}".strip()
                if suffixes:
                    # Avoid adding a suffix if message already ends with punctuation; add softly
                    suf = random.choice(suffixes)
                    if not msg.endswith(suf):
                        msg = f"{msg} {suf}".strip()
            except Exception:
                pass
        return msg
    except Exception:
        return str(text or '')

def _audio_qsize():
    """Return current audio queue size, handling async mode if enabled.

    Defined early to avoid NameError when used by speak_status during
    module initialization, before later helpers are parsed.
    """
    try:
        if bool(config.get('audio', {}).get('async_mode', False)) and getattr(state, '_async_audio_queue', None) is not None:
            return state._async_audio_queue.qsize()
        return audio_queue.qsize()
    except Exception:
        try:
            return audio_queue.qsize()
        except Exception:
            return 0

def speak_status(message, priority=None):
    """Queue a message for speech with alert-aware segmentation and rate limits.

    Applies tone formatting, dedup windows, smart verbosity suppression, and
    segmenting of alert prefixes (e.g., "Warning!") from the tail for natural
    pacing. Uses a priority queue and telemetry tracking for announcements.

    Args:
        message: Text to speak. Alert keywords influence priority and pacing.

    Returns:
        None. Enqueues segments into `audio_queue` or directly speaks if needed.
    """
    global _last_enqueued_time
    now = time.time()
    # Apply phrasing variants for common lines
    try:
        varied_message = _apply_phrase_variant(message)
    except Exception:
        varied_message = message
    # Early priority hint from raw content for formatting
    try:
        ml_tmp = str(varied_message).lower()
        hint_urgent = ('urgent!' in ml_tmp) or ('critical' in ml_tmp) or (('warning!' in ml_tmp) and ('very close' in ml_tmp))
        hint_warning = ('warning!' in ml_tmp) or ('caution!' in ml_tmp)
        priority_hint = 0 if hint_urgent else (1 if hint_warning else 2)
    except Exception:
        priority_hint = 2
    # Apply tone style and verbosity formatting
    try:
        varied_message = _format_with_tone_and_verbosity(varied_message, priority_hint)
    except Exception:
        pass
    # Alert-type filtering based on user preferences (urgency and type)
    try:
        prefs = _get_active_alert_prefs()
    except Exception:
        prefs = {
            'urgency_enabled': ['critical', 'warning', 'info'],
            'types_enabled': {'critical': True, 'warning': True, 'clear': True, 'summary': True, 'system': True},
            'sensitivity_scale': 1.0,
            'mute_all': False
        }
    try:
        # During automated tests, bypass global mute to ensure deterministic behavior
        # Pytest sets PYTEST_CURRENT_TEST; also allow THIRDEYE_TEST_MODE override
        test_mode = False
        try:
            import os as _os
            test_mode = bool(_os.environ.get('PYTEST_CURRENT_TEST')) or bool(_os.environ.get('THIRDEYE_TEST_MODE'))
        except Exception:
            test_mode = False
        if bool(prefs.get('mute_all', False)) and (not test_mode):
            return
    except Exception:
        pass
    # Early urgency/type classification for filtering
    try:
        ml_tmp = str(varied_message).lower()
        urgent_allowed = ('critical' in prefs.get('urgency_enabled', [])) or ('urgent' in prefs.get('urgency_enabled', []))
        warning_allowed = ('warning' in prefs.get('urgency_enabled', [])) or ('caution' in prefs.get('urgency_enabled', []))
        info_allowed = ('info' in prefs.get('urgency_enabled', [])) or ('clear' in prefs.get('urgency_enabled', []))
        # Determine rough event type from keywords
        etype_guess = 'system'
        if ('urgent' in ml_tmp) or ('critical' in ml_tmp):
            etype_guess = 'critical'
        elif ('warning' in ml_tmp) or ('caution' in ml_tmp) or ('alert' in ml_tmp):
            etype_guess = 'warning'
        elif ('clear' in ml_tmp) or ('all clear' in ml_tmp) or ('safe' in ml_tmp):
            etype_guess = 'clear'
        elif ('summary' in ml_tmp) or ('report' in ml_tmp) or ('status' in ml_tmp) or ('overview' in ml_tmp):
            etype_guess = 'summary'
        # Urgency gate
        if ('urgent' in ml_tmp or 'critical' in ml_tmp) and not urgent_allowed:
            return
        if ('warning' in ml_tmp or 'caution' in ml_tmp or 'alert' in ml_tmp) and not warning_allowed:
            return
        if (etype_guess in ('clear', 'summary', 'system')) and not info_allowed:
            return
        # Type gate
        types_enabled = prefs.get('types_enabled', {})
        if isinstance(types_enabled, dict) and (types_enabled.get(etype_guess) is False):
            return
    except Exception:
        pass
    # De-duplicate repeated messages; allow repeats primarily for warnings/critical
    try:
        ml_local = str(varied_message).lower()
        is_urgent_local = ('urgent!' in ml_local) or ('critical' in ml_local) or (('warning!' in ml_local) and ('very close' in ml_local))
        is_warning_local = ('warning!' in ml_local) or ('caution!' in ml_local)
        dedup_win = float(config.get('audio', {}).get('dedup_window_s', 2.0))
        urgent_win = max(0.3, 0.4 * dedup_win)
        info_win = float(config.get('audio', {}).get('info_repeat_window_s', 12.0))
        hist = getattr(state, '_recent_messages', {})
        last = float(hist.get(ml_local, 0.0) or 0.0)
        # For non-hazard info messages, enforce a longer repeat window
        if (not is_urgent_local and not is_warning_local) and ((now - last) < info_win):
            return
        if (is_urgent_local or is_warning_local) and ((now - last) < urgent_win):
            return
        hist[ml_local] = now
        state._recent_messages = hist
    except Exception:
        pass
    # Note: speech rate now applied per message segment in audio_worker
    # Opportunistic earcon for non-hazard status updates
    try:
        lvl, side = _infer_earcon_for_message(varied_message)
        if lvl and bool(getattr(state, '_earcons_enabled', True)):
            # Respect global hush for non-critical earcons
            if (now >= getattr(state, '_global_silence_until', 0.0)):
                try:
                    play_spatial_earcon(lvl, side=side)
                except Exception:
                    play_earcon(lvl, side=side)
    except Exception:
        pass
    # Determine priority from message content
    ml = str(varied_message).lower()
    is_urgent = ('urgent!' in ml) or ('critical' in ml) or (('warning!' in ml) and ('very close' in ml))
    is_warning = ('warning!' in ml) or ('caution!' in ml)
    priority_val = 0 if is_urgent else (1 if is_warning else 2)
    # Allow explicit override from caller
    try:
        if priority is not None:
            priority_val = int(priority)
    except Exception:
        pass

    # Optional AI phrasing rewrite for non-urgent/info messages
    try:
        phr = getattr(state, '_phrasing_provider', None)
        pcb = getattr(state, '_phr_cb', {'open': False, 'threshold': 3})
        phr_cb_open = bool(pcb.get('open', False))
        if phr and priority >= 2 and not phr_cb_open:
            # Build minimal context
            lvl, side_hint = _infer_earcon_for_message(varied_message)
            ctx = {
                'raw': varied_message,
                'urgency': 'info',
                'sides': [side_hint] if side_hint else [],
                'labels': [],
                'corridor_hint': getattr(state, 'last_corridor_side', ''),
                'travel_mode': getattr(state, 'travel_mode', 'WALKING')
            }
            rewritten = None
            try:
                rewritten = phr.phrase(ctx)
            except Exception:
                rewritten = None
            if rewritten and isinstance(rewritten, str) and len(rewritten.strip()) >= 3:
                varied_message = rewritten.strip()
                ml = varied_message.lower()
                is_urgent = ('urgent!' in ml) or ('critical' in ml) or (('warning!' in ml) and ('very close' in ml))
                is_warning = ('warning!' in ml) or ('caution!' in ml)
                priority_val = 0 if is_urgent else (1 if is_warning else 2)
                try:
                    state.telemetry['events'].append({'time': time.time(), 'type': 'ai_phrase_used'})
                except Exception:
                    pass
            else:
                # Phrasing provider failure; apply circuit breaker accounting
                try:
                    with state.lock:
                        pcb['fail_count'] = int(pcb.get('fail_count', 0) or 0) + 1
                        thr = int(pcb.get('threshold', 3))
                        if pcb['fail_count'] >= thr and not bool(pcb.get('open', False)):
                            pcb['open'] = True
                            pcb['opened_at'] = time.time()
                            last_announce = float(pcb.get('last_announce', 0.0) or 0.0)
                            if (time.time() - last_announce) > 10.0:
                                pcb['last_announce'] = time.time()
                                try:
                                    speak_status('AI phrasing unavailable; using built-in phrasing')
                                except Exception:
                                    pass
                            state._phr_cb = pcb
                except Exception:
                    pass
    except Exception:
        pass

    # Smart verbosity: previously dropped non-critical chatter under load.
    # Full fix: add force_speak_all override to avoid silent system.
    try:
        smart_verbosity = bool(config.get('announcements', {}).get('smart_verbosity', True))
        force_all = bool(config.get('audio', {}).get('force_speak_all', True))
        if smart_verbosity and (not force_all) and priority >= 2:
            # Consider FPS and recent queue latency
            last_fps = float(getattr(state, 'last_fps', 0.0) or 0.0)
            fps_thresh = float(config.get('audio', {}).get('smart_fps_threshold', 12.0))
            lat_hist = list(state.performance_stats.get('tts_queue_latencies', []))
            recent_lat = float(np.mean(lat_hist[-10:])) if lat_hist[-10:] else 0.0
            lat_thresh = float(config.get('audio', {}).get('smart_queue_latency_s', 0.7))
            # Under load, instead of dropping, gently rate-limit via an internal flag
            if (last_fps > 0 and last_fps < fps_thresh) or (recent_lat > lat_thresh):
                # Apply a stricter transient rate limit for info messages
                rate_limit = max(0.8, float(config.get('audio', {}).get('enqueue_rate_limit_s', 0.5)))
                if (now - _last_enqueued_time) < rate_limit:
                    return
    except Exception:
        pass

    # Simple enqueue rate limit to avoid overlap bursts (bypass for urgent)
    rate_limit = float(config.get('audio', {}).get('enqueue_rate_limit_s', 0.5))
    try:
        if getattr(state, 'realtime_mode', False):
            if is_urgent or is_warning:
                rate_limit = min(rate_limit, 0.15)
    except Exception:
        pass
    # For segmented messages, we will enqueue segments together and update _last_enqueued_time after
    will_segment = False
    try:
        ml_full = str(varied_message).strip()
        ml_lower = ml_full.lower()
        # Detect known alert-style prefixes
        prefixes = ('urgent!', 'warning!', 'alert!', 'caution!', 'caution.', 'heads up.', 'heads up!')
        for p in prefixes:
            if ml_lower.startswith(p):
                will_segment = True
                break
    except Exception:
        will_segment = False
    if (not is_urgent) and (not will_segment) and (now - _last_enqueued_time < rate_limit):
        return
    # Cap queue size to prevent backlog
    # Increase queue capacity under force_speak_all to avoid dropping enqueues
    max_q = int(config.get('audio', {}).get('max_queue_size', 4))
    try:
        if bool(config.get('audio', {}).get('force_speak_all', True)):
            max_q = max(max_q, 8)
    except Exception:
        pass
    try:
        if getattr(state, 'realtime_mode', False):
            if is_urgent or is_warning:
                max_q = max(max_q, 3)
    except Exception:
        pass
    # Assign log level for downstream logging
    try:
        is_hint = ('heads up' in ml) or ('tip:' in ml) or ('guidance' in ml)
        log_level = 'critical' if is_urgent else ('warning' if is_warning else ('hint' if is_hint else 'info'))
    except Exception:
        log_level = 'info'

    if _audio_qsize() >= max_q:
        # Allow urgent announcements via safe preemption and enqueue
        if is_urgent:
            try:
                # Preempt: clear queues so urgent can run next
                if bool(config.get('audio', {}).get('urgent_preempt', True)):
                    try:
                        while audio_queue.qsize() > 0:
                            _ = audio_queue.get_nowait()
                            audio_queue.task_done()
                        try:
                            if getattr(state, '_async_audio_queue', None) is not None:
                                while state._async_audio_queue.qsize() > 0:
                                    _ = state._async_audio_queue.get_nowait()
                                    state._async_audio_queue.task_done()
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # Signal worker to stop AVFoundation immediately if speaking
                    try:
                        with state.lock:
                            state._preempt_now_ts = time.time()
                    except Exception:
                        pass
                # Enqueue urgent payload at highest priority so worker speaks it next
                payload_u = {'msg': varied_message, 'enqueue_ts': now, 'urgent': True, 'rate': float(config.get('audio', {}).get('speech_rate_default', 0.5)), 'level': 'critical'}
                try:
                    if bool(config.get('audio', {}).get('async_mode', False)) and getattr(state, '_async_audio_loop', None) is not None and getattr(state, '_async_audio_queue', None) is not None:
                        asyncio.run_coroutine_threadsafe(state._async_audio_queue.put((0, now, payload_u)), state._async_audio_loop)
                    else:
                        audio_queue.put_nowait((0, now, payload_u))
                except Exception:
                    audio_queue.put_nowait((0, now, payload_u))
                _last_enqueued_time = now
            except Exception:
                pass
        return
    try:
        # Enqueue with priority and timestamp for latency metrics
        if will_segment:
            # Split into prefix and tail so only alert part is fast
            try:
                # Find first space after prefix; prefix ends at first whitespace following punctuation
                # Handle typical forms like "URGENT! ..." / "Warning! ..." / "Caution. ..." / "Heads up. ..."
                # We split on the first space after the leading keyword and punctuation
                seg_msg = str(varied_message)
                lower = seg_msg.lower()
                split_idx = None
                for p in ('urgent!', 'warning!', 'alert!', 'caution!', 'caution.', 'heads up.', 'heads up!'):
                    if lower.startswith(p):
                        # position right after prefix
                        after = len(p)
                        # skip any spaces
                        while after < len(seg_msg) and seg_msg[after].isspace():
                            after += 1
                        split_idx = after
                        break
                if split_idx is None or split_idx >= len(seg_msg):
                    # No tail found; enqueue as single message
                    payload = {'msg': varied_message, 'enqueue_ts': now, 'urgent': bool(is_urgent), 'rate': float(config.get('audio', {}).get('speech_rate_default', 0.5)), 'level': log_level}
                    try:
                        if bool(config.get('audio', {}).get('async_mode', False)) and getattr(state, '_async_audio_loop', None) is not None and getattr(state, '_async_audio_queue', None) is not None:
                            asyncio.run_coroutine_threadsafe(state._async_audio_queue.put((priority_val, now, payload)), state._async_audio_loop)
                        else:
                            audio_queue.put_nowait((priority_val, now, payload))
                    except Exception:
                        audio_queue.put_nowait((priority_val, now, payload))
                else:
                    prefix_text = seg_msg[:split_idx].strip()
                    tail_text = seg_msg[split_idx:].strip()
                    # Choose rates: faster for alert prefix; default for tail
                    prefix_rate = 0.62 if is_urgent else 0.56
                    tail_rate = float(config.get('audio', {}).get('speech_rate_default', 0.5))
                    # Enqueue prefix with urgent flag to bypass rate limit/preempt if needed
                    payload_p = {'msg': prefix_text, 'enqueue_ts': now, 'urgent': bool(is_urgent or is_warning), 'rate': prefix_rate, 'level': log_level}
                    try:
                        if bool(config.get('audio', {}).get('async_mode', False)) and getattr(state, '_async_audio_loop', None) is not None and getattr(state, '_async_audio_queue', None) is not None:
                            asyncio.run_coroutine_threadsafe(state._async_audio_queue.put((priority_val, now, payload_p)), state._async_audio_loop)
                        else:
                            audio_queue.put_nowait((priority_val, now, payload_p))
                    except Exception:
                        audio_queue.put_nowait((priority_val, now, payload_p))
                    # Enqueue tail immediately after with non-urgent flag at normal speed
                    payload_t = {'msg': tail_text, 'enqueue_ts': now, 'urgent': False, 'rate': tail_rate, 'level': 'info'}
                    try:
                        if bool(config.get('audio', {}).get('async_mode', False)) and getattr(state, '_async_audio_loop', None) is not None and getattr(state, '_async_audio_queue', None) is not None:
                            asyncio.run_coroutine_threadsafe(state._async_audio_queue.put((priority_val, now, payload_t)), state._async_audio_loop)
                        else:
                            audio_queue.put_nowait((priority_val, now, payload_t))
                    except Exception:
                        audio_queue.put_nowait((priority_val, now, payload_t))
                _last_enqueued_time = now
            except Exception:
                # Fallback to single message if segmentation fails
                payload_f = {'msg': varied_message, 'enqueue_ts': now, 'urgent': bool(is_urgent), 'rate': float(config.get('audio', {}).get('speech_rate_default', 0.5)), 'level': log_level}
                try:
                    if bool(config.get('audio', {}).get('async_mode', False)) and getattr(state, '_async_audio_loop', None) is not None and getattr(state, '_async_audio_queue', None) is not None:
                        asyncio.run_coroutine_threadsafe(state._async_audio_queue.put((priority_val, now, payload_f)), state._async_audio_loop)
                    else:
                        audio_queue.put_nowait((priority_val, now, payload_f))
                except Exception:
                    audio_queue.put_nowait((priority_val, now, payload_f))
                _last_enqueued_time = now
        else:
            payload_e = {'msg': varied_message, 'enqueue_ts': now, 'urgent': bool(is_urgent), 'rate': float(config.get('audio', {}).get('speech_rate_default', 0.5)), 'level': log_level}
            try:
                if bool(config.get('audio', {}).get('async_mode', False)) and getattr(state, '_async_audio_loop', None) is not None and getattr(state, '_async_audio_queue', None) is not None:
                    asyncio.run_coroutine_threadsafe(state._async_audio_queue.put((priority_val, now, payload_e)), state._async_audio_loop)
                else:
                    audio_queue.put_nowait((priority_val, now, payload_e))
            except Exception:
                audio_queue.put_nowait((priority_val, now, payload_e))
            _last_enqueued_time = now
    except Exception:
        # Fallback: enqueue to main audio queue; as last resort, direct speak
        try:
            payload_f = {'msg': varied_message, 'enqueue_ts': now, 'urgent': bool(is_urgent), 'rate': float(config.get('audio', {}).get('speech_rate_default', 0.5)), 'level': log_level}
            audio_queue.put_nowait((priority_val, now, payload_f))
            _last_enqueued_time = now
        except Exception:
            speak(varied_message)
            _last_enqueued_time = now
    try:
        if log_level == 'critical':
            logging.error(varied_message)
        elif log_level == 'warning':
            logging.warning(varied_message)
        elif log_level == 'hint':
            logging.debug(varied_message)
        else:
            logging.info(varied_message)
    except Exception:
        logging.info(varied_message)
    # Mirror into announcement feed for external clients
    try:
        with state.lock:
            entry = {
                'time': now,
                'message': str(varied_message),
                'level': str(log_level or 'info'),
                'urgent': bool(is_urgent)
            }
            feed = getattr(state, '_announce_feed', None)
            if isinstance(feed, list):
                feed.append(entry)
                if len(feed) > 100:
                    del feed[:len(feed)-100]
            elif feed is not None:
                try:
                    feed.append(entry)
                except Exception:
                    pass
    except Exception:
        pass
    # Telemetry: record announcement and classify types
    try:
        with state.lock:
            state.telemetry['announce_ts'].append(now)
            m = str(varied_message).lower()
            if 'critical' in m or 'urgent' in m:
                state.telemetry['critical_ts'].append(now)
            # Crowd summaries or proximity summaries
            if ('crowd ahead' in m) or ('people nearby' in m) or ('vehicle(s) nearby' in m):
                state.telemetry['summary_ts'].append(now)
            
            # Enhanced event type classification via expanded keyword sets
            etype = 'system'
            importance = 'low'
            # Keyword groups for classification (lowercase match)
            # Critical: imminent danger or emergency situations
            crit_words = (
                'critical','urgent','emergency','danger','dangerous','imminent',
                'alarm','hazard ahead','mayday','panic'
            )
            # Warning: advisory or cautionary notices
            warn_words = (
                'warning','caution','watch','alert','hazard','heads up','be careful',
                'stay alert','take care','watch out'
            )
            # Clear: resolution or safety confirmation
            clear_words = (
                'clear','safe','all clear','resolved','safe now','area safe','clear ahead'
            )
            # Summary/Status: periodic updates or situational reports
            summary_words = (
                'summary','report','status','update','overview','digest','situation report',
                'crowd','nearby','people nearby','vehicles nearby','vehicle(s) nearby','daily status'
            )

            if any(w in m for w in crit_words):
                etype = 'critical'
                importance = 'high'
            elif any(w in m for w in warn_words):
                etype = 'warning'
                importance = 'medium'
            elif any(w in m for w in clear_words):
                etype = 'clear'
                importance = 'low'
            elif any(w in m for w in summary_words):
                etype = 'summary'
                importance = 'medium'
            
            # Update event type and importance counters (deque-based; fallback to int)
            try:
                state.telemetry['event_type_counts'][etype].append(now)
            except Exception:
                state.telemetry['event_type_counts'][etype] = int(state.telemetry['event_type_counts'].get(etype, 0) or 0) + 1
            try:
                state.telemetry['importance_counts'][importance].append(now)
            except Exception:
                state.telemetry['importance_counts'][importance] = int(state.telemetry['importance_counts'].get(importance, 0) or 0) + 1
            
            # Generic event log for dashboard
            state.telemetry['events'].append({
                'time': now,
                'type': etype,
                'importance': importance,
                'message': str(varied_message)
            })
            # Update per-area adaptive counters
            try:
                pos = dict(getattr(state, 'positioning', {}) or {})
                lat = pos.get('lat'); lon = pos.get('lon')
                if (lat is not None) and (lon is not None):
                    cell = (round(float(lat), 4), round(float(lon), 4))
                    ac = getattr(state, '_area_event_counts', {})
                    if cell not in ac:
                        ac[cell] = {'critical': 0, 'warning': 0, 'clear': 0, 'summary': 0, 'system': 0, 'last': 0.0}
                    ac[cell][etype] = int(ac[cell].get(etype, 0) or 0) + 1
                    ac[cell]['last'] = time.time()
                    state._area_event_counts = ac
            except Exception:
                pass
    except Exception:
        pass
# Use speak_status() for important events

def play_earcon(level='nearby', side='center'):
    """Play a short system sound earcon; side indicates lateral cue.
    Note: macOS afplay has no pan; we differentiate sides using distinct tones.
    """
    # Respect earcon enablement flag
    try:
        if not bool(getattr(state, '_earcons_enabled', True)):
            return
    except Exception:
        pass
    # Choose distinct tones per side to imply direction
    base_map = {
        'critical': '/System/Library/Sounds/Basso.aiff',
        'warning': '/System/Library/Sounds/Ping.aiff',
        'nearby': '/System/Library/Sounds/Pop.aiff',
        'success': '/System/Library/Sounds/Hero.aiff',
        'info': '/System/Library/Sounds/Blow.aiff',
        'error': '/System/Library/Sounds/Basso.aiff'
    }
    left_map = {
        'warning': '/System/Library/Sounds/Glass.aiff',
        'nearby': '/System/Library/Sounds/Funk.aiff',
        'success': '/System/Library/Sounds/Tink.aiff'
    }
    right_map = {
        'warning': '/System/Library/Sounds/Submarine.aiff',
        'nearby': '/System/Library/Sounds/Tink.aiff',
        'success': '/System/Library/Sounds/Glass.aiff'
    }
    path = None
    s = str(side).lower()
    if s == 'left':
        path = left_map.get(level, base_map.get(level))
    elif s == 'right':
        path = right_map.get(level, base_map.get(level))
    else:
        path = base_map.get(level)
    if path:
        try:
            vol = float(getattr(state, '_earcon_volume', 0.8) or 0.8)
            vol = max(0.0, min(1.0, vol))
            subprocess.Popen(['afplay', '-v', f'{vol}', path])
        except Exception:
            pass
    # Provide haptic feedback when available
    try:
        perform_haptic(level)
    except Exception:
        pass

def _compute_spatial_params(side: str, depth_m: float):
    """Compute spatial audio parameters: pan (-1..1) and volume scale (0..1).
    Pan uses -1 for left, 0 center/ahead, +1 for right. Volume increases as distance decreases.
    """
    try:
        sa = config.get('spatial_audio', {})
        enabled = bool(sa.get('enabled', False))
        if not enabled:
            return 0.0, 1.0
        max_dist = float(sa.get('max_distance', 10.0))
        near = float(sa.get('near_distance', 1.0))
        far = float(sa.get('far_distance', max_dist))
        curve = str(sa.get('curve', 'linear') or 'linear').lower()
        intensity = float(sa.get('intensity', 0.7))
        # Pan: default discrete based on coarse side
        s = (side or 'ahead').lower()
        pan = -1.0 if s == 'left' else (1.0 if s == 'right' else 0.0)
        # If a recent pan hint exists (computed from bbox→angle mapping), prefer it
        try:
            hint_ttl = float(sa.get('pan_hint_ttl_s', 0.5))
            ts = float(getattr(state, '_last_spatial_pan_ts', 0.0) or 0.0)
            if ts > 0.0 and (time.time() - ts) <= hint_ttl:
                pan_hint = float(getattr(state, '_last_spatial_pan', pan))
                pan = max(-1.0, min(1.0, pan_hint))
        except Exception:
            pass
        # Distance-based volume scaling with configurable curve
        d = float(depth_m or 0.0)
        d = max(0.0, d)
        d_eff = min(max(d, near), far)
        if far > near:
            r = (far - d_eff) / (far - near)  # 1 at near, 0 at far
        else:
            r = 1.0 if d_eff <= near else 0.0
        r = max(0.0, min(1.0, r))
        if curve == 'exp':
            gamma = 1.0 + 2.0 * max(0.0, min(1.0, intensity))
            base = r ** gamma
        elif curve == 'log':
            import math
            k = 4.0
            base = math.log1p(k * r) / math.log1p(k)
        else:
            base = r
        gain = float(sa.get('distance_scaling', 1.0))
        vol_scale = max(0.0, min(1.0, base * (0.6 + 0.4 * intensity) * gain))
        return pan, vol_scale
    except Exception:
        return 0.0, 1.0

def play_spatial_earcon(level='nearby', side='center', depth_m: float = 0.0):
    """Spatially aware earcon: choose side-specific tone and scale volume with distance.
    Records telemetry event 'spatial_earcon'. Falls back to play_earcon if disabled.
    """
    try:
        if not bool(getattr(state, '_earcons_enabled', True)):
            return
    except Exception:
        pass
    try:
        sa = config.get('spatial_audio', {})
        if not bool(sa.get('enabled', False)):
            # Spatial audio disabled, use standard earcon
            return play_earcon(level, side=side)
        # Rate-limit spatial earcons
        min_interval = float(sa.get('min_interval_s', 0.2))
        now = time.time()
        # Allow urgent hazard tones to bypass rate-limit
        bypass_rl = str(level or '').lower() in ('warning', 'critical')
        with state.lock:
            last_ts = float(getattr(state, '_last_spatial_earcon_ts', 0.0) or 0.0)
            if not bypass_rl and (now - last_ts) < max(0.0, min_interval):
                return
            state._last_spatial_earcon_ts = now
        pan, vol_scale = _compute_spatial_params(side, depth_m)
        # Clamp and derive effective volume
        base_vol = float(getattr(state, '_earcon_volume', 0.8) or 0.8)
        eff_vol = max(0.0, min(1.0, base_vol * vol_scale))
        # Optional ambient-noise-aware gain boost
        try:
            noise_lvl = float(state.environment_context.get('noise_level', 0.0) or 0.0)
            noise_thr = float(sa.get('noise_trigger_level', 0.7))
            gain_boost = float(sa.get('noise_gain_boost', 0.15))
            if noise_lvl >= noise_thr:
                eff_vol = max(0.0, min(1.0, eff_vol * (1.0 + max(0.0, gain_boost))))
        except Exception:
            pass
        # Choose tone based on side using existing mapping
        base_map = {
            'critical': '/System/Library/Sounds/Basso.aiff',
            'warning': '/System/Library/Sounds/Ping.aiff',
            'nearby': '/System/Library/Sounds/Pop.aiff',
            'success': '/System/Library/Sounds/Hero.aiff',
            'info': '/System/Library/Sounds/Blow.aiff',
            'error': '/System/Library/Sounds/Basso.aiff'
        }
        left_map = {
            'warning': '/System/Library/Sounds/Glass.aiff',
            'nearby': '/System/Library/Sounds/Funk.aiff',
            'success': '/System/Library/Sounds/Tink.aiff'
        }
        right_map = {
            'warning': '/System/Library/Sounds/Submarine.aiff',
            'nearby': '/System/Library/Sounds/Tink.aiff',
            'success': '/System/Library/Sounds/Glass.aiff'
        }
        s = str(side or 'ahead').lower()
        if s == 'left':
            path = left_map.get(level, base_map.get(level))
        elif s == 'right':
            path = right_map.get(level, base_map.get(level))
        else:
            path = base_map.get(level)
        if path:
            try:
                subprocess.Popen(['afplay', '-v', f'{eff_vol}', path])
            except Exception:
                # Fallback to non-spatial earcon
                try:
                    subprocess.Popen(['afplay', '-v', f'{base_vol}', path])
                except Exception:
                    pass
        # Telemetry event
        try:
            # Derive angle from hint if available, otherwise from side using FOV
            sa = config.get('spatial_audio', {})
            fov_deg = float(sa.get('horizontal_fov_deg', 60.0))
            hint_ttl = float(sa.get('pan_hint_ttl_s', 0.5))
            angle_deg = 0.0
            try:
                ts = float(getattr(state, '_last_spatial_pan_ts', 0.0) or 0.0)
                if ts > 0.0 and (time.time() - ts) <= hint_ttl:
                    angle_deg = float(getattr(state, '_last_spatial_angle_deg', 0.0) or 0.0)
                else:
                    angle_deg = (-fov_deg / 2.0) if s == 'left' else ((fov_deg / 2.0) if s == 'right' else 0.0)
            except Exception:
                angle_deg = 0.0
            with state.lock:
                state.telemetry['events'].append({
                    'time': time.time(),
                    'type': 'spatial_earcon',
                    'side': s,
                    'depth_m': float(depth_m or 0.0),
                    'pan': float(pan),
                    'angle_deg': float(angle_deg),
                    'vol': float(eff_vol),
                    'level': level
                })
        except Exception:
            pass
        # Optional haptic
        try:
            perform_haptic(level)
        except Exception:
            pass
    except Exception:
        # Any error -> standard earcon
        try:
            play_earcon(level, side=side)
        except Exception:
            pass

# macOS haptics (trackpad) via AppKit; safe fallback if unavailable
try:
    from AppKit import NSHapticFeedbackManager, NSHapticFeedbackPatternGeneric, NSHapticFeedbackPerformanceTimeNow
    HAVE_HAPTICS = True
except Exception:
    HAVE_HAPTICS = False

def perform_haptic(level='nearby'):
    """Emit a short haptic pulse when platform support is available.

    Parameters:
    - level: semantic intensity hint (e.g., 'nearby', 'warning', 'critical').
      Current implementation uses a generic pattern; future mapping may vary
      pattern strength by `level` when supported.

    Returns:
    - None. No-op if haptics are unavailable.
    """
    if not HAVE_HAPTICS:
        return
    try:
        mgr = NSHapticFeedbackManager.defaultPerformer()
        # Use generic haptic; stronger pattern for warnings/critical if available
        pattern = NSHapticFeedbackPatternGeneric
        mgr.performFeedback_pattern_performanceTime_(None, pattern, NSHapticFeedbackPerformanceTimeNow)
    except Exception:
        pass

# Heartbeat haptic for suppressed, unchanged hazards
def emit_hazard_heartbeat(candidates, config):
    """Emit a subtle haptic heartbeat for persistent hazards when speech is gated.

    Runs a global and per-track cooldown to avoid over-stimulation, chooses the
    closest candidate, and triggers a single haptic pulse. Designed to provide
    awareness without adding audio chatter when repeated alerts were suppressed.

    Parameters:
    - candidates: list of `(depth_m, detection)` tuples for hazards considered
      important but suppressed this tick.
    - config: runtime configuration dict. Uses `haptics.heartbeat_enabled`,
      `haptics.heartbeat_interval_s`, `haptics.per_track_interval_s`, and
      `haptics.heartbeat_level`.

    Returns:
    - bool indicating whether a heartbeat was emitted.
    """
    try:
        hcfg = config.get('haptics', {})
        if not bool(hcfg.get('heartbeat_enabled', True)):
            return False
        min_global = float(hcfg.get('heartbeat_interval_s', 3.5))
        min_pt = float(hcfg.get('per_track_interval_s', 5.0))
        level = str(hcfg.get('heartbeat_level', 'nearby'))
    except Exception:
        min_global, min_pt, level = 3.5, 5.0, 'nearby'

    now = time.time()
    try:
        last_global = float(getattr(state, '_last_heartbeat_ts', 0.0) or 0.0)
        if (now - last_global) < min_global:
            return False
    except Exception:
        pass

    # Choose the closest candidate
    try:
        candidates = [c for c in candidates if (c[0] is not None)]
    except Exception:
        pass
    if not candidates:
        return False
    try:
        candidates.sort(key=lambda tup: tup[0])
    except Exception:
        pass
    depth_m, det = candidates[0]
    track_id = det.get('track_id')

    # Per-track gating
    try:
        if not hasattr(state, '_heartbeat_per_track'):
            state._heartbeat_per_track = {}
        last_pt = float(state._heartbeat_per_track.get(track_id, 0.0) or 0.0) if (track_id is not None) else 0.0
        if (track_id is not None) and ((now - last_pt) < min_pt):
            return False
    except Exception:
        pass

    # Emit haptic heartbeat only (no voice)
    try:
        perform_haptic(level)
    except Exception:
        try:
            perform_haptic('nearby')
        except Exception:
            pass

    # Record telemetry and timestamps
    try:
        with state.lock:
            state._last_heartbeat_ts = now
            if track_id is not None:
                state._heartbeat_per_track[track_id] = now
            try:
                state.telemetry['events'].append({'time': now, 'type': 'heartbeat', 'track_id': track_id, 'depth_m': float(depth_m)})
            except Exception:
                pass
    except Exception:
        pass
    return True

def emit_multimodal_alert(level: str, side: str = 'center', depth_m: float = 0.0, msg: str = None, priority: int = None):
    """Emit a combined haptic + spatial audio + optional voice alert.
    Prevents user overload by gating voice in noisy or high-backlog contexts while ensuring redundancy.
    Adapts alert urgency and type based on contextual scenarios.

    Returns True if any modality was emitted.
    """
    emitted = False
    try:
        lvl = str(level or 'warning').lower()
        
        # Apply contextual scenario adaptations
        try:
            contextual_scenarios = state.environment_context.get('contextual_scenarios', {})
            active_scenarios = contextual_scenarios.get('active', [])
            max_urgency = contextual_scenarios.get('max_urgency', 'low')
            
            # Escalate urgency based on active scenarios
            if active_scenarios:
                if max_urgency == 'high' and lvl == 'warning':
                    lvl = 'critical'  # Escalate warnings to critical in high-urgency scenarios
                elif max_urgency == 'medium' and lvl == 'info':
                    lvl = 'warning'   # Escalate info to warning in medium-urgency scenarios
                
                # Modify message based on scenario context
                if msg and active_scenarios:
                    scenario_details = contextual_scenarios.get('details', {})
                    
                    # Add contextual prefixes for specific scenarios
                    if 'road_crossing' in active_scenarios:
                        road_scenario = scenario_details.get('road_crossing', {})
                        if road_scenario.get('urgency') == 'high':
                            msg = f"Traffic alert: {msg}"
                            try:
                                speak_async(msg)
                            except Exception:
                                pass
                        elif 'crosswalk_detected' in road_scenario.get('cues', []):
                            msg = f"Crosswalk: {msg}"
                            try:
                                speak_async(msg)
                            except Exception:
                                pass
                    

                    
                    if 'crowded_area' in active_scenarios:
                        crowd_scenario = scenario_details.get('crowded_area', {})
                        person_count = crowd_scenario.get('person_count', 0)
                        if person_count >= 8:
                            msg = f"Dense crowd: {msg}"
                            try:
                                speak_async(msg)
                            except Exception:
                                pass
                        elif person_count >= 5:
                            msg = f"Crowded area: {msg}"
                            try:
                                speak_async(msg)
                            except Exception:
                                pass
                        elif 'bottleneck_detected' in crowd_scenario.get('cues', []):
                            msg = f"Narrow passage: {msg}"
                            try:
                                speak_async(msg)
                            except Exception:
                                pass
                
                # Log contextual adaptation telemetry
                try:
                    telemetry_data = {
                        'timestamp': time.time(),
                        'event': 'contextual_alert_adaptation',
                        'original_level': level,
                        'adapted_level': lvl,
                        'active_scenarios': active_scenarios,
                        'max_urgency': max_urgency,
                        'depth_m': depth_m,
                        'side': side
                    }
                    log_telemetry(telemetry_data)
                except Exception:
                    pass
        except Exception:
            pass
        
        # Respect user preferences for urgency/type; allow earcons/haptics unless muted
        try:
            prefs = _get_active_alert_prefs()
        except Exception:
            prefs = {'urgency_enabled': ['critical', 'warning', 'info'], 'types_enabled': {}, 'mute_all': False}
        if bool(prefs.get('mute_all', False)):
            return False
        # Urgency gate for voice; earcons/haptics can still emit
        urg_enabled = list(prefs.get('urgency_enabled', []))
        if (lvl == 'critical') and (('critical' not in urg_enabled) and ('urgent' not in urg_enabled)):
            # Skip speaking later
            voice_blocked = True
        elif (lvl == 'warning') and ('warning' not in urg_enabled and 'caution' not in urg_enabled):
            voice_blocked = True
        else:
            voice_blocked = False
        mm_cfg = config.get('multimodal_cues', {})
        # Context inputs
        noise_lvl = float(getattr(state, 'environment_context', {}).get('noise_level', 0.0) or 0.0)
        voice_noise_thr = float(mm_cfg.get('voice_noise_threshold', 0.75))
        backlog_thr = int(mm_cfg.get('tts_backlog_threshold', 6))
        # Queue backlog
        try:
            qsize = int(getattr(globals(), 'audio_queue', None).qsize()) if 'audio_queue' in globals() else 0
        except Exception:
            qsize = 0
        # Base: always provide earcon + haptic for spatial redundancy
        try:
            play_spatial_earcon(lvl, side=side or 'center', depth_m=float(depth_m or 0.0))
            emitted = True
        except Exception:
            try:
                play_earcon(lvl, side=side or 'center')
                emitted = True
            except Exception:
                pass
        try:
            perform_haptic(lvl)
            emitted = True
        except Exception:
            pass
        # Voice gating rules
        voice_always_for_critical = bool(mm_cfg.get('voice_always_for_critical', True))
        allow_voice = True
        if lvl == 'critical':
            allow_voice = voice_always_for_critical
        else:
            # For warnings, suppress voice in noisy scenes or when backlog high
            allow_voice = (noise_lvl < voice_noise_thr) and (qsize < backlog_thr)
        # Respect global silence window for non-critical hints
        if lvl != 'critical':
            try:
                allow_voice = allow_voice and (time.time() >= float(getattr(state, '_global_silence_until', 0.0) or 0.0))
            except Exception:
                pass
        # Speak if allowed and message provided
        to_speak = str(msg or '').strip()
        if allow_voice and (not voice_blocked) and to_speak:
            try:
                # Choose priority if provided, else map by level
                p = int(priority) if priority is not None else (10 if lvl == 'critical' else 9)
                # Debug: announcement trigger details
                try:
                    logging.info(
                        f"Announcement triggered lvl={lvl} side={side} depth={float(depth_m or 0.0):.2f}m msg='{to_speak}' "
                        f"priority={p} noise={float(noise_lvl):.2f} backlog={int(qsize)} voice_allowed={allow_voice} blocked={voice_blocked}"
                    )
                except Exception:
                    pass
                speak_status(to_speak, priority=p)
                emitted = True
                # Record telemetry of multimodal decision
                try:
                    with state.lock:
                        state.telemetry['events'].append({
                            'time': time.time(),
                            'type': 'multimodal_alert',
                            'level': lvl,
                            'side': side,
                            'depth_m': float(depth_m or 0.0),
                            'noise_level': float(noise_lvl),
                            'tts_backlog': int(qsize),
                            'spoken': True
                        })
                except Exception:
                    pass
            except Exception:
                pass
        else:
            # Log that voice was suppressed to avoid overload
            try:
                with state.lock:
                    state.telemetry['events'].append({
                        'time': time.time(),
                        'type': 'multimodal_alert',
                        'level': lvl,
                        'side': side,
                        'depth_m': float(depth_m or 0.0),
                        'noise_level': float(noise_lvl),
                        'tts_backlog': int(qsize),
                        'spoken': False
                    })
            except Exception:
                pass
    except Exception:
        pass
    return emitted

def audio_worker():
    """Background worker to handle queued TTS announcements with de-duplication."""
    last_spoken_at = {}
    # Initialize TTS backend once
    try:
        init_tts_backend()
    except Exception:
        pass
    # Record backend and current volume for telemetry/diagnostics
    try:
        with state.lock:
            state.telemetry['events'].append({
                'time': time.time(),
                'type': 'audio_backend_init',
                'backend': str(getattr(state, '_speech_backend', 'say')),
                'speech_volume': float(getattr(state, '_speech_volume', 1.0) or 1.0)
            })
    except Exception:
        pass
    while state.running:
        # Heartbeat for watchdog liveness tracking
        try:
            with state.lock:
                state._last_audio_tick = time.time()
        except Exception:
            pass
        # Handle preempt request: stop AVFoundation immediately if currently speaking
        try:
            ts = float(getattr(state, '_preempt_now_ts', 0.0) or 0.0)
            backend = getattr(state, '_speech_backend', 'say')
            if ts > 0.0:
                with state.lock:
                    state._preempt_now_ts = 0.0
                if backend == 'avfoundation' and getattr(state, '_av_synth', None) is not None:
                    if state._av_synth.isSpeaking():
                        # 0 = immediate boundary in AVSpeechBoundary
                        try:
                            state._av_synth.stopSpeakingAtBoundary_(0)
                        except Exception:
                            pass
        except Exception:
            pass
        try:
            item = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            dequeue_ts = time.time()
            # Support priority items: (priority, enq_ts, payload)
            if isinstance(item, tuple) and len(item) == 3 and isinstance(item[2], dict):
                priority, _enq, payload = item
                msg = str(payload.get('msg', ''))
                enq_ts = float(payload.get('enqueue_ts', dequeue_ts))
                is_urgent = bool(payload.get('urgent', False))
                speak_rate = payload.get('rate', None)
                log_level = str(payload.get('level', '') or '')
            elif isinstance(item, dict):
                # Legacy dict item
                msg = str(item.get('msg', ''))
                enq_ts = float(item.get('enqueue_ts', dequeue_ts))
                is_urgent = False
                speak_rate = None
                log_level = ''
            else:
                # Legacy string item
                msg = str(item)
                enq_ts = dequeue_ts
                is_urgent = False
                speak_rate = None
                log_level = ''
            last_time = last_spoken_at.get(msg, 0.0)
            # Skip if identical message was spoken very recently
            dedup_s = float(config.get('audio', {}).get('dedup_window_s', 2.0))
            if dequeue_ts - last_time < dedup_s:
                # Let the finally block mark task done
                continue
            # Urgent preemption: stop ongoing speech if supported
            try:
                urgent_preempt = bool(config.get('audio', {}).get('urgent_preempt', True))
                backend = getattr(state, '_speech_backend', 'say')
                if urgent_preempt and is_urgent and backend == 'avfoundation' and getattr(state, '_av_synth', None) is not None:
                    if state._av_synth.isSpeaking():
                        # 0 = immediate boundary in AVSpeechBoundary
                        state._av_synth.stopSpeakingAtBoundary_(0)
            except Exception:
                pass
            # Serialize speech to avoid overlapping TTS; use macOS AVFoundation if available
            start_ts = time.time()
            # Record queue latency (time spent waiting before speech begins)
            try:
                with state.lock:
                    queue_latency = max(0.0, start_ts - enq_ts)
                    state.performance_stats['tts_queue_latencies'].append(queue_latency)
                    # Add to histogram data (in milliseconds)
                    state.telemetry['queue_latencies']['tts_queue'].append(queue_latency * 1000.0)
            except Exception:
                pass
            # Apply per-message speech rate if provided
            try:
                if speak_rate is not None:
                    state._speech_rate = float(speak_rate)
                else:
                    # Fallback to default rate if not specified
                    state._speech_rate = float(config.get('audio', {}).get('speech_rate_default', 0.5))
            except Exception:
                pass
            # Pre-roll spatial earcon for non-hazard informational messages (direction + distance)
            try:
                lvl, side_hint = _infer_earcon_for_message(msg)
                if lvl and bool(getattr(state, '_earcons_enabled', True)) and (time.time() >= getattr(state, '_global_silence_until', 0.0)):
                    depth_hint = _parse_depth_hint_from_message(msg)
                    try:
                        play_spatial_earcon(lvl, side=side_hint, depth_m=depth_hint)
                    except Exception:
                        play_earcon(lvl, side=side_hint)
            except Exception:
                pass
            # Log playback with appropriate level
            try:
                low = msg.lower()
                lvl = log_level or ('critical' if (('critical' in low) or ('urgent' in low)) else ('warning' if (('warning' in low) or ('caution' in low) or ('alert' in low)) else ('hint' if (('heads up' in low) or ('guidance' in low) or ('tip:' in low)) else 'info')))
                if lvl == 'critical':
                    logging.error(f"Audio CRITICAL: {msg}")
                elif lvl == 'warning':
                    logging.warning(f"Audio WARNING: {msg}")
                elif lvl == 'hint':
                    logging.debug(f"Audio HINT: {msg}")
                else:
                    logging.info(f"Audio: {msg}")
            except Exception:
                logging.info(msg)
            # Enforce a global minimum gap between spoken messages to avoid overlap
            try:
                gap_s = float(config.get('audio', {}).get('global_min_speak_gap_s', 0.8))
            except Exception:
                gap_s = 0.8
            try:
                last_end = float(getattr(state, '_last_audio_work', 0.0) or 0.0)
                now_gap = time.time()
                remaining = gap_s - max(0.0, (now_gap - last_end))
                if gap_s > 0.0 and remaining > 0.0:
                    time.sleep(min(gap_s, remaining))
            except Exception:
                pass
            tts_speak(msg, block=True)
            end_ts = time.time()
            # Record actual speak duration
            try:
                with state.lock:
                    speak_latency = max(0.0, end_ts - start_ts)
                    state.performance_stats['tts_speak_latencies'].append(speak_latency)
                    # Add to histogram data (in milliseconds)
                    state.telemetry['queue_latencies']['tts_speak'].append(speak_latency * 1000.0)
            except Exception:
                pass
            last_spoken_at[msg] = end_ts
            try:
                with state.lock:
                    state._last_audio_work = end_ts
            except Exception:
                pass
        except Exception:
            # Swallow TTS errors; task_done will be handled in finally
            pass
        finally:
            try:
                audio_queue.task_done()
            except Exception:
                pass



async def async_audio_worker():
    """Asyncio-based audio worker that mirrors audio_worker behavior using an async queue."""
    # Initialize TTS backend once in this thread/loop context
    try:
        init_tts_backend()
    except Exception:
        pass
    last_spoken_at = {}
    while state.running:
        try:
            with state.lock:
                state._last_audio_tick = time.time()
        except Exception:
            pass
        try:
            item = await state._async_audio_queue.get()
        except Exception:
            await asyncio.sleep(0.1)
            continue
        try:
            dequeue_ts = time.time()
            if isinstance(item, tuple) and len(item) == 3 and isinstance(item[2], dict):
                priority, _enq, payload = item
                msg = str(payload.get('msg', ''))
                enq_ts = float(payload.get('enqueue_ts', dequeue_ts))
                is_urgent = bool(payload.get('urgent', False))
                speak_rate = payload.get('rate', None)
                log_level = str(payload.get('level', '') or '')
            elif isinstance(item, dict):
                msg = str(item.get('msg', ''))
                enq_ts = float(item.get('enqueue_ts', dequeue_ts))
                is_urgent = False
                speak_rate = None
                log_level = ''
            else:
                msg = str(item)
                enq_ts = dequeue_ts
                is_urgent = False
                speak_rate = None
                log_level = ''
            last_time = last_spoken_at.get(msg, 0.0)
            dedup_s = float(config.get('audio', {}).get('dedup_window_s', 2.0))
            if dequeue_ts - last_time < dedup_s:
                continue
            # Urgent preempt for AVFoundation
            try:
                urgent_preempt = bool(config.get('audio', {}).get('urgent_preempt', True))
                backend = getattr(state, '_speech_backend', 'say')
                if urgent_preempt and is_urgent and backend == 'avfoundation' and getattr(state, '_av_synth', None) is not None:
                    if state._av_synth.isSpeaking():
                        state._av_synth.stopSpeakingAtBoundary_(0)
            except Exception:
                pass
            start_ts = time.time()
            try:
                with state.lock:
                    queue_latency = max(0.0, start_ts - enq_ts)
                    state.performance_stats['tts_queue_latencies'].append(queue_latency)
                    state.telemetry['queue_latencies']['tts_queue'].append(queue_latency * 1000.0)
            except Exception:
                pass
            try:
                if speak_rate is not None:
                    state._speech_rate = float(speak_rate)
                else:
                    state._speech_rate = float(config.get('audio', {}).get('speech_rate_default', 0.5))
            except Exception:
                pass
            # Earcon pre-roll
            try:
                lvl, side_hint = _infer_earcon_for_message(msg)
                if lvl and bool(getattr(state, '_earcons_enabled', True)) and (time.time() >= getattr(state, '_global_silence_until', 0.0)):
                    depth_hint = _parse_depth_hint_from_message(msg)
                    try:
                        play_spatial_earcon(lvl, side=side_hint, depth_m=depth_hint)
                    except Exception:
                        play_earcon(lvl, side=side_hint)
            except Exception:
                pass
            # Log playback
            try:
                low = msg.lower()
                lvl = log_level or ('critical' if (('critical' in low) or ('urgent' in low)) else ('warning' if (('warning' in low) or ('caution' in low) or ('alert' in low)) else ('hint' if (('heads up' in low) or ('guidance' in low) or ('tip:' in low)) else 'info')))
                if lvl == 'critical':
                    logging.error(f"Audio CRITICAL: {msg}")
                elif lvl == 'warning':
                    logging.warning(f"Audio WARNING: {msg}")
                elif lvl == 'hint':
                    logging.debug(f"Audio HINT: {msg}")
                else:
                    logging.info(f"Audio: {msg}")
            except Exception:
                logging.info(msg)
            # Global min gap
            try:
                gap_s = float(config.get('audio', {}).get('global_min_speak_gap_s', 0.8))
            except Exception:
                gap_s = 0.8
            try:
                last_end = float(getattr(state, '_last_audio_work', 0.0) or 0.0)
                now_gap = time.time()
                remaining = gap_s - max(0.0, (now_gap - last_end))
                if gap_s > 0.0 and remaining > 0.0:
                    await asyncio.sleep(min(gap_s, remaining))
            except Exception:
                pass
            # Speak using executor to avoid blocking loop
            try:
                loop = asyncio.get_running_loop()
            except Exception:
                loop = getattr(state, '_async_audio_loop', None)
            if loop:
                await loop.run_in_executor(None, tts_speak, msg, True)
            else:
                tts_speak(msg, block=True)
            end_ts = time.time()
            try:
                with state.lock:
                    speak_latency = max(0.0, end_ts - start_ts)
                    state.performance_stats['tts_speak_latencies'].append(speak_latency)
                    state.telemetry['queue_latencies']['tts_speak'].append(speak_latency * 1000.0)
            except Exception:
                pass
            last_spoken_at[msg] = end_ts
            try:
                with state.lock:
                    state._last_audio_work = end_ts
            except Exception:
                pass
        except Exception:
            pass
        finally:
            try:
                state._async_audio_queue.task_done()
            except Exception:
                pass


async def _make_async_audio_queue():
    """Create an asyncio PriorityQueue on the current event loop."""
    try:
        q = asyncio.PriorityQueue()
        return q
    except Exception:
        # Return None to signal failure so caller can fall back
        return None


def calculate_blind_navigation_priority(detection):
    """Calculate priority specifically for blind navigation needs"""
    if not config.get('blind_navigation', {}).get('enabled', True):
        return 0
        
    priority = 0
    label = detection['label'].lower()
    
    # Get configuration values
    bn = config.get('blind_navigation', {})
    path_width = float(bn.get('path_width_factor', 0.2))
    weights = bn.get('weights', {})
    trip_priority = float(weights.get('trip', bn.get('trip_hazard_priority', 3)))
    overhead_priority = float(weights.get('overhead', bn.get('overhead_hazard_priority', 2)))
    path_priority = float(weights.get('path', bn.get('path_obstacle_priority', 3)))
    critical_priority = float(weights.get('critical', bn.get('critical_distance_priority', 5)))
    important_classes = bn.get('important_classes', [])
    class_boost_w = float(weights.get('class_boost', 2.0))
    # Adaptive multipliers from recent real-world stats
    adapt_mult_trip = adapt_mult_overhead = adapt_mult_path = 1.0
    try:
        if bool(bn.get('adaptive_tuning_enabled', True)):
            stats = getattr(state, 'nav_weight_stats', None)
            if isinstance(stats, dict):
                # Normalize frequencies and add a small floor
                trip_f = float(stats.get('trip', {}).get('freq', 0.0) or 0.0)
                over_f = float(stats.get('overhead', {}).get('freq', 0.0) or 0.0)
                path_f = float(stats.get('path', {}).get('freq', 0.0) or 0.0)
                # Scale multipliers within [0.9, 1.5]
                def mult(f):
                    return max(0.9, min(1.5, 0.9 + 0.6 * f))
                adapt_mult_trip = mult(trip_f)
                adapt_mult_overhead = mult(over_f)
                adapt_mult_path = mult(path_f)
    except Exception:
        pass
    
    # Higher priority for obstacles in walking path (center of frame)
    center_x = detection.get('center_x', 0.5)
    path_left = 0.5 - path_width
    path_right = 0.5 + path_width
    if center_x > path_left and center_x < path_right:
        priority += path_priority * adapt_mult_path
        
    # Higher priority for trip hazards (low objects)
    bottom = detection.get('bottom', 0)
    if bottom > 0.7:  # Lower part of the frame
        priority += trip_priority * adapt_mult_trip
        
    # Higher priority for overhead hazards
    top = detection.get('top', 0)
    if top < 0.3:  # Upper part of the frame
        priority += overhead_priority * adapt_mult_overhead
        
    # Critical priority for very close objects
    distance = detection.get('depth_m', 10)
    try:
        crit_th = float(config['distances']['critical'])
        logging.info(f"Priority distance check: label={label} depth_m={distance} critical_th={crit_th}")
    except Exception:
        logging.info(f"Priority distance check: label={label} depth_m={distance} critical_th=?")
    if distance is not None and distance < config['distances']['critical']:
        priority += critical_priority
        
    # Boost priority for important classes for blind navigation
    if label in important_classes:
        priority += class_boost_w
        
    return priority

def calculate_importance_score(detection, speed, context):
    """Calculate how important this detection is for a blind person"""
    label = detection['label'].lower()
    depth = detection['depth_m']
    conf = detection['conf']
    
    # Base importance from class preferences
    pref = state.class_prefs.get(label, {'priority': 1, 'max_m': 10.0, 'phrase': label, 'urgency_multiplier': 1.0})
    base_importance = pref['priority'] * 0.3

    # Add blind navigation specific priority
    blind_priority = calculate_blind_navigation_priority(detection)
    base_importance += blind_priority * 0.2

    # Fresh network LiDAR boost: slightly increase base and speed weight
    network_boost = 1.0
    try:
        now = time.time()
        use_network = bool(getattr(state, '_use_network_depth', False))
        ext_time = float(getattr(state, '_external_depth_time', 0.0) or 0.0)
        # Per-mode freshness override for network LiDAR boost
        fresh_default = float(config.get('network_depth_freshness_s', 0.8))
        try:
            mode_upper = str(getattr(state, 'travel_mode', 'WALKING')).upper()
        except Exception:
            mode_upper = 'WALKING'
        fresh_s = float(config.get('network_depth_freshness_s_per_mode', {}).get(mode_upper, fresh_default))
        if use_network and (now - ext_time) <= fresh_s:
            base_importance += 0.05
            network_boost = 1.1
    except Exception:
        pass
    
    # Distance factor (closer = more important)
    if depth is not None:
        distance_factor = max(0.1, (pref['max_m'] - depth) / pref['max_m'])
    else:
        distance_factor = 0.1
    
    # Speed factor (faster = more urgent) with ROI-aware weighting
    speed_factor = min(2.0, speed * 0.5) if speed > 0 else 0.1
    try:
        roi_weight_outside = float(config.get('speed_detection', {}).get('roi_outside_speed_weight', 0.6))
        track_id = detection.get('track_id')
        frame_shape = (CAMH, CAMW, 3)
        in_corridor = roi_in_center_corridor(detection, frame_shape, config)
        crossing_soon = predict_crossing_to_center(state.object_tracks.get(track_id), frame_shape, config)
        speed_factor *= (1.0 if (in_corridor or crossing_soon) else roi_weight_outside)
    except Exception:
        pass
    # Apply network LiDAR freshness boost
    speed_factor *= network_boost
    
    # Confidence factor
    conf_factor = conf * 0.3

    # Proximity confidence boost: amplify importance when proximity is high
    try:
        prox_conf = float(detection.get('proximity_confidence', 0.0))
        # Weight can be tuned via config
        prox_weight = float(config.get('proximity', {}).get('importance_weight', 0.6))
        base_importance += prox_conf * prox_weight * 0.5  # additive boost
        # Couple proximity to speed urgency a bit
        speed_factor *= (1.0 + min(0.5, prox_conf * 0.5))
    except Exception:
        pass

    # Person-specific boost: prioritize approaching people via small base and speed multipliers
    try:
        if label == 'person':
            ann_cfg = config.get('announcements', {})
            base_boost = float(ann_cfg.get('person_importance_base_boost', 0.15))
            speed_mult = float(ann_cfg.get('person_speed_boost_mult', 1.15))
            base_importance += base_boost
            # Emphasize when close or moving towards user
            close_warning_m = float(config.get('distances', {}).get('warning', WARNING_DISTANCE_M))
            if (depth is not None and depth <= close_warning_m) or (speed >= FAST_SPEED_THRESHOLD):
                speed_factor *= speed_mult
    except Exception:
        pass

    # Context factors
    # Egocentric motion suppression using IMU: if device is moving, reduce perceived object motion urgency
    try:
        imu = getattr(state, 'imu', {})
        accel = imu.get('user_acceleration', [0.0, 0.0, 0.0])
        gyro = imu.get('rotation_rate', [0.0, 0.0, 0.0])
        ego_motion = float(np.linalg.norm(accel)) + 0.05 * float(np.linalg.norm(gyro))
        mv_thresh = float(config.get('announcements', {}).get('movement_threshold', 0.1))
        if ego_motion > mv_thresh:
            speed_factor *= 0.6
    except Exception:
        pass
    # Intersection-aware boost: prefer traffic-related prompts near intersections
    try:
        tags = context.get('scene_tags', [])
        if 'intersection' in tags and label in ('traffic light','stop sign','car','bus','truck'):
            base_importance += 0.2
            speed_factor *= 1.15
    except Exception:
        pass
    time_since_last = time.time() - state.last_announcement_time
    silence_factor = min(1.0, time_since_last / state.silence_threshold)
    
    # Movement factor (if user is moving, be more selective)
    movement_factor = 1.0
    if len(state.user_movement_history) > 3:
        recent_movement = np.mean(state.user_movement_history[-3:])
        if recent_movement > state.movement_threshold:
            movement_factor = 0.7  # Be more selective when user is moving
    
    # Scene stability factor
    stability_factor = context.get('scene_stability', 1.0)
    
    # Calculate final importance score
    importance = (base_importance + distance_factor + speed_factor + conf_factor) * \
                silence_factor * movement_factor * stability_factor * pref['urgency_multiplier']
    
    return min(1.0, importance)


def should_announce(detection, speed, context):
    """Decide whether a detection should trigger an announcement.

    Evaluates hazard proximity, class filters, temporal windows, and context
    signals (travel mode, corridor hint, scene tags). Returns a tuple decision
    and metadata used by announcer logic.

    Args:
        detection: Detection object with bbox, class, confidence, and distance.
        speed: Current travel speed hint (m/s or categorical).
        context: Dict of environment and state flags influencing suppression.

    Returns:
        A bool indicating whether to announce and a reason/metadata payload.
    """
    """AI decision: Should we announce this detection?"""
    # Real-time mode: always allow critical or fast-approach hazards
    try:
        realtime = bool(getattr(state, 'realtime_mode', False))
    except Exception:
        realtime = False
    if not state.context_aware:
        return True
    # Check if camera is covered/blocked (including partial occlusion) and persist a suppression window
    if state.frame is not None:
        try:
            def _is_camera_occluded(frame, cfg):
                try:
                    # Convert to grayscale defensively
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = frame
                    # Thresholds (with sane defaults)
                    low_luma_thr = float(cfg.get('camera', {}).get('cover_low_luma_thresh', 50))
                    low_frac_thr = float(cfg.get('camera', {}).get('cover_low_frac_thresh', 0.45))
                    std_thr = float(cfg.get('camera', {}).get('cover_std_thresh', 12.0))
                    edge_frac_thr = float(cfg.get('camera', {}).get('cover_edge_frac_thresh', 0.008))
                    # Compute low-luma fraction
                    low_frac = float((gray < low_luma_thr).sum()) / float(gray.size)
                    # Edge density as a proxy for scene structure
                    try:
                        edges = cv2.Canny(gray, 50, 150)
                        edge_frac = float((edges > 0).sum()) / float(edges.size)
                    except Exception:
                        edge_frac = 0.0
                    # Global brightness/variance
                    mean_val = float(np.mean(gray))
                    std_val = float(np.std(gray))
                    # Heuristics: dark or mostly dark, low texture, and low variance
                    if mean_val < 30.0:
                        return True
                    if (low_frac >= low_frac_thr and (edge_frac <= edge_frac_thr or std_val <= std_thr)):
                        return True
                    return False
                except Exception:
                    return False

            if _is_camera_occluded(state.frame, config):
                # Persist a short suppression window to avoid flicker and block clear-path
                try:
                    state._camera_dark_until = time.time() + 6.0
                except Exception:
                    pass
                speak_status("Warning! Camera appears to be covered.", priority=10)
        except Exception:
            pass
            
    # Travel-mode-aware distance thresholds
    mode = getattr(state, 'travel_mode', 'WALKING')
    factor = 1.0
    if str(mode).upper() in ('DRIVING', 'CYCLING'):
        factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
    critical_m = CRITICAL_DISTANCE_M * factor
    warning_m = WARNING_DISTANCE_M * factor
    nearby_m = NEARBY_DISTANCE_M * factor
    # Speed threshold scaling by travel mode
    speed_thr_factor = 1.0
    if str(mode).upper() in ('DRIVING', 'CYCLING'):
        speed_thr_factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.2
    fast_thr_local = FAST_SPEED_THRESHOLD * speed_thr_factor
    very_fast_thr_local = VERY_FAST_THRESHOLD * speed_thr_factor
    # Fuse center-derived looming speed into an effective speed estimate
    eff_speed = (speed or 0.0) if (speed is not None) else 0.0
    try:
        cs = float(getattr(state, 'center_approach_mps', 0.0) or 0.0)
        frame_shape = (CAMH, CAMW, 3)
        in_corridor = roi_in_center_corridor(detection, frame_shape, config)
        # Weight center speed more when object is in center corridor
        try:
            blend_in = float(config.get('speed_detection', {}).get('center_corridor_blend_in', 0.8))
            blend_out = float(config.get('speed_detection', {}).get('center_corridor_blend_out', 0.5))
        except Exception:
            blend_in, blend_out = 0.8, 0.5
        eff_speed = max(eff_speed, cs * (blend_in if in_corridor else blend_out))
    except Exception:
        pass
    # ROI-aware suppression: side-passing unless crossing trajectory
    try:
        frame_shape = (CAMH, CAMW, 3)
        in_corridor = roi_in_center_corridor(detection, frame_shape, config)
        crossing_soon = predict_crossing_to_center(state.object_tracks.get(detection.get('track_id')), frame_shape, config)
        current_depth = detection.get('depth_m')
        if (not in_corridor) and (not crossing_soon):
            if (eff_speed < fast_thr_local) and (current_depth is not None) and (current_depth > nearby_m):
                # In realtime mode, be slightly more permissive
                if not realtime:
                    return False
    except Exception:
        pass

    # Near intersections, suppress non-urgent 'person' spam and prioritize traffic prompts
    try:
        tags = state.environment_context.get('scene_tags', [])
        if ('intersection' in tags) and (str(detection.get('label','')).lower() == 'person'):
            if (current_depth is not None) and (current_depth > critical_m) and (eff_speed < fast_thr_local):
                if not realtime:
                    return False
    except Exception:
        pass
    
    importance = calculate_importance_score(detection, eff_speed, context)
    label = detection['label']
    current_depth = detection['depth_m']

    # Must meet minimum importance threshold
    min_thr = float(state.min_importance_score)
    if realtime:
        min_thr = max(0.4, min_thr * 0.85)
    # Load-aware adaptation: raise threshold under system load to reduce chatter
    try:
        audio_cfg = config.get('audio', {})
        ann_cfg = config.get('announcements', {})
        last_fps = float(getattr(state, 'last_fps', 0.0) or 0.0)
        fps_thresh = float(audio_cfg.get('smart_fps_threshold', 12.0))
        lat_hist = list(state.performance_stats.get('tts_queue_latencies', []))
        recent_lat = float(np.mean(lat_hist[-10:])) if lat_hist[-10:] else 0.0
        lat_thresh = float(audio_cfg.get('smart_queue_latency_s', 0.7))
        boost_mult = float(ann_cfg.get('importance_boost_under_load', 1.15))
        # If either FPS is low or TTS queue latency is high, be stricter
        if ((last_fps > 0 and last_fps < fps_thresh) or (recent_lat > lat_thresh)):
            # In realtime, be slightly less strict to avoid missing hazards
            adj_mult = boost_mult * (0.9 if realtime else 1.0)
            min_thr = min(0.95, max(0.1, min_thr * adj_mult))
    except Exception:
        pass
    # Crowd tag suppression: raise threshold for slow-moving persons in crowds
    try:
        tags = context.get('scene_tags', []) or getattr(state, 'environment_context', {}).get('scene_tags', [])
        tags_lower = [str(t).lower() for t in (tags or [])]
        crowd_mult = float(config.get('announcements', {}).get('crowd_slow_person_importance_mult', 1.0))
        if (crowd_mult > 1.0) and ('crowd' in tags_lower) and (str(detection.get('label','')).lower() == 'person'):
            if (current_depth is not None) and (current_depth > nearby_m) and (eff_speed < fast_thr_local):
                min_thr *= crowd_mult
    except Exception:
        pass
    # Hysteresis gating to avoid flip-flop near threshold
    try:
        ann_cfg = config.get('announcements', {})
        if bool(ann_cfg.get('hysteresis_enabled', True)):
            window_h = float(ann_cfg.get('hysteresis_window_s', 3.0))
            delta_h = float(ann_cfg.get('hysteresis_delta', 0.08))
            # Key by track if available, else label+side
            tid = detection.get('track_id')
            frame_shape = state.frame.shape if state.frame is not None else (CAMH, CAMW, 3)
            try:
                side_key = compute_bearing_side(detection, frame_shape, config)
            except Exception:
                side_key = 'ahead'
            key = ('track', tid) if tid is not None else ('ls', str(label).lower(), side_key)
            if not hasattr(state, '_hysteresis_state'):
                state._hysteresis_state = {}
            last_rec = state._hysteresis_state.get(key)
            if last_rec and ((time.time() - float(last_rec.get('time', 0.0))) <= window_h):
                prev_score = float(last_rec.get('score', 0.0))
                prev_allowed = bool(last_rec.get('allowed', False))
                # If previously allowed, keep allowing unless score drops by > delta
                if prev_allowed and ((prev_score - importance) < delta_h):
                    pass  # treat as allowed (skip strict threshold check)
                else:
                    # If previously suppressed, keep suppressing unless score increases by > delta
                    if (not prev_allowed) and ((importance - prev_score) < delta_h):
                        # Early suppression path
                        return False
            # Fall back to threshold check
    except Exception:
        pass
    if importance < min_thr:
        return False
    
    # Check if depth is valid and stable
    if current_depth is None:
        return False

    # Encounter memory suppression: recently passed object, not critical
    try:
        tid = detection.get('track_id')
        if tid and (tid in state.encounter_memory):
            last_pass = float(state.encounter_memory[tid].get('last_pass', 0.0))
            if (time.time() - last_pass) < 8.0 and (current_depth > critical_m):
                return False
    except Exception:
        pass

    # Context modes gating to reduce chatter
    try:
        mode_name = getattr(state, 'context_mode', None)
        tags = context.get('scene_tags', [])
        if mode_name == 'silent_crowd' and ('crowd' in tags):
            # Stricter threshold in crowd; suppress distant slow persons
            if str(detection.get('label','')).lower() == 'person' and (current_depth > warning_m) and (speed < FAST_SPEED_THRESHOLD):
                return False
            if importance < max(state.min_importance_score, 0.75):
                return False
        elif mode_name == 'indoor_calm':
            movement_recent = 0.0
            try:
                if len(state.user_movement_history) >= 3:
                    movement_recent = float(np.mean(state.user_movement_history[-3:]))
            except Exception:
                movement_recent = 0.0
            if context.get('noise_level', 0.0) < 0.5 and movement_recent < 0.2:
                if importance < max(state.min_importance_score, 0.7):
                    return False
    except Exception:
        pass
    
    # Check if this is a significant change from recent announcements
    recent_announcements = [a for a in state.announcement_history if 
                          time.time() - a['time'] < 15.0]  # Extended to 15 seconds
    
    # Enhanced repetition prevention
    if recent_announcements:
        # Don't repeat similar announcements with minor depth changes
        for recent in recent_announcements:
            if (recent['label'] == label and 
                abs(recent['depth'] - current_depth) < 0.8):  # Increased threshold from 0.5 to 0.8
                # Allow announcement only if depth has significantly decreased (more urgent)
                if current_depth >= recent['depth'] - 0.3:
                    try:
                        with state.lock:
                            state.telemetry['suppressed_ts'].append(time.time())
                    except Exception:
                        pass
                    return False
    
    # Check for rapid-fire announcements
    if len(recent_announcements) >= 2:  # Lowered from 3 to 2 to prevent rapid announcements
        try:
            with state.lock:
                state.telemetry['suppressed_ts'].append(time.time())
        except Exception:
            pass
        return False
    
    # Additional check for stationary objects
    # If object is not moving but depth is fluctuating, don't announce repeatedly
    if eff_speed < 0.4:  # Slow-moving or stationary
        track_id = detection.get('track_id')
        if track_id and track_id in state.object_tracks:
            track_data = state.object_tracks[track_id]
            if len(track_data) > 5:  # Enough history points
                # Calculate depth variance
                depths = [d for _, d, _ in track_data if d is not None]
                if len(depths) > 3:
                    # Check if depth is relatively stable
                    depth_std = np.std(depths[-3:])  # Standard deviation of recent depths
                    if depth_std < 0.25:  # Depth is stable (low variation)
                        # Track-level suppression unless depth has decreased meaningfully
                        last = state.per_track_last_announce.get(track_id)
                        if last and (current_depth >= last.get('depth', current_depth) - 0.4):
                            try:
                                with state.lock:
                                    state.telemetry['suppressed_ts'].append(time.time())
                            except Exception:
                                pass
                            return False
    
    # Label+side dedup window to avoid repeating the same bearing
    try:
        ann_cfg = config.get('announcements', {})
        frame_shape = state.frame.shape if state.frame is not None else (CAMH, CAMW, 3)
        side_key = compute_bearing_side(detection, frame_shape, config)
        dedup_win = float(ann_cfg.get('label_side_dedup_window_s', 3.0))
        if not hasattr(state, '_recent_label_side'):
            state._recent_label_side = {}
        last_ts = float(state._recent_label_side.get((str(label).lower(), side_key), 0.0) or 0.0)
        if (time.time() - last_ts) < dedup_win:
            return False
    except Exception:
        pass

    # Special cases for critical situations
    # Proximity confidence override: high proximity can escalate warnings
    try:
        prox_conf = float(detection.get('proximity_confidence', 0.0))
        prox_warn_thr = float(config.get('proximity', {}).get('warn_gate_threshold', 0.65))
        prox_crit_thr = float(config.get('proximity', {}).get('crit_gate_threshold', 0.85))
    except Exception:
        prox_conf, prox_warn_thr, prox_crit_thr = 0.0, 0.65, 0.85

    if current_depth <= critical_m or prox_conf >= prox_crit_thr:
        return True  # Always announce critical situations
    
    if eff_speed >= VERY_FAST_THRESHOLD or (prox_conf >= prox_warn_thr and eff_speed >= FAST_SPEED_THRESHOLD):
        return True  # Always announce very fast objects

    # Record hysteresis decision
    try:
        ann_cfg = config.get('announcements', {})
        if bool(ann_cfg.get('hysteresis_enabled', True)):
            tid = detection.get('track_id')
            frame_shape = state.frame.shape if state.frame is not None else (CAMH, CAMW, 3)
            try:
                side_key = compute_bearing_side(detection, frame_shape, config)
            except Exception:
                side_key = 'ahead'
            key = ('track', tid) if tid is not None else ('ls', str(label).lower(), side_key)
            if not hasattr(state, '_hysteresis_state'):
                state._hysteresis_state = {}
            state._hysteresis_state[key] = {'time': time.time(), 'score': float(importance), 'allowed': True}
    except Exception:
        pass
    return True


def analyze_lane_paths(depth_map, detections, config):
    """Advanced lane path analysis with multi-lane detection and path scoring"""
    if depth_map is None:
        return []
    
    try:
        h, w = depth_map.shape
        lane_cfg = config.get('lane_guidance', {})
        
        # Enhanced depth analysis with multiple bands
        bands = []
        for i, band_start in enumerate([0.4, 0.5, 0.6, 0.7]):
            band_end = min(0.9, band_start + 0.2)
            y0 = int(h * band_start)
            y1 = int(h * band_end)
            bands.append(depth_map[y0:y1, :])
        
        # Multi-scale free space detection with sensitivity controls
        depth_free_m = float(config.get('free_space', {}).get('depth_free_m', 3.0))
        depth_sensitivity = float(config.get('blind_navigation', {}).get('depth_sensitivity', 0.8))
        depth_free_m *= depth_sensitivity  # Apply sensitivity adjustment
        
        lanes_all_bands = []
        
        for band_idx, band in enumerate(bands):
            if band.size == 0:
                continue
                
            # Adaptive threshold based on band position (closer bands need more clearance)
            band_depth_threshold = depth_free_m * (1.0 + 0.3 * band_idx)
            
            # Free space mask with morphological operations
            free_mask = (np.isfinite(band)) & (band >= band_depth_threshold)
            free_u8 = free_mask.astype(np.uint8) * 255
            
            # Multi-scale morphological operations
            kernel_sizes = [3, 5, 7]
            processed_masks = []
            
            for k_size in kernel_sizes:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                opened = cv2.morphologyEx(free_u8, cv2.MORPH_OPEN, kernel)
                closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
                processed_masks.append(closed)
            
            # Combine multi-scale results with weighted voting
            combined = np.zeros_like(free_u8)
            weights = [0.5, 0.3, 0.2]  # Prefer smaller kernels for precision
            for mask, weight in zip(processed_masks, weights):
                combined += (mask > 0).astype(np.uint8) * weight
            
            # Threshold combined result
            lane_threshold = float(lane_cfg.get('lane_analysis', {}).get('confidence_threshold', 0.6))
            final_mask = (combined >= lane_threshold).astype(np.uint8)
            
            # Extract lane segments
            col_free = final_mask.mean(axis=0) if final_mask.size > 0 else np.zeros(w)
            min_free_ratio = float(config.get('blind_navigation', {}).get('lane_min_free_ratio', 0.85))
            col_binary = (col_free >= min_free_ratio).astype(np.uint8)
            
            # Apply obstacle penalties from YOLO detections
            obstacle_penalty = np.zeros(w, dtype=np.float32)
            for det in detections:
                depth = det.get('depth_m')
                if depth is None or depth > band_depth_threshold:
                    continue
                    
                bbox = det.get('bbox', (0, 0, 0, 0))
                x1, y1, x2, y2 = map(int, bbox)
                
                # Penalty based on proximity and object type
                label = str(det.get('label', '')).lower()
                penalty_factor = 1.0
                if label in ('car', 'bus', 'truck', 'motorcycle'):
                    penalty_factor = 2.0
                elif label in ('person', 'bicycle'):
                    penalty_factor = 1.5
                
                # Distance-based penalty (closer = higher penalty)
                dist_penalty = max(0.1, 1.0 - (depth / band_depth_threshold))
                total_penalty = penalty_factor * dist_penalty
                
                # Apply penalty with spatial spread
                spread = max(5, int((x2 - x1) * 0.5))
                x_start = max(0, x1 - spread)
                x_end = min(w, x2 + spread)
                obstacle_penalty[x_start:x_end] = np.maximum(
                    obstacle_penalty[x_start:x_end], total_penalty
                )
            
            # Adjust lane detection based on penalties
            col_adjusted = np.maximum(0, col_binary - obstacle_penalty)
            
            # Find contiguous lane segments
            min_lane_width = max(1, int(float(config.get('blind_navigation', {}).get('lane_min_width_frac', 0.12)) * w))
            lanes_band = []
            
            i = 0
            while i < w:
                if col_adjusted[i] > 0.5:
                    j = i
                    while j < w and col_adjusted[j] > 0.5:
                        j += 1
                    
                    seg_width = j - i
                    if seg_width >= min_lane_width:
                        # Calculate lane metrics
                        seg_data = band[:, i:j]
                        avg_depth = float(np.nanmean(seg_data)) if seg_data.size > 0 else 0.0
                        depth_std = float(np.nanstd(seg_data)) if seg_data.size > 0 else 0.0
                        
                        # Lane center and side classification
                        cx = (i + j) / 2.0
                        side = 'left' if cx < w * 0.33 else ('right' if cx > w * 0.67 else 'center')
                        
                        # Calculate lane quality score
                        width_score = min(1.0, seg_width / (w * 0.3))  # Prefer wider lanes
                        depth_score = min(1.0, avg_depth / band_depth_threshold)
                        consistency_score = max(0.1, 1.0 - (depth_std / max(0.1, avg_depth)))
                        
                        # Center preference bonus
                        center_distance = abs(cx - w/2.0) / (w/2.0)
                        center_bonus = (1.0 - center_distance) * float(lane_cfg.get('path_scoring', {}).get('center_preference_factor', 0.15))
                        
                        quality_score = (
                            width_score * float(lane_cfg.get('path_scoring', {}).get('width_weight', 0.4)) +
                            depth_score * float(lane_cfg.get('path_scoring', {}).get('depth_weight', 0.3)) +
                            consistency_score * float(lane_cfg.get('path_scoring', {}).get('continuity_weight', 0.1)) +
                            center_bonus
                        )
                        
                        lanes_band.append({
                            'start': i,
                            'end': j,
                            'center': cx,
                            'width_px': seg_width,
                            'width_frac': float(seg_width) / float(w),
                            'avg_depth_m': avg_depth,
                            'depth_std': depth_std,
                            'side': side,
                            'quality_score': quality_score,
                            'band_index': band_idx,
                            'band_weight': 1.0 - (band_idx * 0.2)  # Prefer closer bands
                        })
                    i = j
                else:
                    i += 1
            
            lanes_all_bands.extend(lanes_band)
        
        return lanes_all_bands
        
    except Exception as e:
        return []


def calculate_optimal_path(lanes, config):
    """Select the best traversable path from detected lane segments.

    This function merges nearby lane segments across multiple distance bands,
    computes a comprehensive score per merged lane (width, depth, stability,
    center preference, and merge bonus), and returns the highest scoring lane.
    A simple temporal smoothing is applied to reduce jitter between frames.

    Parameters:
    - lanes: list of dicts describing lane segments produced by lane analysis.
      Expected keys include `start`, `end`, `center`, `width_px`, `width_frac`,
      `avg_depth_m`, `depth_std`, `side`, `quality_score`, `band_index`,
      and `band_weight`.
    - config: runtime configuration dict. Uses `lane_guidance.lane_analysis`
      for merge thresholds and `blind_navigation.path_smoothing_factor` for
      temporal smoothing.

    Returns:
    - dict representing the optimal merged lane with keys such as `center`,
      `width_px`, `width_frac`, `avg_depth_m`, `depth_std`, `side`, and
      `final_score`. Returns None if no suitable lane is found.
    """
    if not lanes:
        return None
    
    try:
        lane_cfg = config.get('lane_guidance', {})
        
        # Group lanes by proximity (merge nearby lanes)
        merge_threshold = float(lane_cfg.get('lane_analysis', {}).get('lane_merge_threshold', 0.3))
        merged_lanes = []
        
        # Sort lanes by center position
        sorted_lanes = sorted(lanes, key=lambda x: x['center'])
        
        i = 0
        while i < len(sorted_lanes):
            current_group = [sorted_lanes[i]]
            j = i + 1
            
            # Merge nearby lanes: cluster by center proximity to form a wider, stable corridor
            while j < len(sorted_lanes):
                center_diff = abs(sorted_lanes[j]['center'] - current_group[-1]['center'])
                if center_diff <= merge_threshold * sorted_lanes[j].get('width_px', 1):
                    current_group.append(sorted_lanes[j])
                    j += 1
                else:
                    break
            
            # Create merged lane from group
            if len(current_group) == 1:
                merged_lanes.append(current_group[0])
            else:
                # Merge multiple lanes into one
                total_weight = sum(lane['quality_score'] * lane['band_weight'] for lane in current_group)
                if total_weight > 0:
                    merged_lane = {
                        'start': min(lane['start'] for lane in current_group),
                        'end': max(lane['end'] for lane in current_group),
                        'center': sum(lane['center'] * lane['quality_score'] * lane['band_weight'] for lane in current_group) / total_weight,
                        'width_px': max(lane['width_px'] for lane in current_group),
                        'width_frac': max(lane['width_frac'] for lane in current_group),
                        'avg_depth_m': sum(lane['avg_depth_m'] * lane['quality_score'] * lane['band_weight'] for lane in current_group) / total_weight,
                        'depth_std': np.mean([lane['depth_std'] for lane in current_group]),
                        'side': current_group[0]['side'],  # Use first lane's side
                        'quality_score': total_weight / len(current_group),
                        'band_index': min(lane['band_index'] for lane in current_group),
                        'band_weight': max(lane['band_weight'] for lane in current_group),
                        'merged_count': len(current_group)
                    }
                    merged_lanes.append(merged_lane)
            
            i = j
        
        # Select best path based on comprehensive scoring
        if not merged_lanes:
            return None
        
        # Enhanced scoring with multiple factors: width, depth, consistency, and merge coverage
        for lane in merged_lanes:
            # Base quality score
            score = lane['quality_score'] * lane['band_weight']

            # Width bonus (prefer wider paths)
            width_bonus = min(0.3, lane['width_frac'] * 2.0)
            
            # Depth bonus (prefer deeper/safer paths)
            depth_bonus = min(0.2, lane['avg_depth_m'] / 5.0)
            
            # Consistency bonus (prefer stable depth)
            consistency_bonus = max(0, 0.15 - lane['depth_std'] * 0.1)
            
            # Merge bonus (prefer paths that span multiple detection bands)
            merge_bonus = lane.get('merged_count', 1) * 0.05
            
            lane['final_score'] = score + width_bonus + depth_bonus + consistency_bonus + merge_bonus
        
        # Select best lane
        best_lane = max(merged_lanes, key=lambda x: x['final_score'])
        
        # Apply path smoothing if enabled
        smoothing_factor = float(config.get('blind_navigation', {}).get('path_smoothing_factor', 0.3))
        if hasattr(state, '_last_optimal_path') and state._last_optimal_path:
            last_center = state._last_optimal_path.get('center', best_lane['center'])
            best_lane['center'] = (
                smoothing_factor * last_center + 
                (1.0 - smoothing_factor) * best_lane['center']
            )
        
        state._last_optimal_path = best_lane
        return best_lane
        
    except Exception as e:
        return None


def generate_steering_guidance(optimal_path, frame_width, config):
    """Translate the optimal path into steering guidance and confidence.

    Maps the lateral offset of the chosen path center to a bounded steering
    angle, classifies the correction (straight/gentle/sharp), and computes a
    confidence score based on path quality, width, and average depth.

    Parameters:
    - optimal_path: dict returned by `calculate_optimal_path`. Must contain
      `center`, `width_frac`, `avg_depth_m`, and may include `final_score`.
    - frame_width: integer width of the current frame in pixels.
    - config: runtime configuration dict. Reads
      `lane_guidance.steering_guidance` for angle thresholds and
      `blind_navigation` for responsiveness and confidence heuristics.

    Returns:
    - dict with keys:
      `steer_angle_deg`, `guidance_type`, `confidence`, `guidance_message`,
      plus lane metadata echoes (`lane_width_frac`, `avg_depth_m`, `side`).
      On failure or when no path is available, returns a safe neutral guidance.
    """
    if not optimal_path:
        return {
            'steer_angle_deg': 0.0,
            'guidance_type': 'none',
            'confidence': 0.0,
            'guidance_message': 'No clear path detected'
        }
    
    try:
        steering_cfg = config.get('lane_guidance', {}).get('steering_guidance', {})
        blind_nav_cfg = config.get('blind_navigation', {})
        
        # Calculate steering angle
        center = frame_width / 2.0
        path_center = optimal_path['center']
        offset = path_center - center
        
        # Normalize offset to [-1, 1]
        normalized_offset = offset / center
        
        # Apply steering angle mapping
        max_angle = float(steering_cfg.get('max_steering_angle_deg', 30.0))
        steer_angle = normalized_offset * max_angle
        # Responsiveness scaling (0-1 typical; >1 for more aggressive)
        responsiveness = float(blind_nav_cfg.get('steering_responsiveness', 0.7))
        steer_angle *= responsiveness
        steer_angle = max(-max_angle, min(max_angle, steer_angle))
        
        # Determine guidance type and message
        gentle_threshold = float(steering_cfg.get('gentle_correction_threshold_deg', 5.0))
        sharp_threshold = float(steering_cfg.get('sharp_correction_threshold_deg', 15.0))
        
        abs_angle = abs(steer_angle)
        
        if abs_angle < gentle_threshold:
            guidance_type = 'straight'
            message = 'straight_ahead'
        elif abs_angle < sharp_threshold:
            guidance_type = 'gentle'
            message = 'slight_left' if steer_angle < 0 else 'slight_right'
        else:
            guidance_type = 'sharp'
            message = 'turn_left' if steer_angle < 0 else 'turn_right'
        
        # Calculate confidence based on path quality and width
        base_confidence = optimal_path.get('final_score', 0.5)
        width_pref = float(blind_nav_cfg.get('wide_path_preference', 1.5))
        width_confidence = min(1.0, optimal_path.get('width_frac', 0.1) * 3.0 * width_pref)
        depth_confidence = min(1.0, optimal_path.get('avg_depth_m', 1.0) / 3.0)
        # Penalize narrow passages
        narrow_thr = float(blind_nav_cfg.get('narrow_path_threshold', 0.15))
        if optimal_path.get('width_frac', 0.0) < narrow_thr:
            base_confidence *= 0.7
        
        overall_confidence = (base_confidence + width_confidence + depth_confidence) / 3.0
        # Enforce minimum threshold
        min_conf = float(blind_nav_cfg.get('confidence_threshold', 0.25))
        overall_confidence = max(0.0, min(1.0, overall_confidence))
        if overall_confidence < min_conf:
            guidance_type = 'none'
            message = 'path_blocked'
            steer_angle = 0.0
        
        return {
            'steer_angle_deg': float(steer_angle),
            'guidance_type': guidance_type,
            'confidence': float(overall_confidence),
            'guidance_message': message,
            # Align keys with consumers
            'lane_width_frac': optimal_path.get('width_frac', 0.0),
            'avg_depth_m': optimal_path.get('avg_depth_m', 0.0),
            'side': optimal_path.get('side', 'center'),
            # Preserve legacy keys for compatibility
            'path_width_frac': optimal_path.get('width_frac', 0.0),
            'path_depth_m': optimal_path.get('avg_depth_m', 0.0),
            'path_side': optimal_path.get('side', 'center')
        }
        
    except Exception as e:
        return {
            'steer_angle_deg': 0.0,
            'guidance_type': 'error',
            'confidence': 0.0,
            'guidance_message': 'path_blocked'
        }


def provide_lane_guidance_audio(nav_guidance, config, current_time):
    """Queue spoken lane guidance with urgency-aware rate limiting.

    Determines whether guidance should be spoken based on recent timing,
    guidance type (straight/gentle/sharp), confidence, and angle magnitude.
    When eligible, enqueues a single guidance message and logs telemetry.

    Parameters:
    - nav_guidance: dict from `generate_steering_guidance`.
    - config: runtime configuration dict. Uses
      `lane_guidance.audio_feedback` for phrases and timing.
    - current_time: float epoch seconds used for rate limiting and logging.

    Returns:
    - None. Side effects: updates `state.audio_queue` and
      `state._last_lane_guidance_time`; emits telemetry via `log_telemetry`.
    """
    try:
        lane_cfg = config.get('lane_guidance', {})
        audio_cfg = lane_cfg.get('audio_feedback', {})
        
        if not lane_cfg.get('audio_guidance_enabled', True):
            return
        
        # Check if enough time has passed since last guidance
        guidance_interval = float(audio_cfg.get('guidance_interval_s', 2.0))
        urgent_interval = float(audio_cfg.get('urgent_guidance_interval_s', 1.0))
        
        last_guidance_time = getattr(state, '_last_lane_guidance_time', 0.0)
        time_since_last = current_time - last_guidance_time
        
        guidance_type = nav_guidance.get('guidance_type', 'none')
        confidence = float(nav_guidance.get('confidence', 0.0))
        steer_angle = abs(float(nav_guidance.get('steer_angle_deg', 0.0)))
        
        # Determine if guidance is urgent (sharp turns or low confidence)
        is_urgent = (guidance_type == 'sharp') or (confidence < 0.4) or (steer_angle > 20.0)
        required_interval = urgent_interval if is_urgent else guidance_interval
        
        if time_since_last < required_interval:
            return
        
        # Get guidance message
        message_key = nav_guidance.get('guidance_message', 'straight_ahead')
        phrases = audio_cfg.get('guidance_phrases', {})
        guidance_text = phrases.get(message_key, message_key.replace('_', ' '))
        
        # Add context for narrow passages or blocked paths
        path_width = float(nav_guidance.get('lane_width_frac', 0.0))
        if path_width > 0 and path_width < 0.15:
            guidance_text = phrases.get('narrow_passage', 'Narrow passage ahead') + '. ' + guidance_text
        elif confidence < 0.3:
            guidance_text = phrases.get('path_blocked', 'Path blocked')
        
        # Enqueue audio guidance
        if guidance_text and confidence > 0.2:
            try:
                # Create guidance announcement
                guidance_announcement = {
                    'text': guidance_text,
                    'priority': 'guidance',
                    'urgency': 'high' if is_urgent else 'medium',
                    'source': 'lane_guidance',
                    'confidence': confidence,
                    'steer_angle': steer_angle,
                    'timestamp': current_time
                }
                
                # Speak via central audio system using priority queue
                try:
                    speak_status(guidance_text, priority=(0 if is_urgent else 2))
                except Exception:
                    pass
                
                # Log telemetry for guidance
                log_telemetry({
                    'event': 'lane_guidance_provided',
                    'guidance_type': guidance_type,
                    'message': message_key,
                    'confidence': confidence,
                    'steer_angle_deg': float(nav_guidance.get('steer_angle_deg', 0.0)),
                    'path_width_frac': path_width,
                    'is_urgent': is_urgent,
                    'timestamp': current_time
                })
                
                state._last_lane_guidance_time = current_time
                
            except Exception as e:
                pass
                
    except Exception as e:
        pass


def update_context(detections, current_time):
    """Update environmental context for better decision making"""
    # Expire temporary focus center and restore default ROI
    try:
        if hasattr(state, '_focus_center_until') and state._focus_center_until:
            if time.time() > float(state._focus_center_until):
                state.roi_ratio = 0.7
                state._focus_center_until = 0.0
    except Exception:
        pass
    # Calculate scene stability (how much the scene has changed)
    if len(detections) > 0:
        # Simple stability metric based on detection count and positions
        detection_count = len(detections)
        if hasattr(state, 'last_detection_count'):
            stability_change = abs(detection_count - state.last_detection_count) / max(1, state.last_detection_count)
            state.environment_context['scene_stability'] = max(0.1, 1.0 - stability_change)
        state.last_detection_count = detection_count
    
    # Update noise level (too many detections = noisy environment)
    noise = min(1.0, len(detections) / 10.0)
    # Boost in explicit crowd contexts or crowd mode
    try:
        tags = state.environment_context.get('scene_tags', [])
        if ('crowd' in tags) or (getattr(state, 'context_mode', None) == 'silent_crowd'):
            noise = max(noise, 0.8)
    except Exception:
        pass
    state.environment_context['noise_level'] = noise
    
    # Simple semantic tags for announcer
    tags = []
    person_count = sum(1 for d in detections if d.get('label') == 'person')
    if person_count >= 5:
        tags.append('crowd')
    if any(d.get('label') in ('traffic light', 'stop sign') for d in detections):
        tags.append('intersection')
    # Crosswalk detection (heuristic pattern of horizontal stripes near bottom of frame)
    try:
        frame_local = None
        with state.lock:
            if state.frame is not None:
                frame_local = state.frame
        if frame_local is not None:
            h, w = frame_local.shape[:2]
            y0 = int(h * 0.55)
            roi = frame_local[y0:h, :]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=int(w * 0.3), maxLineGap=15)
            horiz_count = 0
            if lines is not None:
                for l in lines:
                    x1, y1, x2, y2 = l[0]
                    dy = abs(y2 - y1)
                    dx = abs(x2 - x1)
                    if dx > 0 and (dy / float(dx + 1e-6)) < 0.15:  # near horizontal
                        horiz_count += 1
            # Require enough horizontal segments to indicate striped pattern
            if horiz_count >= 4:
                tags.append('crosswalk')
    except Exception:
        pass
    # Unknown environment tag based on positioning switch and stability drop
    try:
        pos = getattr(state, 'positioning', {})
        indoor = bool(pos.get('indoor', False))
        provider = str(pos.get('provider', '') or '')
        stability = float(state.environment_context.get('scene_stability', 1.0) or 1.0)
        if indoor and (provider in ('cv', 'wifi', 'ble')) and (stability < 0.5) and ('intersection' not in tags) and ('crosswalk' not in tags):
            tags.append('unknown_env')
    except Exception:
        pass
        # Fuse crowd-sourced hints into scene tags, preserving local-first behavior
    try:
        tags = _fuse_hints_into_scene_tags(tags)
    except Exception:
        pass
    state.environment_context['scene_tags'] = tags
    
    # Track user movement (simplified - could be enhanced with IMU data)
    if len(detections) > 0:
        # Assume user movement if detection positions change significantly
        current_centers = [d['center'] for d in detections if d.get('center')]
        if len(current_centers) > 0 and hasattr(state, 'last_centers'):
            movement = np.mean([np.linalg.norm(np.array(c1) - np.array(c2)) 
                              for c1, c2 in zip(current_centers, state.last_centers)])
            state.user_movement_history.append(movement)
            if len(state.user_movement_history) > 10:
                state.user_movement_history.pop(0)
        state.last_centers = current_centers
        # Tune ROI ratio based on recent movement (clamped)
        try:
            if len(state.user_movement_history) >= 3:
                recent = float(np.mean(state.user_movement_history[-3:]))
                # Convert pixel movement to a small adjustment; clamp to [0.6, 0.9]
                target_roi = 0.6 + min(0.3, max(0.0, recent) * 0.002)
                state.roi_ratio = max(0.6, min(0.9, target_roi))
        except Exception:
            pass

    # Compute ROI occupancy balance (left vs right)
    try:
        left = 0
        right = 0
        for d in detections:
            c = d.get('center')
            if not c:
                continue
            w = state.frame.shape[1] if state.frame is not None else CAMW
            if c[0] < w * 0.5:
                left += 1
            else:
                right += 1
        total = max(1, left + right)
        state.environment_context['roi_balance'] = {
            'left': left / total,
            'right': right / total
        }
    except Exception:
        pass

    # Free-space corridor detection from depth map (left/right open band)
    try:
        dm = state.depth_map_m
        if dm is not None:
            h, w = dm.shape
            # Focus on lower-middle band where path matters
            top = int(h * 0.45)
            bottom = int(h * 0.90)
            band = dm[top:bottom, :]
            # Near obstacle mask within warning distance
            near_mask = np.isfinite(band) & (band <= WARNING_DISTANCE_M)
            # Column occupancy (fraction of rows that are near)
            col_occ = near_mask.mean(axis=0) if near_mask.size > 0 else np.zeros(w)
            half = max(1, w // 2)
            # Columns considered free if near occupancy is low
            occ_thresh = 0.15
            left_free_cols = np.where(col_occ[:half] < occ_thresh)[0]
            right_free_cols = np.where(col_occ[half:] < occ_thresh)[0]
            left_width = len(left_free_cols) / float(half)
            right_width = len(right_free_cols) / float(w - half)
            side = None
            width = 0.0
            # Require sufficient free width and adequate average depth
            min_width = 0.5
            min_avg_depth = 2.5
            try:
                avg_left_depth = float(np.nanmean(band[:, :half]))
                avg_right_depth = float(np.nanmean(band[:, half:]))
            except Exception:
                avg_left_depth = avg_right_depth = 0.0
            if left_width >= min_width and avg_left_depth >= min_avg_depth and left_width >= right_width:
                side, width = 'left', float(left_width)
            elif right_width >= min_width and avg_right_depth >= min_avg_depth:
                side, width = 'right', float(right_width)
            state.environment_context['safe_corridor'] = {'side': side, 'width': width}
    except Exception:
        pass

    # Occupancy-grid free-space with morphological opening and persistence
    try:
        dm = state.depth_map_m
        if dm is not None:
            h, w = dm.shape
            depth_free_m = float(config.get('free_space', {}).get('depth_free_m', 3.0))
            ker = int(config.get('free_space', {}).get('kernel_open', 5))
            band_h_frac = float(config.get('free_space', {}).get('center_band_height_frac', 0.22))
            min_corr_frac = float(config.get('free_space', {}).get('min_corridor_frac', 0.55))
            persist_s = float(config.get('free_space', {}).get('persistence_s', 1.2))
            # Adjust thresholds based on travel mode for dynamic adaptation
            try:
                mode = str(getattr(state, 'travel_mode', 'WALKING') or 'WALKING').upper()
                if mode in ('DRIVING', 'CYCLING'):
                    depth_free_m *= 1.3
                    ker = max(3, int(ker * 1.1))
            except Exception:
                pass
            # Center horizontal band
            band_h = max(4, int(h * band_h_frac))
            y0 = (h - band_h) // 2
            y1 = y0 + band_h
            band = dm[y0:y1, :]
            # Free if beyond threshold and finite
            free_mask = (np.isfinite(band)) & (band >= depth_free_m)
            free_u8 = free_mask.astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ker, ker))
            opened = cv2.morphologyEx(free_u8, cv2.MORPH_OPEN, kernel)
            # Column-wise free ratio
            col_free = (opened > 0).mean(axis=0) if opened.size > 0 else np.zeros(w)
            # Consider columns free if ratio exceeds 0.85
            lane_thr = float(config.get('blind_navigation', {}).get('lane_min_free_ratio', 0.85))
            col_bin = (col_free >= lane_thr).astype(np.uint8)
            # Obstacle mask from YOLO detections: block columns with near obstacles
            try:
                obstacle_mask = np.zeros(w, dtype=np.uint8)
                bn = config.get('blind_navigation', {})
                # Use warning distance for blocking threshold
                mode = getattr(state, 'travel_mode', 'WALKING')
                factor = 1.0
                if str(mode).upper() in ('DRIVING', 'CYCLING'):
                    factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
                block_m = WARNING_DISTANCE_M * factor
                # Slight dilation radius in columns for safety
                dilate_cols = int(bn.get('obstacle_dilate_cols', 8))
                for d in detections:
                    dep = d.get('depth_m')
                    if dep is None:
                        continue
                    if float(dep) <= float(block_m):
                        x1, y1, x2, y2 = map(int, d.get('bbox', (0,0,0,0)))
                        i0 = max(0, min(w-1, x1 - dilate_cols))
                        i1 = max(0, min(w, x2 + dilate_cols))
                        obstacle_mask[i0:i1] = 1
                # Suppress free columns where obstacles present
                col_nav = np.where(obstacle_mask == 1, 0, col_bin)
            except Exception:
                col_nav = col_bin
            corr_width = float(col_bin.sum()) / float(max(1, w))
            # Persistence logic
            cp = state.environment_context.get('clear_path', {'persist': False, 'corridor_frac': 0.0, 'last_true_since': 0.0})
            nowt = time.time()
            if corr_width >= min_corr_frac:
                if cp.get('last_true_since', 0.0) <= 0.0:
                    cp['last_true_since'] = nowt
                cp['corridor_frac'] = corr_width
                cp['persist'] = (nowt - cp['last_true_since']) >= persist_s
            else:
                cp['last_true_since'] = 0.0
                cp['persist'] = False
                cp['corridor_frac'] = corr_width
            state.environment_context['clear_path'] = cp

            # Enhanced lane detection with multi-lane analysis and path scoring
            try:
                lane_cfg = config.get('lane_guidance', {})
                if lane_cfg.get('enabled', True) and lane_cfg.get('multi_lane_detection', True):
                    # Use enhanced lane analysis
                    detected_lanes = analyze_lane_paths(dm, detections, config)
                    state.environment_context['lanes'] = detected_lanes
                    
                    # Calculate optimal path
                    optimal_path = calculate_optimal_path(detected_lanes, config)
                    
                    # Generate steering guidance
                    if optimal_path:
                        steering_guidance = generate_steering_guidance(optimal_path, w, config)
                        
                        # Enhanced navigation guidance with more detailed information
                        state.environment_context['nav_guidance'] = {
                            'side': optimal_path.get('side', 'center'),
                            'lane_width_frac': float(optimal_path.get('width_frac', 0.0)),
                            'avg_depth_m': float(optimal_path.get('avg_depth_m', 0.0)),
                            'steer_angle_deg': steering_guidance['steer_angle_deg'],
                            'confidence': steering_guidance['confidence'],
                            'guidance_type': steering_guidance['guidance_type'],
                            'guidance_message': steering_guidance['message'],
                            'path_quality_score': float(optimal_path.get('final_score', 0.0)),
                            'path_consistency': float(1.0 - optimal_path.get('depth_std', 1.0)),
                            'available_lanes': len(detected_lanes),
                            'timestamp': nowt
                        }
                        
                        # Store detailed path information for telemetry
                        state.environment_context['optimal_path_details'] = {
                            'center_px': float(optimal_path.get('center', w/2.0)),
                            'width_px': int(optimal_path.get('width_px', 0)),
                            'depth_std': float(optimal_path.get('depth_std', 0.0)),
                            'band_coverage': int(optimal_path.get('merged_count', 1)),
                            'quality_components': {
                                'width_score': min(1.0, optimal_path.get('width_frac', 0.0) * 3.0),
                                'depth_score': min(1.0, optimal_path.get('avg_depth_m', 0.0) / 3.0),
                                'final_score': float(optimal_path.get('final_score', 0.0))
                            }
                        }
                    else:
                        # No optimal path found
                        state.environment_context['nav_guidance'] = {
                            'side': None,
                            'lane_width_frac': 0.0,
                            'avg_depth_m': 0.0,
                            'steer_angle_deg': 0.0,
                            'confidence': 0.0,
                            'guidance_type': 'blocked',
                            'guidance_message': 'path_blocked',
                            'path_quality_score': 0.0,
                            'path_consistency': 0.0,
                            'available_lanes': len(detected_lanes),
                            'timestamp': nowt
                        }
                else:
                    # Fallback to basic lane detection
                    lanes = []
                    min_lane_w = max(1, int(float(config.get('blind_navigation', {}).get('lane_min_width_frac', 0.12)) * w))
                    i = 0
                    while i < w:
                        if col_nav[i] == 1:
                            j = i
                            while j < w and col_nav[j] == 1:
                                j += 1
                            seg_w = j - i
                            if seg_w >= min_lane_w:
                                seg = band[:, i:j]
                                try:
                                    avg_depth = float(np.nanmean(seg))
                                except Exception:
                                    avg_depth = 0.0
                                cx = (i + j) / 2.0
                                side = 'left' if cx < w * 0.33 else ('right' if cx > w * 0.67 else 'center')
                                lanes.append({
                                    'start': i,
                                    'end': j,
                                    'width_frac': float(seg_w) / float(w),
                                    'avg_depth_m': avg_depth,
                                    'side': side
                                })
                            i = j
                        else:
                            i += 1
                    state.environment_context['lanes'] = lanes

                    # Basic steering guidance
                    if lanes:
                        best = max(lanes, key=lambda L: (L['width_frac'], L['avg_depth_m']))
                        cx = (best['start'] + best['end']) / 2.0
                        center = w / 2.0
                        norm = (cx - center) / center
                        max_deg = float(config.get('blind_navigation', {}).get('max_steer_deg', 28.0))
                        steer_deg = float(max(-max_deg, min(max_deg, norm * max_deg)))
                        confidence = float(0.5 * best['width_frac'] + 0.5 * min(1.0, best['avg_depth_m'] / max(1e-3, depth_free_m)))
                        state.environment_context['nav_guidance'] = {
                            'side': best['side'],
                            'lane_width_frac': float(best['width_frac']),
                            'avg_depth_m': float(best['avg_depth_m']),
                            'steer_angle_deg': steer_deg,
                            'confidence': confidence,
                            'guidance_type': 'basic',
                            'guidance_message': 'straight_ahead',
                            'timestamp': nowt
                        }
                    else:
                        state.environment_context['nav_guidance'] = {
                            'side': None,
                            'lane_width_frac': 0.0,
                            'avg_depth_m': 0.0,
                            'steer_angle_deg': 0.0,
                            'confidence': 0.0,
                            'guidance_type': 'none',
                            'guidance_message': 'path_blocked',
                            'timestamp': nowt
                        }
            except Exception:
                pass
    except Exception:
        pass

    # Detect fast approaching hazards (vehicles or looming center depth)
    try:
        boost = False
        # Vehicle approach via tracked speed
        for d in detections:
            label = str(d.get('label','')).lower()
            if label in ('car','bus','truck','motorcycle','bicycle'):
                tid = d.get('track_id')
                if tid is None:
                    continue
                track = state.object_tracks.get(tid, [])
                spd = calculate_object_speed(track, current_time)
                dep = d.get('depth_m')
                if dep is None and state.depth_map_m is not None:
                    dep = get_bbox_median_depth(d['bbox'], state.depth_map_m)
                try:
                    th = float(config['hazard']['looming_depth_threshold_m'])
                    logging.info(f"Looming vehicle check: tid={tid} speed={spd:.2f}m/s depth_m={dep} depth_th={th}")
                except Exception:
                    logging.info(f"Looming vehicle check: tid={tid} speed={spd:.2f}m/s depth_m={dep}")
                if spd >= FAST_SPEED_THRESHOLD and dep is not None and dep <= float(config['hazard']['looming_depth_threshold_m']):
                    boost = True
                    break
        # Looming fallback using center depth change
        if not boost:
            center_speed = float(getattr(state, 'center_approach_mps', 0.0) or 0.0)
            center_depth = getattr(state, 'center_depth_m', None)
            try:
                thd = float(config['hazard']['looming_depth_threshold_m'])
                ths = float(config['hazard']['looming_speed_threshold_mps'])
                logging.info(f"Center looming check: speed={center_speed:.2f}m/s depth_m={center_depth} speed_th={ths} depth_th={thd}")
            except Exception:
                logging.info(f"Center looming check: speed={center_speed:.2f}m/s depth_m={center_depth}")
            if center_depth is not None and center_speed >= float(config['hazard']['looming_speed_threshold_mps']) and center_depth <= float(config['hazard']['looming_depth_threshold_m']):
                boost = True
        # Activate boost window
        if boost:
            state.hazard_boost_active_until = max(state.hazard_boost_active_until, current_time + float(config['hazard']['boost_duration_s']))
            # Prefer smaller imgsz during boost for speed
            try:
                state._yolo_img_override = int(config['hazard']['yolo_imgsz_boost'])
            except Exception:
                state._yolo_img_override = None
        else:
            # Allow override to clear when boost expires
            if current_time >= state.hazard_boost_active_until:
                state._yolo_img_override = None
    except Exception:
        pass

    # Estimate a simple safe corridor using depth map slices
    try:
        dm = state.depth_map_m
        if dm is not None:
            h, w = dm.shape
            # Slices: left third, center third, right third
            thirds = [
                dm[:, : w // 3],
                dm[:, w // 3: 2 * w // 3],
                dm[:, 2 * w // 3:]
            ]
            mode = getattr(state, 'travel_mode', 'WALKING')
            factor = 1.0
            if str(mode).upper() in ('DRIVING', 'CYCLING'):
                factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
            warning_m = WARNING_DISTANCE_M * factor
            safe_fracs = []
            for sl in thirds:
                # Fraction of pixels considered safe (beyond warning)
                safe_fracs.append(float(np.mean(sl > warning_m)))
            idx = int(np.argmax(safe_fracs))
            side = {0: 'left', 1: 'ahead', 2: 'right'}[idx]
            width = safe_fracs[idx]
            state.environment_context['safe_corridor'] = {'side': side, 'width': width}
        else:
            state.environment_context['safe_corridor'] = {'side': None, 'width': 0.0}
    except Exception:
        pass

    # Scene tags (crowd, intersection heuristic)
    try:
        persons = [d for d in detections if d.get('label','').lower() == 'person']
        near_persons = [d for d in persons if d.get('depth_m') is not None and d['depth_m'] <= 6.0]
        tags = []
        if len(near_persons) >= 4:
            tags.append('crowd')
        has_tl = any(d.get('label','').lower() == 'traffic light' for d in detections)
        has_stop = any(d.get('label','').lower() == 'stop sign' for d in detections)
        if has_tl or has_stop:
            tags.append('intersection')
        # Add moving_camera tag when IMU indicates significant device motion
        try:
            imu = state.imu
            ego_motion = float(np.linalg.norm(imu.get('user_acceleration',[0,0,0]))) + 0.05*float(np.linalg.norm(imu.get('rotation_rate',[0,0,0])))
            if ego_motion > state.movement_threshold:
                tags.append('moving_camera')
        except Exception:
            pass
        # Fuse crowd-sourced hints into scene tags, preserving local-first behavior
        try:
            tags = _fuse_hints_into_scene_tags(tags)
        except Exception:
            pass
        state.environment_context['scene_tags'] = tags
    except Exception:
        pass

    # Real-world stats for adaptive blind-priority tuning
    try:
        bn = config.get('blind_navigation', {})
        if bool(bn.get('adaptive_tuning_enabled', True)) and detections:
            # Initialize stats with exponential decay over tuning_window_s
            with state.lock:
                if not hasattr(state, 'nav_weight_stats') or not isinstance(state.nav_weight_stats, dict):
                    state.nav_weight_stats = {
                        'trip': {'score': 0.0, 'last_ts': 0.0},
                        'overhead': {'score': 0.0, 'last_ts': 0.0},
                        'path': {'score': 0.0, 'last_ts': 0.0}
                    }
                tw = float(bn.get('tuning_window_s', 900.0))
                def decay(prev, dt):
                    try:
                        lam = max(1e-6, 1.0 / tw)
                        return prev * math.exp(-lam * dt)
                    except Exception:
                        return prev
                nowt = time.time()
                for d in detections:
                    center_x = float(d.get('center_x', 0.5) or 0.5)
                    bottom = float(d.get('bottom', 0.0) or 0.0)
                    top = float(d.get('top', 0.0) or 0.0)
                    depth_m = float(d.get('depth_m', 10.0) or 10.0)
                    # Trip hazard: low object within 3m
                    if bottom > 0.7 and depth_m <= 3.0:
                        s = state.nav_weight_stats['trip']
                        dt = (nowt - s['last_ts']) if s['last_ts'] else 0.0
                        s['score'] = decay(s['score'], dt) + 1.0
                        s['last_ts'] = nowt
                    # Overhead hazard: upper third within 3m
                    if top < 0.3 and depth_m <= 3.0:
                        s = state.nav_weight_stats['overhead']
                        dt = (nowt - s['last_ts']) if s['last_ts'] else 0.0
                        s['score'] = decay(s['score'], dt) + 1.0
                        s['last_ts'] = nowt
                    # Path obstacle: center lane within 4m
                    path_w = float(bn.get('path_width_factor', 0.2))
                    if (0.5 - path_w) < center_x < (0.5 + path_w) and depth_m <= 4.0:
                        s = state.nav_weight_stats['path']
                        dt = (nowt - s['last_ts']) if s['last_ts'] else 0.0
                        s['score'] = decay(s['score'], dt) + 1.0
                        s['last_ts'] = nowt
                # Normalize to frequency within [0,1]
                total = sum(v['score'] for v in state.nav_weight_stats.values())
                if total > 0:
                    for k, v in state.nav_weight_stats.items():
                        v['freq'] = max(0.0, min(1.0, v['score'] / total))
                else:
                    for k in state.nav_weight_stats:
                        state.nav_weight_stats[k]['freq'] = 0.0
    except Exception:
        pass


def detect_road_crossing_scenario(detections, depth_map_m, frame):
    """Detect if user is about to cross a road based on multiple cues"""
    scenario = {
        'detected': False,
        'confidence': 0.0,
        'urgency': 'low',
        'cues': []
    }
    
    try:
        # Get configuration parameters
        ctx_cfg = config.get('contextual_scenarios', {})
        if not ctx_cfg.get('enabled', True) or not ctx_cfg.get('road_crossing', {}).get('enabled', True):
            return scenario
            
        road_cfg = ctx_cfg.get('road_crossing', {})
        traffic_range = float(road_cfg.get('traffic_detection_range_m', 15.0))
        vehicle_proximity = float(road_cfg.get('vehicle_proximity_threshold_m', 8.0))
        terrain_std_threshold = float(road_cfg.get('terrain_change_std_threshold', 2.0))
        confidence_threshold = float(road_cfg.get('confidence_threshold', 0.5))
        
        # Check for traffic-related objects
        traffic_objects = ['car', 'bus', 'truck', 'motorcycle', 'bicycle', 'traffic light', 'stop sign']
        nearby_traffic = [d for d in detections 
                         if d.get('label', '').lower() in traffic_objects 
                         and d.get('depth_m', float('inf')) <= traffic_range]
        
        if nearby_traffic:
            scenario['cues'].append('traffic_nearby')
            scenario['confidence'] += 0.3
        
        # Check for crosswalk pattern (already implemented in update_context)
        tags = state.environment_context.get('scene_tags', [])
        if 'crosswalk' in tags:
            scenario['cues'].append('crosswalk_detected')
            scenario['confidence'] += 0.4
        
        # Check for intersection markers
        if 'intersection' in tags:
            scenario['cues'].append('intersection_detected')
            scenario['confidence'] += 0.3
        
        # Analyze movement pattern - if user is moving toward road edge
        if frame is not None and depth_map_m is not None:
            h, w = frame.shape[:2]
            # Check bottom third for road edge (sudden depth change)
            bottom_third = depth_map_m[int(h * 0.67):, :]
            if bottom_third.size > 0:
                # Look for depth discontinuities that might indicate curb/road edge
                valid_depths = bottom_third[np.isfinite(bottom_third)]
                if len(valid_depths) > 0:
                    depth_std = np.std(valid_depths)
                    if depth_std > terrain_std_threshold:  # High variation suggests terrain change
                        scenario['cues'].append('terrain_change')
                        scenario['confidence'] += 0.2
        
        # Determine urgency based on proximity of moving vehicles
        moving_vehicles = [d for d in nearby_traffic 
                          if d.get('label', '').lower() in ['car', 'bus', 'truck', 'motorcycle']
                          and d.get('depth_m', float('inf')) <= vehicle_proximity]
        
        if moving_vehicles:
            scenario['urgency'] = 'high'
            scenario['confidence'] += 0.2
        elif nearby_traffic:
            scenario['urgency'] = 'medium'
        
        scenario['detected'] = scenario['confidence'] >= confidence_threshold
        
    except Exception as e:
        pass
    
    return scenario





def detect_crowded_area_scenario(detections, depth_map_m, frame):
    """Detect if user is entering a crowded area"""
    scenario = {
        'detected': False,
        'confidence': 0.0,
        'urgency': 'medium',
        'cues': [],
        'person_count': 0,
        'density': 0.0
    }
    
    try:
        # Get configuration parameters
        ctx_cfg = config.get('contextual_scenarios', {})
        if not ctx_cfg.get('enabled', True) or not ctx_cfg.get('crowded_area', {}).get('enabled', True):
            return scenario
            
        crowd_cfg = ctx_cfg.get('crowded_area', {})
        person_range = float(crowd_cfg.get('person_detection_range_m', 10.0))
        low_crowd = int(crowd_cfg.get('low_crowd_threshold', 3))
        medium_crowd = int(crowd_cfg.get('medium_crowd_threshold', 5))
        high_crowd = int(crowd_cfg.get('high_crowd_threshold', 8))
        moving_speed = float(crowd_cfg.get('moving_person_speed_threshold_mps', 0.5))
        bottleneck_width = float(crowd_cfg.get('bottleneck_corridor_width_threshold', 0.4))
        confidence_threshold = float(crowd_cfg.get('confidence_threshold', 0.4))
        
        # Count nearby people
        people = [d for d in detections 
                 if d.get('label', '').lower() == 'person'
                 and d.get('depth_m', float('inf')) <= person_range]
        
        scenario['person_count'] = len(people)
        
        # Calculate crowd density
        if frame is not None:
            frame_area = frame.shape[0] * frame.shape[1]
            total_person_area = sum(
                (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1])
                for d in people if 'bbox' in d
            )
            scenario['density'] = total_person_area / frame_area if frame_area > 0 else 0.0
        
        # Determine crowd level
        if scenario['person_count'] >= high_crowd:
            scenario['cues'].append('high_crowd_density')
            scenario['confidence'] += 0.7
            scenario['urgency'] = 'high'
        elif scenario['person_count'] >= medium_crowd:
            scenario['cues'].append('medium_crowd_density')
            scenario['confidence'] += 0.5
        elif scenario['person_count'] >= low_crowd:
            scenario['cues'].append('low_crowd_density')
            scenario['confidence'] += 0.3
        
        # Check for crowd movement patterns
        moving_people = []
        for person in people:
            tid = person.get('track_id')
            if tid and tid in state.object_tracks:
                track = state.object_tracks[tid]
                speed = calculate_object_speed(track, time.time())
                if speed > moving_speed:  # Moving person
                    moving_people.append(person)
        
        if len(moving_people) >= 3:
            scenario['cues'].append('crowd_movement')
            scenario['confidence'] += 0.2
        
        # Check for bottleneck situations (narrow passage with people)
        if depth_map_m is not None and people:
            # Analyze free space width
            safe_corridor = state.environment_context.get('safe_corridor', {})
            corridor_width = safe_corridor.get('width', 1.0)
            
            if corridor_width < bottleneck_width and scenario['person_count'] >= 2:
                scenario['cues'].append('bottleneck_detected')
                scenario['confidence'] += 0.3
                scenario['urgency'] = 'high'
        
        # Check existing crowd tag from update_context
        tags = state.environment_context.get('scene_tags', [])
        if 'crowd' in tags:
            scenario['cues'].append('crowd_tag_active')
            scenario['confidence'] += 0.4
        
        scenario['detected'] = scenario['confidence'] >= confidence_threshold
        
    except Exception as e:
        pass
    
    return scenario


def analyze_contextual_scenarios(detections, current_time):
    """Analyze all contextual scenarios and return combined assessment"""
    scenarios = {}
    
    try:
        frame_local = None
        depth_map_local = None
        
        with state.lock:
            if state.frame is not None:
                frame_local = state.frame.copy()
            if state.depth_map_m is not None:
                depth_map_local = state.depth_map_m.copy()
        
        # Detect each scenario type
        scenarios['road_crossing'] = detect_road_crossing_scenario(detections, depth_map_local, frame_local)
        scenarios['crowded_area'] = detect_crowded_area_scenario(detections, depth_map_local, frame_local)
        
        # Determine overall context and priority
        active_scenarios = [name for name, scenario in scenarios.items() if scenario['detected']]
        
        # Calculate combined urgency
        max_urgency = 'low'
        if any(scenarios[s]['urgency'] == 'high' for s in active_scenarios):
            max_urgency = 'high'
        elif any(scenarios[s]['urgency'] == 'medium' for s in active_scenarios):
            max_urgency = 'medium'
        
        # Store in environment context
        state.environment_context['contextual_scenarios'] = {
            'active': active_scenarios,
            'details': scenarios,
            'max_urgency': max_urgency,
            'timestamp': current_time
        }
        
        # Log telemetry for scenario detection
        try:
            if active_scenarios:
                telemetry_data = {
                    'timestamp': current_time,
                    'event': 'contextual_scenario_detected',
                    'scenarios': active_scenarios,
                    'urgency': max_urgency,
                    'details': {name: {
                        'confidence': scenarios[name]['confidence'],
                        'cues': scenarios[name]['cues']
                    } for name in active_scenarios}
                }
                log_telemetry(telemetry_data)
        except Exception:
            pass
            
    except Exception as e:
        pass
    
    return scenarios


def announcement_thread():
    # AI-powered announcement system that thinks like a human assistant
    windows_by_phrase = {}
    # Add object-specific depth smoothing
    object_depth_history = {}
    
    while state.running:
        time.sleep(0.1)
        # Heartbeat for watchdog liveness tracking
        try:
            with state.lock:
                state._last_announce_tick = time.time()
        except Exception:
            pass
        # Prefer queued detections and depth for synchronized consumption
        dets = []
        depth_map = None
        with state.lock:
            if not state.enable_announce:
                continue
            # Fallback values from shared state
            dets_state = list(state.detections)
            depth_state = state.depth_map_m
        try:
            if getattr(state, 'pipelines_async_enabled', False):
                dp = _queue_get_latest(state.detection_queue)
                dm = _queue_get_latest(state.depth_map_queue)
                if isinstance(dp, dict) and ('detections' in dp):
                    dets = dp.get('detections', []) or []
                else:
                    dets = dets_state
                if isinstance(dm, dict) and ('depth' in dm):
                    depth_map = dm.get('depth', None)
                else:
                    depth_map = depth_state
            else:
                dets = dets_state
                depth_map = depth_state
        except Exception:
            dets = dets_state
            depth_map = depth_state
        now = time.time()
        # Apply fixed cadence if configured (forces repeat every fixed interval)
        try:
            fc = float(config.get('announcements', {}).get('fixed_cadence_s', 0.0) or 0.0)
            if fc > 0:
                state.announce_cooldown_s = fc
        except Exception:
            pass
        cooldown_ok = (now - state.last_announce_time) >= state.announce_cooldown_s
        # Immediate alert if camera appears covered/dark
        try:
            if state.frame is not None:
                def _is_camera_occluded(frame, cfg):
                    try:
                        if frame.ndim == 3 and frame.shape[2] == 3:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = frame
                        low_luma_thr = float(cfg.get('camera', {}).get('cover_low_luma_thresh', 50))
                        low_frac_thr = float(cfg.get('camera', {}).get('cover_low_frac_thresh', 0.45))
                        std_thr = float(cfg.get('camera', {}).get('cover_std_thresh', 12.0))
                        edge_frac_thr = float(cfg.get('camera', {}).get('cover_edge_frac_thresh', 0.008))
                        low_frac = float((gray < low_luma_thr).sum()) / float(gray.size)
                        try:
                            edges = cv2.Canny(gray, 50, 150)
                            edge_frac = float((edges > 0).sum()) / float(edges.size)
                        except Exception:
                            edge_frac = 0.0
                        mean_val = float(np.mean(gray))
                        std_val = float(np.std(gray))
                        if mean_val < 30.0:
                            return True
                        if (low_frac >= low_frac_thr and (edge_frac <= edge_frac_thr or std_val <= std_thr)):
                            return True
                        return False
                    except Exception:
                        return False

                if _is_camera_occluded(state.frame, config) and (now - max(0.0, float(getattr(state, '_last_camera_cover_alert', 0.0) or 0.0))) > 8.0:
                    speak_status("Warning! Camera appears to be covered.", priority=10)
                    state._last_camera_cover_alert = now
                    state.last_announce_time = now
                    # Persist suppression for clear-path announcements while camera is dark/covered
                    try:
                        state._camera_dark_until = now + 6.0
                    except Exception:
                        pass
        except Exception:
            pass
        # Depth may be unavailable; proceed using YOLO-only detections and safe fallbacks
        depth_available = depth_map is not None

        # Update context for intelligent decision making
        try:
            update_context(dets, now)
        except Exception as e:
            try:
                logging.warning(f"update_context error: {e}")
            except Exception:
                pass
            continue
        
        # Analyze contextual scenarios for adaptive alerting
        try:
            analyze_contextual_scenarios(dets, now)
        except Exception as e:
            try:
                logging.warning(f"analyze_contextual_scenarios error: {e}")
            except Exception:
                pass
        spoken_this_tick = False
        # Instant hazard path: announce closest critical obstacle immediately, bypassing cooldown
        try:
            if bool(config.get('announcements', {}).get('instant_hazard', True)) and dets:
                mode = getattr(state, 'travel_mode', 'WALKING')
                factor = 1.0
                if str(mode).upper() in ('DRIVING', 'CYCLING'):
                    factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
                # Apply user sensitivity scaling
                try:
                    prefs = _get_active_alert_prefs()
                    sens_scale = float(prefs.get('sensitivity_scale', 1.0) or 1.0)
                except Exception:
                    sens_scale = 1.0
                critical_m = CRITICAL_DISTANCE_M * factor * max(0.5, min(2.0, sens_scale))
                w = state.frame.shape[1] if state.frame is not None else CAMW
                # Pick high-confidence detections and fuse depth from ROI under each bbox
                candidates = []
                for d in dets:
                    # Confidence threshold modulated by sensitivity (lower sensitivity -> higher threshold)
                    try:
                        conf_scale = 1.0
                        if sens_scale < 1.0:
                            conf_scale = 1.05 + (1.0 - sens_scale) * 0.2
                        elif sens_scale > 1.0:
                            conf_scale = 1.0 - min(0.3, (sens_scale - 1.0) * 0.15)
                    except Exception:
                        conf_scale = 1.0
                    if d.get('conf', 0.0) < state.min_announce_conf * 0.8 * conf_scale:
                        continue
                    x1, y1, x2, y2 = map(int, d.get('bbox', (0,0,0,0)))
                    try:
                        h, w_dm = depth_map.shape[:2]
                        x1c = max(0, min(w_dm-1, x1))
                        x2c = max(0, min(w_dm, x2))
                        y1c = max(0, min(h-1, y1))
                        y2c = max(0, min(h, y2))
                        roi = depth_map[y1c:y2c, x1c:x2c]
                        if roi.size > 0:
                            valid = roi[roi > 0]
                            if valid.size > 0:
                                fused = float(np.percentile(valid, 20))
                            else:
                                fused = float(d.get('depth_m') or np.inf)
                        else:
                            fused = float(d.get('depth_m') or np.inf)
                    except Exception:
                        fused = float(d.get('depth_m') or np.inf)
                    d['depth_m_fused'] = fused
                    # Consider as candidate if fused depth is within 1.2x critical distance
                    if fused <= critical_m * 1.2:
                        candidates.append(d)
                # Near-field fallback: treat very large boxes as critical even if depth missing
                if not candidates:
                    for d in dets:
                        x1, y1, x2, y2 = map(int, d.get('bbox', (0,0,0,0)))
                        bw = max(1, x2 - x1)
                        bh = max(1, y2 - y1)
                        fw = w
                        fh = state.frame.shape[0] if state.frame is not None else CAMH
                        if (bw / float(fw)) >= 0.55 or (bh / float(fh)) >= 0.55:
                            candidates.append(d)
                    # Ensure these fallback candidates get a synthesized close depth for messaging
                    for d in candidates:
                        if d.get('depth_m') is None:
                            d['depth_m'] = max(getattr(state, 'min_m', 0.2), 0.2)
                            d['depth_m_fused'] = d['depth_m']
                if candidates:
                    candidates.sort(key=lambda d: float(d.get('depth_m_fused', d.get('depth_m', np.inf))))
                    d = candidates[0]
                    frame_shape = state.frame.shape if state.frame is not None else (CAMH, CAMW, 3)
                    side = compute_bearing_side(d, frame_shape, config)
                    fine_side = compute_bearing_fine_side(d, frame_shape, config)
                    side_phrase = 'slightly left' if fine_side == 'slight_left' else (
                        'slightly right' if fine_side == 'slight_right' else (
                            'center' if fine_side == 'center' else side
                        )
                    )
                    label = d.get('label','object')
                    # Force announcement with higher priority, include fused distance and severity
                    try:
                        ddepth = float(d.get('depth_m_fused', d.get('depth_m', 0.0)) or 0.0)
                    except Exception:
                        ddepth = 0.0
                    prefix = "Critical!" if ddepth <= critical_m else "Warning!"
                    # Multimodal alert: haptic + spatial earcon + gated voice
                    try:
                        ang_deg, pan_val = compute_angle_pan_from_detection(d, frame_shape, config)
                        _set_spatial_pan_hint(ang_deg, pan_val)
                    except Exception:
                        pass
                    # Suppress hazard voice when camera is dark/covered or depth is unreliable
                    try:
                        camera_dark_active = (time.time() < float(getattr(state, '_camera_dark_until', 0.0) or 0.0))
                    except Exception:
                        camera_dark_active = False
                    roi_valid_ok = True
                    try:
                        # Validate depth within detection bbox to avoid noise-triggered false positives
                        if depth_map is not None:
                            bx1, by1, bx2, by2 = d.get('bbox', (0, 0, 0, 0))
                            h, w = depth_map.shape[:2]
                            x1 = int(max(0, min(w - 1, bx1)))
                            x2 = int(max(0, min(w, bx2)))
                            y1 = int(max(0, min(h - 1, by1)))
                            y2 = int(max(0, min(h, by2)))
                            if x2 > x1 and y2 > y1:
                                roi = depth_map[y1:y2, x1:x2]
                                total = roi.size
                                valid = int((roi > 0).sum()) if total > 0 else 0
                                valid_ratio = (valid / float(total)) if total > 0 else 0.0
                                # Use configurable thresholds with safe defaults
                                min_ratio = float(config.get('hazard', {}).get('min_depth_valid_ratio', 0.02))
                                min_roi_frac = float(config.get('hazard', {}).get('min_roi_valid_fraction', min_ratio))
                                min_roi_pixels = int(config.get('hazard', {}).get('min_roi_valid_pixels', 50))
                                roi_valid_ok = (valid_ratio >= min_roi_frac) and (valid >= min_roi_pixels)
                    except Exception:
                        pass
                    if (not camera_dark_active) and roi_valid_ok:
                        # Qualitative distance phrasing (no numeric meters)
                        dist_phrase = "too close." if ddepth <= critical_m else "near."
                        emit_multimodal_alert('critical' if ddepth <= critical_m else 'warning', side=side, depth_m=ddepth,
                                              msg=f"{prefix} {label} {side_phrase}, {dist_phrase}", priority=10)
                        with state.lock:
                            now_i = time.time()
                            state.last_announce_time = now_i
                            state.last_announcement_time = now_i
                            # Short block tuned by severity to prevent chatter
                            block_s = float(config.get('audio', {}).get('global_silence_after_critical_s', 2.5)) if ddepth <= critical_m else float(config.get('audio', {}).get('hint_block_after_critical_s', 2.0))
                            state._global_silence_until = now_i + block_s
                            # Avoid frame skipping briefly to stabilize
                            try:
                                state._avoid_frame_skip_until = max(state._avoid_frame_skip_until, now_i + float(config.get('hazard', {}).get('boost_duration_s', 1.6)))
                            except Exception:
                                state._avoid_frame_skip_until = now_i + 1.6
                            # Activate hazard boost to prioritize depth refresh and YOLO speed
                            try:
                                bd = float(config.get('hazard', {}).get('boost_duration_s', 1.6))
                                state.hazard_boost_active_until = max(getattr(state, 'hazard_boost_active_until', 0.0), now_i + bd)
                                try:
                                    state._yolo_img_override = int(config.get('hazard', {}).get('yolo_imgsz_boost'))
                                except Exception:
                                    state._yolo_img_override = None
                            except Exception:
                                pass
                        spoken_this_tick = True
                    else:
                        # Record suppression reason for diagnostics
                        try:
                            with state.lock:
                                state.telemetry['events'].append({
                                    'time': time.time(),
                                    'type': 'multimodal_alert',
                                    'level': 'critical' if ddepth <= critical_m else 'warning',
                                    'side': side,
                                    'depth_m': float(ddepth or 0.0),
                                    'spoken': False,
                                    'reason': 'camera_dark' if camera_dark_active else 'depth_invalid_roi'
                                })
                        except Exception:
                            pass
        except Exception:
            pass
        # Contextual critical event guidance (crosswalks, intersections, unknown env)
        try:
            if not spoken_this_tick:
                tags = state.environment_context.get('scene_tags', [])
                now_i = time.time()
                # Crosswalk guidance
                if ('crosswalk' in tags):
                    last_ts = float(getattr(state, '_last_crosswalk_guidance', 0.0) or 0.0)
                    cd = float(config.get('announcements', {}).get('crosswalk_guidance_cooldown_s', 12.0))
                    if (now_i - last_ts) >= cd and (now_i >= getattr(state, '_global_silence_until', 0.0)):
                        emit_multimodal_alert('warning', side='center', depth_m=float(getattr(state, 'center_depth_m', 0.0) or 0.0),
                                              msg='Crosswalk ahead. Align center and wait for safe signal.', priority=7)
                        state._last_crosswalk_guidance = now_i
                        spoken_this_tick = True
                # Intersection guidance
                if (not spoken_this_tick) and ('intersection' in tags):
                    last_ts = float(getattr(state, '_last_intersection_guidance', 0.0) or 0.0)
                    cd = float(config.get('announcements', {}).get('intersection_guidance_cooldown_s', 10.0))
                    if (now_i - last_ts) >= cd and (now_i >= getattr(state, '_global_silence_until', 0.0)):
                        emit_multimodal_alert('warning', side='center', depth_m=float(getattr(state, 'center_depth_m', 0.0) or 0.0),
                                              msg='Intersection ahead. Watch for traffic; use crosswalk if present.', priority=7)
                        state._last_intersection_guidance = now_i
                        spoken_this_tick = True
                # Unknown environment guidance on indoor transition
                if (not spoken_this_tick) and ('unknown_env' in tags):
                    last_ts = float(getattr(state, '_last_unknown_env_prompt', 0.0) or 0.0)
                    cd = float(config.get('announcements', {}).get('unknown_env_prompt_cooldown_s', 15.0))
                    if (now_i - last_ts) >= cd and (now_i >= getattr(state, '_global_silence_until', 0.0)):
                        emit_multimodal_alert('warning', side='center', depth_m=float(getattr(state, 'center_depth_m', 0.0) or 0.0),
                                              msg='Entering unknown indoor environment. Proceed cautiously; I’ll guide locally.', priority=6)
                        state._last_unknown_env_prompt = now_i
                        spoken_this_tick = True
        except Exception:
            pass
        # Depth-only fallback when YOLO misses but near obstacle is present
        try:
            if (not spoken_this_tick) and depth_available:
                mode = getattr(state, 'travel_mode', 'WALKING')
                factor = 1.0
                if str(mode).upper() in ('DRIVING', 'CYCLING'):
                    factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
                critical_m = CRITICAL_DISTANCE_M * factor
                dm = depth_map
                h, w = dm.shape[:2]
                corr_frac = float(config.get('hazard', {}).get('path_width_factor', 0.25))
                corr_frac = max(0.1, min(0.8, corr_frac))
                x1 = int(max(0, (0.5 - corr_frac / 2.0) * w))
                x2 = int(min(w, (0.5 + corr_frac / 2.0) * w))
                center_roi = dm[:, x1:x2]
                center_valid = center_roi[center_roi > 0]
                center_min = float(np.min(center_valid)) if center_valid.size else np.inf
                # side estimation by comparing left and right halves
                left_valid = dm[:, :w//2]
                left_valid = left_valid[left_valid > 0]
                right_valid = dm[:, w//2:]
                right_valid = right_valid[right_valid > 0]
                left_min = float(np.min(left_valid)) if left_valid.size else np.inf
                right_min = float(np.min(right_valid)) if right_valid.size else np.inf
                side = 'left' if left_min < right_min else ('right' if right_min < left_min else 'center')
                # Suppress during camera-dark window and require sufficient valid fraction in center ROI
                try:
                    camera_dark_active = (time.time() < float(getattr(state, '_camera_dark_until', 0.0) or 0.0))
                except Exception:
                    camera_dark_active = False
                valid_frac = 0.0
                try:
                    total = center_roi.size
                    valid_cnt = center_valid.size
                    valid_frac = (valid_cnt / float(total)) if total > 0 else 0.0
                except Exception:
                    valid_frac = 0.0
                min_ratio = float(config.get('hazard', {}).get('min_depth_valid_ratio', 0.02))
                depth_reliable = valid_frac >= min_ratio
                # Require sufficient mass of pixels below critical to avoid noise
                min_below_frac = float(config.get('hazard', {}).get('min_below_critical_fraction', 0.06))
                min_below_pixels = int(config.get('hazard', {}).get('min_below_critical_pixels', 200))
                try:
                    below_mask = (center_roi > 0) & (center_roi <= critical_m)
                    below_cnt = int(np.count_nonzero(below_mask))
                except Exception:
                    below_cnt = 0
                below_frac = (below_cnt / float(total)) if total > 0 else 0.0
                near_mass_ok = (below_frac >= min_below_frac) or (below_cnt >= min_below_pixels)
                # Persistence window to avoid one-off spikes
                persist_s = float(config.get('hazard', {}).get('fallback_persist_s', 0.3))
                try:
                    if near_mass_ok:
                        if getattr(state, '_center_below_since', 0.0) == 0.0:
                            state._center_below_since = time.time()
                    else:
                        state._center_below_since = 0.0
                except Exception:
                    pass
                persist_ok = (getattr(state, '_center_below_since', 0.0) > 0.0) and ((time.time() - getattr(state, '_center_below_since', 0.0)) >= persist_s)
                if depth_reliable and near_mass_ok and persist_ok and (not camera_dark_active):
                    if center_min <= critical_m:
                        # Use qualitative phrasing for distance
                        emit_multimodal_alert('critical', side=side, depth_m=center_min,
                                              msg=f"Critical! Obstacle ahead {side}, too close.", priority=10)
                        with state.lock:
                            now_i = time.time()
                            state.last_announce_time = now_i
                            state.last_announcement_time = now_i
                            block_s = float(config.get('audio', {}).get('global_silence_after_critical_s', 2.5))
                            state._global_silence_until = now_i + block_s
                        spoken_this_tick = True
                    elif center_min <= critical_m * 1.4:
                        emit_multimodal_alert('warning', side=side, depth_m=center_min,
                                              msg=f"Warning! Obstacle ahead {side}, near.", priority=9)
                        with state.lock:
                            now_i = time.time()
                            state.last_announce_time = now_i
                            state.last_announcement_time = now_i
                            block_s = float(config.get('audio', {}).get('hint_block_after_critical_s', 2.0))
                            state._global_silence_until = now_i + block_s
                        spoken_this_tick = True
                else:
                    try:
                        with state.lock:
                            state.telemetry['events'].append({
                                'time': time.time(),
                                'type': 'multimodal_alert',
                                'level': 'critical' if center_min <= critical_m else 'warning',
                                'side': side,
                                'depth_m': float(center_min if np.isfinite(center_min) else 0.0),
                                'spoken': False,
                                'reason': (
                                    'camera_dark' if camera_dark_active else (
                                        'depth_invalid_center' if (not depth_reliable) else (
                                            'near_mass_low' if (not near_mass_ok) else 'persist_short'
                                        )
                                    )
                                )
                            })
                    except Exception:
                        pass
        except Exception:
            pass
        # Looming fallback alert when YOLO misses but center depth shows fast approach
        try:
            center_speed = float(getattr(state, 'center_approach_mps', 0.0) or 0.0)
            center_depth = getattr(state, 'center_depth_m', None)
            looming_cd = float(config['hazard']['looming_cooldown_s'])
            # Persistence: require speed above threshold to persist and depth to be decreasing beyond deadband
            speed_thr = float(config['hazard']['looming_speed_threshold_mps'])
            persist_ok = False
            if center_speed >= speed_thr:
                if state._center_speed_above_since == 0.0:
                    state._center_speed_above_since = now
                persist_ok = (now - state._center_speed_above_since) >= 0.6
            else:
                state._center_speed_above_since = 0.0

            depth_deadband_ok = False
            try:
                last_cd = float(getattr(state, '_last_center_depth', center_depth) or 0.0)
                if (center_depth is not None) and (last_cd is not None):
                    depth_deadband_ok = (last_cd - center_depth) >= 0.3
            except Exception:
                depth_deadband_ok = False

            if (not spoken_this_tick) and (center_depth is not None) and persist_ok and depth_deadband_ok and (center_depth <= float(config['hazard']['looming_depth_threshold_m'])) and ((now - state._last_looming_alert) >= looming_cd):
                emit_multimodal_alert('warning', side='center', depth_m=float(center_depth or 0.0),
                                      msg="Fast object approaching ahead. Caution!", priority=8)
                with state.lock:
                    state._last_looming_alert = now
                    state.last_announce_time = now
                spoken_this_tick = True
        except Exception:
            pass
        # Adaptive cooldown multiplier when recent warnings happened
        ann_cfg = config.get('announcements', {})
        fixed_cd = float(ann_cfg.get('fixed_cadence_s', 0.0) or 0.0)
        base_cd = float(state.announce_cooldown_s)
        hint_mult = float(config.get('audio', {}).get('hint_cooldown_multiplier_in_warning', 2.0))
        recent_warning = (now - state._last_warning_time) < 6.0
        # Smart cadence: scale cooldown by scene noise_level (longer in crowds)
        noise_lvl = float(state.environment_context.get('noise_level', 0.0) or 0.0)
        # Shorter in sparse scenes, longer in crowds
        cadence_mult = 0.8 + 1.0 * max(0.0, min(1.0, noise_lvl))
        if fixed_cd > 0.0:
            effective_cd = fixed_cd
        else:
            effective_cd = base_cd * (hint_mult if recent_warning else 1.0) * cadence_mult
        # Clamp cooldown to avoid accumulating long silences under load
        try:
            if getattr(state, 'realtime_mode', False):
                effective_cd = max(0.25, min(effective_cd, 1.0))
            else:
                effective_cd = max(0.5, min(effective_cd, 2.0))
        except Exception:
            effective_cd = max(0.5, min(effective_cd, 2.0))
        # Under system load, extend cooldown to reduce chatter
        try:
            ann_cfg = config.get('announcements', {})
            audio_cfg = config.get('audio', {})
            if bool(ann_cfg.get('adaptive_cooldown', True)):
                last_fps = float(getattr(state, 'last_fps', 0.0) or 0.0)
                fps_thresh = float(audio_cfg.get('smart_fps_threshold', 12.0))
                lat_hist = list(getattr(state, 'performance_stats', {}).get('tts_queue_latencies', []))
                recent_lat = float(np.mean(lat_hist[-10:])) if lat_hist[-10:] else 0.0
                lat_thresh = float(audio_cfg.get('smart_queue_latency_s', 0.7))
                load_mult = float(ann_cfg.get('cooldown_under_load_multiplier', 1.5))
                if ((last_fps > 0 and last_fps < fps_thresh) or (recent_lat > lat_thresh)):
                    effective_cd *= max(1.0, load_mult)
                    # In realtime mode, keep cooldown responsive
                    if getattr(state, 'realtime_mode', False):
                        effective_cd = min(effective_cd, 0.6)
        except Exception:
            pass
        # In realtime mode, tighten cooldown for faster response
        try:
            if getattr(state, 'realtime_mode', False):
                effective_cd = min(effective_cd, 0.4)
        except Exception:
            pass
        # Reset global silence if no recent warnings (prevents extended mute)
        try:
            if (now - max(0.0, float(state._last_warning_time or 0.0))) > 8.0:
                state._global_silence_until = 0.0
        except Exception:
            pass

        # Lateral earcons: play side cues for warning-level obstacles; voice only for criticals
        try:
            if dets:
                mode = getattr(state, 'travel_mode', 'WALKING')
                factor = 1.0
                if str(mode).upper() in ('DRIVING', 'CYCLING'):
                    factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
                critical_m = CRITICAL_DISTANCE_M * factor
                warning_m = WARNING_DISTANCE_M * factor
                w = state.frame.shape[1] if state.frame is not None else CAMW
                # Per-side earcon cooldown
                if not hasattr(state, '_last_earcon_side'):
                    state._last_earcon_side = {'left': 0.0, 'right': 0.0}
                earcon_cd = 0.9
                for d in dets:
                    depth = d.get('depth_m')
                    if depth is None:
                        continue
                    if depth <= critical_m:
                        continue  # voice-only for criticals
                    # Require proximity confidence to suppress weak/non-threatening objects
                    min_prox = float(config.get('proximity', {}).get('earcon_min_prox_conf', 0.35))
                    if depth <= warning_m and d.get('conf', 0.0) >= state.min_announce_conf and float(d.get('proximity_confidence', 0.0) or 0.0) >= min_prox:
                        cx = d.get('center', (w//2, 0))[0]
                        side = 'left' if cx < w * 0.4 else 'right' if cx > w * 0.6 else 'center'
                        if side == 'center':
                            continue
                        last_side = state._last_earcon_side.get(side, 0.0)
                        if (time.time() - last_side) >= earcon_cd:
                            lvl = 'warning' if depth <= warning_m else 'nearby'
                            try:
                                play_spatial_earcon(lvl, side=side, depth_m=float(depth or 0.0))
                            except Exception:
                                play_earcon(lvl, side=side)
                            state._last_earcon_side[side] = time.time()
        except Exception:
            pass
        
        # Handle different announcement strategies based on AI decider setting
        if state.ai_decider:
            realtime = bool(getattr(state, 'realtime_mode', False))
            # Opportunistically announce global scene tags
            tags = state.environment_context.get('scene_tags', [])
            # Respect global silence window for non-critical hints
            if (not realtime) and (not spoken_this_tick) and (now >= state._global_silence_until) and ((time.time() - state.last_announce_time) >= effective_cd) and tags:
                if 'intersection' in tags:
                    speak_status("Intersection ahead. Use caution.")
                    with state.lock:
                        state.last_announce_time = now
                    spoken_this_tick = True
                elif 'crowd' in tags:
                    speak_status("Crowd ahead. Slow down and stay aware.")
                    with state.lock:
                        state.last_announce_time = now
                    spoken_this_tick = True

            # Strong clear-path announcement when contiguous walkable corridor persists
            try:
                clear_ctx = state.environment_context.get('clear_path', {})
                persist_ok = bool(clear_ctx.get('persist', False))
                corridor_frac = float(clear_ctx.get('corridor_frac', 0.0) or 0.0)
                min_corridor = float(config['free_space']['min_corridor_frac'])
                # Avoid announcing in crowded scenes or right after warnings
                crowded = ('crowd' in state.environment_context.get('scene_tags', []))
                recent_warning = (now - state._last_warning_time) < 6.0
                # Own cooldown to prevent repeats even if global cooldown resets
                last_cp = float(getattr(state, '_last_clear_path_announce', 0.0) or 0.0)
                cp_cd = 7.5
                cp_cooldown_ok = (now - last_cp) >= cp_cd
                global_ok = (now >= state._global_silence_until) and ((time.time() - state.last_announce_time) >= effective_cd)
                # Suppress clear-path announcements if the camera is currently dark/covered
                camera_dark_active = (time.time() < float(getattr(state, '_camera_dark_until', 0.0) or 0.0))
                if (not realtime) and (not spoken_this_tick) and persist_ok and (corridor_frac >= min_corridor) and (not crowded) and (not recent_warning) and cp_cooldown_ok and global_ok and (not camera_dark_active):
                    # Prefer directional guidance using nav_guidance when available
                    nav = state.environment_context.get('nav_guidance', {})
                    side = nav.get('side', 'ahead')
                    steer = float(nav.get('steer_angle_deg', 0.0) or 0.0)
                    depth_m = float(nav.get('avg_depth_m', 0.0) or 0.0)
                    msg = None
                    try:
                        if abs(steer) >= 5.0 and side and side != 'center':
                            dir_word = 'left' if steer < 0 else 'right'
                            msg = f"Clear path {side}. Veer {abs(steer):.0f}° {dir_word}."
                            try:
                                speak_async(msg)
                            except Exception:
                                pass
                        elif side == 'center':
                            msg = "Clear path ahead."
                            try:
                                speak_async(msg)
                            except Exception:
                                pass
                        else:
                            msg = f"Clear path {side}."
                            try:
                                speak_async(msg)
                            except Exception:
                                pass
                    except Exception:
                        msg = "Clear path ahead."
                        try:
                            speak_async(msg)
                        except Exception:
                            pass
                    emit_multimodal_alert('info', side='left' if steer < 0 else ('right' if steer > 0 else 'center'), depth_m=depth_m, msg=msg, priority=5)
                    with state.lock:
                        state._last_clear_path_announce = now
                        state.last_announce_time = now
                    spoken_this_tick = True
            except Exception:
                pass
            
            # Provide lane guidance audio feedback
            try:
                nav_guidance = state.environment_context.get('nav_guidance', {})
                if nav_guidance and nav_guidance.get('confidence', 0.0) > 0.0:
                    provide_lane_guidance_audio(nav_guidance, config, now)
            except Exception:
                pass
        

            # Proximity summary (every ~9s if multiple relevant objects)
            try:
                # Adaptive summary interval by scene/mode
                summary_base = float(config.get('audio', {}).get('summary_interval_s', 9.0))
                if 'crowd' in state.environment_context.get('scene_tags', []):
                    summary_base += 3.0
                if str(getattr(state, 'travel_mode', 'WALKING')).upper() == 'DRIVING':
                    summary_base += 6.0
                if (not realtime) and (not spoken_this_tick) and (now - state._last_summary >= summary_base) and (now >= state._global_silence_until) and ((time.time() - state.last_announce_time) >= effective_cd):
                    # Bucket by type and distance ranges
                    near = [d for d in dets if d.get('depth_m') is not None and d['depth_m'] <= 5.0 and d['conf'] >= state.min_announce_conf]
                    if len(near) >= 2:
                        people = [d for d in near if d['label'].lower() == 'person']
                        vehicles = [d for d in near if d['label'].lower() in ('car','bus','truck','motorcycle','bicycle')]
                        # Directional approaching vehicle (priority stacking cue)
                        approach_side = None
                        try:
                            w = state.frame.shape[1] if state.frame is not None else CAMW
                            for d in vehicles:
                                tid = d.get('track_id')
                                if tid and tid in state.object_tracks:
                                    spd = calculate_object_speed(state.object_tracks[tid], now)
                                    if spd >= max(0.8, FAST_SPEED_THRESHOLD * 0.7):
                                        cx = d.get('center', (w//2, 0))[0]
                                        approach_side = 'left' if cx < w*0.45 else 'right' if cx > w*0.55 else 'ahead'
                                        break
                        except Exception:
                            pass
                        # Bearing of overall cluster
                        bearing = 'ahead'
                        try:
                            w = state.frame.shape[1] if state.frame is not None else CAMW
                            cxs = [d['center'][0] for d in near if d.get('center')]
                            if cxs:
                                avg_cx = float(np.mean(cxs))
                                bearing = 'left' if avg_cx < w*0.45 else 'right' if avg_cx > w*0.55 else 'ahead'
                        except Exception:
                            pass
                        msg_parts = []
                        if people:
                            msg_parts.append(f"{len(people)} people nearby")
                        if vehicles:
                            if approach_side and approach_side != 'ahead':
                                msg_parts.append(f"car approaching {approach_side}")
                            else:
                                msg_parts.append(f"vehicle(s) nearby")
                        if msg_parts:
                            speak_status(f"{' , '.join(msg_parts)} {bearing}.")
                            with state.lock:
                                state._last_summary = now
                                state.last_announce_time = now
                            spoken_this_tick = True
            except Exception:
                pass

            # Walkable corridor announcement with confidence and cooldown
            try:
                corridor = state.environment_context.get('safe_corridor', {'side': None, 'width': 0.0})
                side = corridor.get('side')
                width = corridor.get('width', 0.0)
                crowd_tags = state.environment_context.get('scene_tags', [])
                if (not realtime) and (not spoken_this_tick) and side and (width >= 0.5) and recent_warning and ('crowd' not in crowd_tags) and (now - state._last_clearance >= 10.0) and (now >= state._global_silence_until) and ((time.time() - state.last_announce_time) >= effective_cd):
                    speak_status(f"Walkable corridor on the {side}.")
                    with state.lock:
                        state._last_clearance = now
                        state.last_announce_time = now
                    spoken_this_tick = True
            except Exception:
                pass

            # Lane guidance announcement: suggest best lane if multiple lanes available
            try:
                bn = config.get('blind_navigation', {})
                if bool(bn.get('lane_guidance_enabled', True)) and (not realtime) and (not spoken_this_tick):
                    lanes = state.environment_context.get('lanes', [])
                    if lanes and (now >= state._global_silence_until) and ((time.time() - state.last_announce_time) >= effective_cd):
                        # Prefer widest lane, break ties by farthest average depth
                        lanes_sorted = sorted(lanes, key=lambda L: (L.get('width_frac', 0.0), L.get('avg_depth_m', 0.0)), reverse=True)
                        best = lanes_sorted[0]
                        # Require minimum width to avoid over-guidance
                        if best.get('width_frac', 0.0) >= float(bn.get('lane_min_width_frac', 0.12)):
                            # Suppress if recent warning to avoid overload; otherwise announce
                            if not recent_warning:
                                side = best.get('side', 'center')
                                if side == 'center':
                                    speak_status("Center lane clear.")
                                else:
                                    speak_status(f"Keep to the {side} lane.")
                                with state.lock:
                                    state.last_announce_time = now
                                spoken_this_tick = True
            except Exception:
                pass

            # Bearing correction prompt when imbalance persists (suppressed if central close person present)
            try:
                bal = state.environment_context.get('roi_balance', {'left':0.0,'right':0.0})
                diff = abs(bal.get('left', 0.0) - bal.get('right', 0.0))
                # Suppress if a central close person is present
                suppress_bearing = False
                try:
                    frame_shape = state.frame.shape if state.frame is not None else (CAMH, CAMW, 3)
                    mode = getattr(state, 'travel_mode', 'WALKING')
                    factor = 1.0
                    if str(mode).upper() in ('DRIVING', 'CYCLING'):
                        factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
                    warning_m = WARNING_DISTANCE_M * factor
                    for d in dets:
                        if d.get('label') == 'person' and (d.get('depth_m') is not None):
                            if roi_in_center_corridor(d, frame_shape, config) and (d['depth_m'] <= max(2.5, warning_m + 0.5)):
                                suppress_bearing = True
                                break
                except Exception:
                    suppress_bearing = False
                # Require nearby obstacles to avoid false prompts
                nearby_present = False
                try:
                    mode = getattr(state, 'travel_mode', 'WALKING')
                    factor = 1.0
                    if str(mode).upper() in ('DRIVING', 'CYCLING'):
                        factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
                    nearby_m = NEARBY_DISTANCE_M * factor
                    # Improved wall detection - lower threshold and check more aggressively
                    for d in dets:
                        if d.get('depth_m') is not None and d.get('conf', 0.0) >= (state.min_announce_conf * 0.8) and d['depth_m'] <= max(10.0, nearby_m + 4.0):
                            nearby_present = True
                            break
                except Exception:
                    nearby_present = True  # be permissive if filter fails
                if (not realtime) and (not spoken_this_tick) and (not suppress_bearing) and nearby_present and diff > 0.25 and (now - state._last_bearing_prompt >= 8.0) and (now >= state._global_silence_until) and ((time.time() - state.last_announce_time) >= effective_cd):
                    side = 'left' if bal.get('left', 0.0) > bal.get('right', 0.0) else 'right'
                    speak_status(f"Center yourself; most obstacles {side}.")
                    with state.lock:
                        state._last_bearing_prompt = now
                        state.last_announce_time = now
                    spoken_this_tick = True
            except Exception:
                pass
            # AI-powered decision making with context awareness
            scored_detections = []
            # Collect potential summary candidates within this tick
            heartbeat_candidates = []
            summary_bucket = []

            # Local helpers for this tick
            def _update_and_smooth_depth(object_depth_history: dict, track_id: str, now: float, current_depth: float) -> float:
                try:
                    if track_id not in object_depth_history:
                        object_depth_history[track_id] = []
                    object_depth_history[track_id].append((now, float(current_depth)))
                    cutoff_time = now - 3.0
                    object_depth_history[track_id] = [(t, d) for t, d in object_depth_history[track_id] if t >= cutoff_time]
                    depths = [d for _, d in object_depth_history[track_id]]
                    n = len(depths)
                    if n == 0:
                        return float(current_depth)
                    if n >= 4:
                        med = float(np.median(depths))
                        abs_dev = np.abs(np.array(depths) - med)
                        mad = float(np.median(abs_dev)) or 1e-6
                        z = abs_dev / (1.4826 * mad)
                        w = np.clip(2.5 - z, 0.0, 1.0)
                        if np.sum(w) > 0:
                            sm = float(np.sum(np.array(depths) * w) / np.sum(w))
                        else:
                            sm = med
                        window = depths[-3:]
                        if len(window) >= 2:
                            deltas = np.diff(window)
                            if np.max(np.abs(deltas)) > 0.6:
                                sm = float(0.7 * med + 0.3 * float(current_depth))
                        return sm
                    elif n >= 2:
                        return float(np.mean(depths[-2:]))
                    else:
                        return float(current_depth)
                except Exception:
                    return float(current_depth)

            def _compute_effective_speed(detection: dict, base_speed: float) -> float:
                eff_speed = (base_speed or 0.0) if (base_speed is not None) else 0.0
                try:
                    cs = float(getattr(state, 'center_approach_mps', 0.0) or 0.0)
                    frame_shape = state.frame.shape if state.frame is not None else (CAMH, CAMW, 3)
                    in_corridor = roi_in_center_corridor(detection, frame_shape, config)
                    eff_speed = max(eff_speed, cs * (0.8 if in_corridor else 0.5))
                except Exception:
                    pass
                # Blend area-growth corroboration as a boost for approach
                try:
                    tid = detection.get('track_id')
                    if tid is not None and tid in state.track_areas:
                        arr = state.track_areas.get(tid, [])
                        now_ts = detection.get('time', time.time())
                        recent = [(t, a) for t, a in arr if (now_ts - t) <= 1.2]
                        if len(recent) >= 2:
                            a0 = float(recent[0][1])
                            a1 = float(recent[-1][1])
                            if a0 > 1.0:
                                ratio = a1 / a0
                                mode = getattr(state, 'travel_mode', 'WALKING')
                                thr = 1.12 if str(mode).upper() == 'WALKING' else 1.25
                                if ratio >= thr:
                                    boost = min(2.0, (ratio - 1.0) * (1.6 if in_corridor else 1.2))
                                    eff_speed = max(eff_speed, boost)
                except Exception:
                    pass
                return eff_speed

            def _adjust_importance_fairness(importance: float, detection: dict, now: float) -> float:
                try:
                    ann_cfg = config.get('announcements', {})
                    if bool(ann_cfg.get('fairness_bias_enabled', True)):
                        cls = str(detection.get('label', '')).lower()
                        tel = state.telemetry
                        window_s = float(ann_cfg.get('class_quota_window_s', 4.0))
                        cls_counts = {}
                        for c, dq in tel.get('per_class_ts', {}).items():
                            try:
                                cls_counts[c] = sum(1 for t in dq if (now - t) <= window_s)
                            except Exception:
                                cls_counts[c] = 0
                        current_cnt = cls_counts.get(cls, 0)
                        min_cnt = min(cls_counts.values()) if cls_counts else 0
                        if current_cnt <= min_cnt:
                            importance *= 1.08
                except Exception:
                    pass
                return importance

            def _side_fairness_boost(importance: float, detection: dict) -> float:
                try:
                    ann_cfg = config.get('announcements', {})
                    if bool(ann_cfg.get('side_round_robin_enabled', True)):
                        frame_shape = state.frame.shape if state.frame is not None else (CAMH, CAMW, 3)
                        side_rr = compute_bearing_side(detection, frame_shape, config)
                        last_side = getattr(state, '_last_side_spoken', None)
                        if last_side and side_rr and (side_rr != last_side):
                            return importance * 1.05
                except Exception:
                    pass
                return importance

            def _consider_heartbeat_candidate(detection: dict, smoothed_depth: float, eff_speed: float):
                try:
                    mode = getattr(state, 'travel_mode', 'WALKING')
                    factor = 1.0
                    if str(mode).upper() in ('DRIVING', 'CYCLING'):
                        factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
                    nearby_m = NEARBY_DISTANCE_M * factor
                    d_m = float(smoothed_depth or detection.get('depth_m') or 1e9)
                    stable = (eff_speed < 0.4)
                    track_id = detection.get('track_id')
                    try:
                        if track_id and (track_id in state.object_tracks) and (len(state.object_tracks[track_id]) >= 5):
                            depths_hist = [d for _, d, _ in state.object_tracks[track_id] if d is not None]
                            if len(depths_hist) >= 3:
                                depth_std = float(np.std(depths_hist[-3:]))
                                stable = stable and (depth_std < 0.25)
                    except Exception:
                        pass
                    if (d_m <= max(nearby_m + 1.0, 6.0)) and stable:
                        return d_m
                except Exception:
                    pass
                return None

            def _per_track_repeat_suppressed(detection: dict, speed: float, now: float, depth_for_announcement: float) -> bool:
                try:
                    tid = detection.get('track_id')
                    if tid is not None:
                        last = state.per_track_last_announce.get(tid)
                        pt_cd = float(config.get('audio', {}).get('per_track_cooldown_s', 2.5))
                        if last:
                            last_time = float(last.get('time', 0.0))
                            last_depth = float(last.get('depth', depth_for_announcement))
                            dt = now - last_time
                            depth_improved = (last_depth - depth_for_announcement) >= 0.3
                            speed_ok = (speed is not None) and (speed >= 1.0)
                            return (dt < pt_cd) and (not depth_improved) and (not speed_ok)
                except Exception:
                    pass
                return False

            def _event_edge_triggered(detection: dict, eff_speed: float, smoothed_depth: float, frame_shape: tuple, now_local: float):
                """Return (triggered: bool, event_type: str) for edge-based announcements.

                Triggers on:
                - New object appears (new track ID)
                - Distance threshold crossing (far→near→warning→critical)
                - Enter center/blocked path region
                - Red-alert classes escalate immediately (fire, gun, smoke)
                - Behavior change to approaching (stationary→approach_slow/fast)
                """
                try:
                    tid = detection.get('track_id')
                    label = str(detection.get('label','')).lower()
                    # Travel-mode-aware thresholds
                    mode = getattr(state, 'travel_mode', 'WALKING')
                    factor = 1.0
                    if str(mode).upper() in ('DRIVING', 'CYCLING'):
                        factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
                    critical_m = CRITICAL_DISTANCE_M * factor
                    warning_m = WARNING_DISTANCE_M * factor
                    nearby_m = NEARBY_DISTANCE_M * factor

                    # Compute current zone
                    d = float(smoothed_depth or detection.get('depth_m') or 1e9)
                    if d <= critical_m:
                        zone = 'critical'
                    elif d <= warning_m:
                        zone = 'warning'
                    elif d <= nearby_m:
                        zone = 'near'
                    else:
                        zone = 'far'
                    zone_rank = {'far': 0, 'near': 1, 'warning': 2, 'critical': 3}
                    last_zone = state._track_last_zone.get(tid)

                    # Center corridor membership
                    center_in = False
                    try:
                        center_in = bool(roi_in_center_corridor(detection, frame_shape, config))
                    except Exception:
                        center_in = False
                    last_center_in = state._track_last_center_in.get(tid, False)

                    # Behavior state from effective speed
                    vf_thr = VERY_FAST_THRESHOLD * (1.4 if str(mode).upper()=='DRIVING' else 1.2 if str(mode).upper()=='CYCLING' else 1.0)
                    fast_thr = FAST_SPEED_THRESHOLD * (1.4 if str(mode).upper()=='DRIVING' else 1.2 if str(mode).upper()=='CYCLING' else 1.0)
                    spd = float(eff_speed or 0.0)
                    if spd >= vf_thr:
                        behavior = 'approach_vfast'
                    elif spd >= fast_thr:
                        behavior = 'approach_fast'
                    elif spd >= 0.3:
                        behavior = 'approach_slow'
                    else:
                        behavior = 'stationary'
                    last_behavior = state._track_last_behavior.get(tid)

                    # Cooldown for edge events to avoid rapid repeats
                    edge_cd = float(config.get('announcements', {}).get('event_edge_cooldown_s', 1.2))
                    last_evt_ts = float(state._track_last_event_ts.get(tid, 0.0) or 0.0)
                    cooldown_ok = (now_local - last_evt_ts) >= edge_cd

                    # Determine event type priority
                    event_type = None
                    # Red-alert immediate escalate on first sight
                    red_alert_classes = {'fire','gun','smoke'}
                    if label in red_alert_classes:
                        # Trigger only once per track unless cooldown passes
                        if cooldown_ok:
                            event_type = 'red_alert'
                    # New object appears
                    if (event_type is None) and (tid is not None) and (tid not in state._track_first_seen):
                        event_type = 'new_object'
                    # Threshold crossing towards danger
                    if (event_type is None) and (last_zone is not None) and (zone_rank.get(zone,0) > zone_rank.get(last_zone,0)):
                        event_type = f"threshold_cross_{zone}"
                    # Enter center corridor
                    if (event_type is None) and center_in and (not last_center_in):
                        event_type = 'enter_center'
                    # Behavior change towards user
                    if (event_type is None) and last_behavior is not None:
                        # Stationary→approach or slow→fast/vfast
                        if (last_behavior == 'stationary' and behavior in ('approach_slow','approach_fast','approach_vfast')) or \
                           (last_behavior == 'approach_slow' and behavior in ('approach_fast','approach_vfast')) or \
                           (last_behavior == 'approach_fast' and behavior == 'approach_vfast'):
                            event_type = 'behavior_' + ('approach_fast' if behavior in ('approach_fast','approach_vfast') else 'approach_slow')

                    # Update state trackers
                    if tid is not None:
                        state._track_first_seen.setdefault(tid, now_local)
                        state._track_last_zone[tid] = zone
                        state._track_last_center_in[tid] = center_in
                        state._track_last_behavior[tid] = behavior
                        if event_type:
                            state._track_last_event_ts[tid] = now_local
                    # Telemetry for edges
                    try:
                        if event_type:
                            state.telemetry['events'].append({
                                'time': now_local,
                                'type': 'edge_event',
                                'edge': event_type,
                                'track_id': tid,
                                'label': label,
                                'depth_m': float(d),
                                'speed_mps': float(spd),
                                'center_in': bool(center_in),
                                'proximity_confidence': float(detection.get('proximity_confidence', 0.0))
                            })
                    except Exception:
                        pass

                    # Respect cooldown if triggered solely by red-alert; others can fire once on actual edge
                    if event_type is not None:
                        if event_type == 'red_alert':
                            return cooldown_ok, event_type
                        return True, event_type
                except Exception:
                    pass
                return False, None

            def _record_announcement_state_and_telemetry(label: str, side: str, actual_depth: float, smoothed_depth: float, speed: float, importance: float, message: str, variant_type: str, detection: dict, now: float):
                try:
                    with state.lock:
                        if not realtime:
                            state.last_announce_time = now
                        state.last_announcement_time = now
                        state._last_side_spoken = side
                        if not hasattr(state, '_recent_label_side'):
                            state._recent_label_side = {}
                        cls = str(label).lower()
                        state._recent_label_side[(cls, side)] = now
                        tid = detection.get('track_id')
                        if tid is not None:
                            state.per_track_last_announce[tid] = {
                                'time': now,
                                'depth': float(smoothed_depth)
                            }
                        if variant_type in ('urgent_critical','warning_critical'):
                            state._global_silence_until = now + float(
                                config.get('audio', {}).get(
                                    'hint_block_after_critical_s',
                                    config.get('audio', {}).get('global_silence_after_critical_s', 2.5)
                                )
                            )
                            try:
                                state._avoid_frame_skip_until = max(state._avoid_frame_skip_until, now + float(config.get('hazard', {}).get('boost_duration_s', 1.6)))
                            except Exception:
                                state._avoid_frame_skip_until = now + 1.6
                        if smoothed_depth <= (WARNING_DISTANCE_M * (1.4 if str(getattr(state, 'travel_mode','WALKING')).upper()=='DRIVING' else 1.25 if str(getattr(state, 'travel_mode','WALKING')).upper()=='CYCLING' else 1.0)):
                            state._last_warning_time = now
                        try:
                            vkey = (variant_type, label.lower(), side)
                            state._last_variant_spoken[vkey] = now
                        except Exception:
                            pass
                        try:
                            if variant_type in ('urgent_critical','urgent_approach'):
                                state.telemetry['preempt_requests_ts'].append(now)
                        except Exception:
                            pass
                        try:
                            tel = state.telemetry
                            if cls not in tel['per_class_ts']:
                                tel['per_class_ts'][cls] = deque(maxlen=5000)
                            tel['per_class_ts'][cls].append(now)
                            if cls not in tel['per_class_history']:
                                tel['per_class_history'][cls] = deque(maxlen=60)
                            current_minute = int(now // 60)
                            if not tel['per_class_history'][cls] or tel['per_class_history'][cls][-1]['minute'] != current_minute:
                                tel['per_class_history'][cls].append({'minute': current_minute, 'count': 1})
                            else:
                                tel['per_class_history'][cls][-1]['count'] += 1
                            tel['events'].append({
                                'time': now,
                                'type': variant_type,
                                'label': cls,
                                'side': side,
                                'depth_m': float(actual_depth),
                                'smoothed_m': float(smoothed_depth),
                                'speed_mps': float(speed or 0.0),
                                'eff_speed_mps': float(speed or 0.0),
                                'proximity_confidence': float(detection.get('proximity_confidence', 0.0)),
                                'area_frac': float(detection.get('area_frac', 0.0)),
                                'approach_speed_mps': float(detection.get('approach_speed_mps', 0.0)),
                                'importance': float(importance),
                                'message': message
                            })
                            if variant_type == 'moderate_approach':
                                tel['moderate_approach_ts'].append(now)
                                if side == 'left':
                                    tel['moderate_left_ts'].append(now)
                                elif side == 'right':
                                    tel['moderate_right_ts'].append(now)
                            elif variant_type == 'slow_approach':
                                tel['slow_approach_ts'].append(now)
                                if side == 'left':
                                    tel['slow_left_ts'].append(now)
                                elif side == 'right':
                                    tel['slow_right_ts'].append(now)
                        except Exception:
                            pass
                except Exception:
                    pass

            def _emit_summary_if_needed(summary_bucket_local: list, now_local: float, cooldown_ok_local: bool) -> bool:
                try:
                    ann_cfg = config.get('announcements', {})
                    audio_cfg = config.get('audio', {})
                    summary_min_items = int(audio_cfg.get('summary_min_items', 2))
                    summary_group_window_s = float(audio_cfg.get('summary_group_window_s', 1.5))
                    summary_interval_s = float(ann_cfg.get('summary_interval_s', audio_cfg.get('summary_interval_s', 9.0)))
                except Exception:
                    summary_min_items = 2
                    summary_group_window_s = 1.5
                    summary_interval_s = 9.0
                try:
                    if cooldown_ok_local and (now_local >= state._global_silence_until):
                        if len(summary_bucket_local) >= summary_min_items and (now_local - state._last_summary >= summary_interval_s):
                            counts = {}
                            sides = {'left': 0, 'right': 0, 'ahead': 0}
                            closest = 1e9
                            for item in summary_bucket_local:
                                lbl = item.get('label', 'object')
                                counts[lbl] = counts.get(lbl, 0) + 1
                                sd = item.get('side', 'ahead')
                                if sd in sides:
                                    sides[sd] += 1
                                dval = float(item.get('depth_m', 1e9) or 1e9)
                                closest = min(closest, dval)
                            top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:3]
                            desc_parts = []
                            for name, cnt in top:
                                plural = '' if cnt == 1 else 's'
                                desc_parts.append(f"{cnt} {name}{plural}")
                            desc = ', '.join(desc_parts)
                            ordered_sides = sorted(sides.items(), key=lambda x: (-x[1], x[0]))
                            side_phrase = 'ahead'
                            if ordered_sides[0][1] > 0:
                                top1 = ordered_sides[0][0]
                                if len(ordered_sides) > 1 and ordered_sides[1][1] > 0:
                                    top2 = ordered_sides[1][0]
                                    if top1 != top2:
                                        if ('ahead' in (top1, top2)):
                                            other = top1 if top2 == 'ahead' else top2
                                            side_phrase = f"{other} and ahead"
                                        else:
                                            side_phrase = f"{top1} and {top2}"
                                    else:
                                        side_phrase = top1
                                else:
                                    side_phrase = top1
                            if closest < 1e8:
                                try:
                                    if closest <= 1.0:
                                        q = 'very close'
                                    elif closest <= 2.5:
                                        q = 'close'
                                    else:
                                        q = 'near'
                                except Exception:
                                    q = 'near'
                                message = f"Nearby: {desc} {side_phrase}; closest {q}."
                            else:
                                message = f"Nearby: {desc} {side_phrase}."
                            speak_status(message)
                            with state.lock:
                                state._last_summary = now_local
                                state.last_announce_time = now_local
                                try:
                                    state.telemetry['events'].append({
                                        'time': now_local,
                                        'type': 'summary',
                                        'importance': 'medium',
                                        'message': message
                                    })
                                except Exception:
                                    pass
                            return True
                except Exception:
                    pass
                return False

            def _emit_heartbeat_if_needed(spoken_flag: bool, heartbeat_cands: list, now_local: float):
                try:
                    if (not spoken_flag) and heartbeat_cands and (now_local >= state._global_silence_until):
                        emit_hazard_heartbeat(heartbeat_cands, config)
                except Exception:
                    pass

            for detection in dets:
                if detection['depth_m'] is None or detection['conf'] < state.min_announce_conf:
                    continue
                
                track_id = detection.get('track_id')
                speed = 0.0
                smoothed_depth = detection['depth_m']
                
                # Apply object-specific depth smoothing
                if track_id:
                    smoothed_depth = _update_and_smooth_depth(object_depth_history, track_id, now, detection['depth_m'])
                
                # Update the detection with smoothed depth
                detection['smoothed_depth'] = smoothed_depth
                
                # Calculate speed with smoothed depth data
                if track_id and track_id in state.object_tracks:
                    track_data = state.object_tracks[track_id]
                    if len(track_data) >= MIN_SAMPLES_FOR_SPEED:
                        speed = calculate_object_speed(track_data, now)
                # Fuse center-derived looming speed into an effective speed estimate for scoring
                eff_speed = _compute_effective_speed(detection, speed)

                # Event-edge gating: only consider detections that triggered an edge
                frame_shape_local = state.frame.shape if state.frame is not None else (CAMH, CAMW, 3)
                edge_ok, edge_type = _event_edge_triggered(detection, eff_speed, smoothed_depth, frame_shape_local, now)
                
                # AI decision: should we announce this?
                if edge_ok and should_announce(detection, eff_speed, state.environment_context):
                    # Use smoothed depth for importance calculation
                    original_depth = detection['depth_m']
                    detection['depth_m'] = smoothed_depth  # Temporarily use smoothed depth
                    importance = calculate_importance_score(detection, eff_speed, state.environment_context)
                    detection['depth_m'] = original_depth  # Restore original depth

                    # Fairness tie-breaker: small boost to underrepresented classes
                    importance = _adjust_importance_fairness(importance, detection, now)

                    # Side fairness: prefer alternating sides when differences are small
                    importance = _side_fairness_boost(importance, detection)

                    # Attach edge type to detection for event-aware phrasing
                    detection['event_type'] = edge_type
                    scored_detections.append((importance, detection, eff_speed, smoothed_depth, edge_type))
                else:
                    cand = _consider_heartbeat_candidate(detection, smoothed_depth, eff_speed)
                    if cand is not None:
                        heartbeat_candidates.append((cand, detection))
            
            # Sort by importance (highest first)
            scored_detections.sort(key=lambda x: x[0], reverse=True)
            
            # Announce detections; in realtime mode allow multiple high-severity messages
            can_speak = ((time.time() - state.last_announce_time) >= state.announce_cooldown_s) or realtime
            if scored_detections and can_speak:
                try:
                    max_msgs_cfg = int(config.get('announcements', {}).get('realtime_bundle_max', 2))
                except Exception:
                    max_msgs_cfg = 2
                max_msgs = max_msgs_cfg if realtime else 1
                announced = 0
                for importance, detection, speed, smoothed_depth, edge_type in scored_detections[:max_msgs]:
                    label = detection['label']
                    actual_depth = detection['depth_m']
                    
                    # Use smoothed depth for announcement decision but actual depth for display
                    # This prevents fluctuations while keeping accurate measurements
                    depth_for_announcement = smoothed_depth
                    # Instant prediction: estimate near-term depth for moving objects when depth updates lag
                    try:
                        audio_cfg = config.get('audio', {})
                        enable_pred = bool(audio_cfg.get('ttc_instant_prediction', True))
                        min_spd = float(audio_cfg.get('ttc_prediction_min_speed_mps', 0.6))
                        min_dt = float(audio_cfg.get('ttc_prediction_min_dt_s', 0.25))
                        deadband = float(audio_cfg.get('ttc_prediction_deadband_m', 0.05))
                        depth_for_ttc = depth_for_announcement
                        depth_gate = depth_for_announcement
                        tid = detection.get('track_id')
                        if enable_pred and (speed is not None) and (speed >= min_spd) and tid in object_depth_history:
                            hist = object_depth_history.get(tid, [])
                            # Find the last time the depth changed beyond deadband
                            last_change_ts = None
                            if len(hist) >= 2:
                                for i in range(len(hist) - 1, 0, -1):
                                    if abs(float(hist[i][1]) - float(hist[i-1][1])) > deadband:
                                        last_change_ts = float(hist[i][0])
                                        break
                                if last_change_ts is None:
                                    last_change_ts = float(hist[-1][0])
                            elif len(hist) == 1:
                                last_change_ts = float(hist[0][0])
                            if last_change_ts is not None:
                                dt_since = float(now - last_change_ts)
                                if dt_since >= min_dt:
                                    # Predict near-term depth by extrapolating with effective speed
                                    pred = float(depth_for_announcement) - float(speed) * dt_since
                                    pred = max(0.2, pred)
                                    # Use the lower of measured vs predicted to gate TTC and approach variants
                                    depth_for_ttc = min(depth_for_announcement, pred)
                                    depth_gate = depth_for_ttc
                        else:
                            depth_for_ttc = depth_for_announcement
                            depth_gate = depth_for_announcement
                    except Exception:
                        depth_for_ttc = depth_for_announcement
                        depth_gate = depth_for_announcement
                    
                    # Bearing cue (center-aware) with fine-grained side for nuanced guidance
                    frame_shape = state.frame.shape if state.frame is not None else (CAMH, CAMW, 3)
                    side = compute_bearing_side(detection, frame_shape, config)
                    fine_side = compute_bearing_fine_side(detection, frame_shape, config)
                    # Normalize side for telemetry/earcons; phrase side for speech
                    side_norm = 'left' if fine_side in ('left','slight_left') else (
                        'right' if fine_side in ('right','slight_right') else 'ahead'
                    )
                    msg_side_phrase = 'slightly left' if fine_side == 'slight_left' else (
                        'slightly right' if fine_side == 'slight_right' else (
                            'center' if fine_side == 'center' else side
                        )
                    )

                    # Per-track cooldown gating to suppress repeats when not closing
                    if _per_track_repeat_suppressed(detection, speed, now, depth_for_announcement):
                        continue
                    
                    # Generate contextually appropriate announcement (event-edge first)
                    # Travel-mode-aware thresholds
                    mode = getattr(state, 'travel_mode', 'WALKING')
                    factor = 1.0
                    if str(mode).upper() in ('DRIVING', 'CYCLING'):
                        factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
                    critical_m = CRITICAL_DISTANCE_M * factor
                    warning_m = WARNING_DISTANCE_M * factor
                    nearby_m = NEARBY_DISTANCE_M * factor
                    # Speed threshold scaling by travel mode
                    speed_thr_factor = 1.0
                    if str(mode).upper() in ('DRIVING', 'CYCLING'):
                        speed_thr_factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.2
                    vf_thr = VERY_FAST_THRESHOLD * speed_thr_factor
                    fast_thr = FAST_SPEED_THRESHOLD * speed_thr_factor
                    # TTC-based gating thresholds
                    audio_cfg = config.get('audio', {})
                    ann_cfg = config.get('announcements', {})
                    smart_ttc = bool(ann_cfg.get('smart_ttc', True))
                    try:
                        ttc_urgent_s = float(audio_cfg.get('ttc_urgent_s', 1.5))
                    except Exception:
                        ttc_urgent_s = 1.5
                    try:
                        ttc_warning_s = float(audio_cfg.get('ttc_warning_s', 3.0))
                    except Exception:
                        ttc_warning_s = 3.0
                    try:
                        ttc_caution_s = float(audio_cfg.get('ttc_caution_s', 6.0))
                    except Exception:
                        ttc_caution_s = 6.0
                    # Compute TTC (time-to-collision) from smoothed depth and effective speed
                    if (speed is not None) and (speed > 1e-3):
                        ttc = float(depth_for_ttc) / float(max(speed, 1e-3))
                    else:
                        ttc = float('inf')

                    # Use effective speed (already blended earlier) for variant gating
                    # Here, 'speed' represents effective speed from scored_detections

                    variant_type = 'none'
                    message = None
                    # Defer micro-guidance suggestion until after variant selection
                    suggestion = None

                    # Prefer event-edge phrasing to avoid continuous-state chatter
                    try:
                        if edge_type:
                            et = str(edge_type)
                            if et == 'new_object':
                                message = f"New {label} {msg_side_phrase}."
                                variant_type = 'new_object'
                            elif et.startswith('threshold_cross_'):
                                z = et.split('threshold_cross_')[-1]
                                if z == 'critical':
                                    message = f"URGENT! {label} very close {msg_side_phrase}!"
                                    variant_type = 'urgent_critical'
                                elif z == 'warning':
                                    message = f"Now caution: {label} {msg_side_phrase}."
                                    variant_type = 'caution'
                                else:  # near
                                    message = f"Now near: {label} {msg_side_phrase}."
                                    variant_type = 'nearby'
                            elif et == 'enter_center':
                                message = f"Path blocked: {label} center."
                                variant_type = 'enter_center'
                                try:
                                    play_spatial_earcon('warning', side=side_norm, depth_m=float(depth_for_announcement or 0.0))
                                except Exception:
                                    play_earcon('warning', side=side_norm)
                            elif et == 'behavior_approach_fast':
                                message = f"Approaching fast: {label} {msg_side_phrase}."
                                variant_type = 'warning_approach'
                                try:
                                    play_spatial_earcon('warning', side=side_norm, depth_m=float(depth_for_announcement or 0.0))
                                except Exception:
                                    play_earcon('warning', side=side_norm)
                            elif et == 'behavior_approach_slow':
                                message = f"Approaching: {label} {msg_side_phrase}."
                                variant_type = 'moderate_approach'
                            elif et == 'red_alert':
                                message = f"Red alert! {label} {msg_side_phrase}!"
                                variant_type = 'urgent_critical'
                                try:
                                    play_spatial_earcon('warning', side=side_norm, depth_m=float(depth_for_announcement or 0.0))
                                except Exception:
                                    play_earcon('warning', side=side_norm)
                    except Exception:
                        pass
                    if depth_for_announcement <= critical_m:
                        if (speed is not None) and (speed >= vf_thr):
                            message = f"URGENT! Fast {label} very close {msg_side_phrase}!"
                            variant_type = 'urgent_critical'
                        else:
                            message = f"Warning! {label} very close {msg_side_phrase}."
                            variant_type = 'warning_critical'
                        # voice-only for criticals; skip earcon
                    elif (
                        ((speed is not None) and (speed >= vf_thr)) or (smart_ttc and (ttc <= ttc_urgent_s))
                    ) and (depth_gate <= max(8.0, nearby_m + 2.0)):
                        message = f"URGENT! Fast {label} approaching {msg_side_phrase}!"
                        variant_type = 'urgent_approach'
                        try:
                            try:
                                ang_deg, pan_val = compute_angle_pan_from_detection(detection, frame_shape, config)
                                _set_spatial_pan_hint(ang_deg, pan_val)
                            except Exception:
                                pass
                            play_spatial_earcon('warning', side=side_norm, depth_m=float(depth_for_announcement or 0.0))
                        except Exception:
                            play_earcon('warning', side=side_norm)
                        # Activate hazard boost to prioritize depth refresh and YOLO speed
                        try:
                            bd = float(config.get('hazard', {}).get('boost_duration_s', 1.6))
                            with state.lock:
                                state.hazard_boost_active_until = max(getattr(state, 'hazard_boost_active_until', 0.0), now + bd)
                                try:
                                    state._yolo_img_override = int(config.get('hazard', {}).get('yolo_imgsz_boost'))
                                except Exception:
                                    state._yolo_img_override = None
                        except Exception:
                            pass
                    elif (
                        ((speed is not None) and (speed >= fast_thr)) or (smart_ttc and (ttc <= ttc_warning_s))
                    ) and (depth_gate <= max(6.0, warning_m + 1.0)):
                        message = f"Warning! {label} approaching fast {msg_side_phrase}."
                        variant_type = 'warning_approach'
                        try:
                            try:
                                ang_deg, pan_val = compute_angle_pan_from_detection(detection, frame_shape, config)
                                _set_spatial_pan_hint(ang_deg, pan_val)
                            except Exception:
                                pass
                            play_spatial_earcon('warning', side=side_norm, depth_m=float(depth_for_announcement or 0.0))
                        except Exception:
                            play_earcon('warning', side=side_norm)
                        try:
                            bd = float(config.get('hazard', {}).get('boost_duration_s', 1.6))
                            with state.lock:
                                state.hazard_boost_active_until = max(getattr(state, 'hazard_boost_active_until', 0.0), now + bd)
                                try:
                                    state._yolo_img_override = int(config.get('hazard', {}).get('yolo_imgsz_boost'))
                                except Exception:
                                    state._yolo_img_override = None
                        except Exception:
                            pass
                    # Intent countdown: anticipatory prompt when TTC is within window (no other message yet)
                    if not message:
                        try:
                            intent_enabled = bool(ann_cfg.get('intent_enabled', True))
                            if intent_enabled and (speed is not None) and (speed >= 0.6):
                                min_s = float(audio_cfg.get('intent_countdown_min_s', 1.2))
                                max_s = float(audio_cfg.get('intent_countdown_max_s', 3.5))
                                # Require within nearby range and not too far
                                if (ttc < float('inf')) and (min_s <= ttc <= max_s) and (depth_gate <= max(6.5, nearby_m + 1.5)):
                                    # Per-track intent cooldown
                                    try:
                                        if not hasattr(state, '_last_intent_ts'):
                                            state._last_intent_ts = {}
                                        tid = detection.get('track_id')
                                        last_it = float(state._last_intent_ts.get(tid, 0.0) or 0.0)
                                        if (time.time() - last_it) >= 2.5:
                                            secs = round(ttc, 1)
                                            message = f"Incoming {label} {msg_side_phrase} in about {secs} seconds."
                                            variant_type = 'intent_countdown'
                                            try:
                                                play_spatial_earcon('nearby', side=side_norm, depth_m=float(depth_for_announcement or 0.0))
                                            except Exception:
                                                play_earcon('nearby', side=side_norm)
                                            with state.lock:
                                                state._last_intent_ts[tid] = time.time()
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                    elif (speed is not None) and (speed >= 1.0) and (depth_for_announcement <= max(5.0, nearby_m + 1.0)) and ((now - state._last_closing_fast) >= 6.0):
                        message = f"Closing fast! {label} {msg_side_phrase}."
                        variant_type = 'closing_fast'
                        try:
                            play_spatial_earcon('warning', side=side_norm, depth_m=float(depth_for_announcement or 0.0))
                        except Exception:
                            play_earcon('warning', side=side_norm)
                        with state.lock:
                            state._last_closing_fast = now
                    elif (speed is not None) and (speed >= 0.6) and (depth_gate <= max(5.0, nearby_m + 1.0)) and (not (smart_ttc and (ttc >= ttc_caution_s))):
                        message = f"Caution. {label} approaching {msg_side_phrase}."
                        variant_type = 'moderate_approach'
                        try:
                            play_spatial_earcon('nearby', side=side_norm, depth_m=float(depth_for_announcement or 0.0))
                        except Exception:
                            play_earcon('nearby', side=side_norm)
                    elif (speed is not None) and (speed >= 0.3) and (depth_gate <= max(5.5, nearby_m + 1.5)) and (not (smart_ttc and (ttc >= ttc_caution_s))):
                        message = f"Heads up. {label} slowly approaching {msg_side_phrase}."
                        variant_type = 'slow_approach'
                        try:
                            play_spatial_earcon('nearby', side=side_norm, depth_m=float(depth_for_announcement or 0.0))
                        except Exception:
                            play_earcon('nearby', side=side_norm)
                    elif depth_for_announcement <= warning_m:
                        message = f"Caution! {label} {msg_side_phrase}."
                        variant_type = 'caution'
                        try:
                            play_spatial_earcon('nearby', side=side_norm, depth_m=float(depth_for_announcement or 0.0))
                        except Exception:
                            play_earcon('nearby', side=side_norm)
                    elif (depth_for_announcement <= nearby_m) and (importance > 0.8):
                        message = f"{label} nearby {msg_side_phrase}."
                        variant_type = 'nearby'
                    # Append qualitative proximity for natural realism on non-critical variants
                    try:
                        if message and variant_type in ('moderate_approach','slow_approach','warning_approach','caution','nearby'):
                            d_m = float(depth_for_announcement or 0.0)
                            q = None
                            try:
                                if d_m <= 1.0:
                                    q = 'very close'
                                elif d_m <= 2.5:
                                    q = 'close'
                                elif d_m <= 7.0:
                                    q = 'near'
                            except Exception:
                                q = None
                            if q:
                                if message.endswith('.'):
                                    message = message[:-1]
                                message = f"{message}, {q}."
                    except Exception:
                        pass
                    # Micro-guidance suggestion based on corridor and severity
                    try:
                        ann_cfg = config.get('announcements', {})
                        if bool(ann_cfg.get('micro_guidance_enabled', True)):
                            corr = state.environment_context.get('safe_corridor', {'side': None, 'width': 0.0})
                            corr_side = corr.get('side')
                            corr_width = float(corr.get('width', 0.0) or 0.0)
                            center_blocked = (fine_side == 'center')
                            if message and center_blocked and corr_side in ('left','right') and (corr_width >= 0.5):
                                # Escalate phrasing based on urgency
                                if variant_type in ('urgent_critical','urgent_approach') or (ttc <= ttc_urgent_s) or (depth_for_announcement <= critical_m):
                                    suggestion = f"Move {corr_side} now"
                                elif variant_type in ('warning_approach','moderate_approach') or (ttc <= ttc_warning_s):
                                    suggestion = f"Prepare to move {corr_side}"
                                else:
                                    suggestion = f"Go slightly {corr_side}"
                    except Exception:
                        suggestion = None
                    # Append suggestion if applicable
                    if message and suggestion:
                        if message.endswith('.'):
                            message = message[:-1]
                        message = f"{message}; {suggestion}."

                    # Compact phrasing under noise/load
                    try:
                        ann_cfg = config.get('announcements', {})
                        audio_cfg = config.get('audio', {})
                        noise_lvl = float(state.environment_context.get('noise_level', 0.0) or 0.0)
                        compact_en = bool(ann_cfg.get('compact_style_enabled', True))
                        noise_thr = float(ann_cfg.get('compact_trigger_noise_level', 0.7))
                        under_noise = compact_en and (noise_lvl >= noise_thr)
                        # Under system load also compact if enabled
                        last_fps = float(getattr(state, 'last_fps', 0.0) or 0.0)
                        fps_thresh = float(audio_cfg.get('smart_fps_threshold', 12.0))
                        lat_hist = list(getattr(state, 'performance_stats', {}).get('tts_queue_latencies', []))
                        recent_lat = float(np.mean(lat_hist[-10:])) if lat_hist[-10:] else 0.0
                        lat_thresh = float(audio_cfg.get('smart_queue_latency_s', 0.7))
                        under_load = bool(ann_cfg.get('compact_trigger_under_load', True)) and (((last_fps > 0 and last_fps < fps_thresh) or (recent_lat > lat_thresh)))
                        # Auto style adjustment under noise/load if personalization enabled
                        try:
                            pers_en = bool(ann_cfg.get('ai_personalization_enabled', True))
                            if pers_en and (under_noise or under_load):
                                with state.lock:
                                    state.ai_style_profile = 'concise'
                                    state._style_auto_until = time.time() + 30.0
                                try:
                                    state.telemetry['events'].append({'time': time.time(), 'type': 'style_auto_adjust', 'reason': 'noise_or_load', 'style': 'concise'})
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        if message and (under_noise or under_load):
                            # Trim verbosity
                            msg = message
                            msg = msg.replace('Warning!', 'Warning.')
                            msg = msg.replace('URGENT!', 'Urgent!')
                            msg = msg.replace('approaching fast', 'fast')
                            msg = msg.replace('slowly approaching', 'slow')
                            msg = msg.replace('approaching', 'near')
                            # Strip any residual numeric meter phrases for compactness
                            if ' around ' in msg and ' meters' in msg:
                                try:
                                    idx = msg.find(' around ')
                                    if idx != -1:
                                        head = msg[:idx].rstrip(' ,')
                                        msg = head if head.endswith('.') else f"{head}."
                                except Exception:
                                    pass
                            message = msg
                            try:
                                speak_async(msg)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Adaptive phrase variety based on style profile
                    try:
                        ann_cfg = config.get('announcements', {})
                        variety_en = bool(ann_cfg.get('ai_phrase_variety_enabled', True))
                        style = str(getattr(state, 'ai_style_profile', ann_cfg.get('style_profile', 'concise')) or 'concise').lower()
                        try:
                            if hasattr(state, '_style_auto_until') and (time.time() <= float(getattr(state, '_style_auto_until', 0.0))):
                                style = 'concise'
                        except Exception:
                            pass
                        # Prefer concise for DRIVING unless explicitly set descriptive
                        try:
                            if str(getattr(state, 'travel_mode', 'WALKING')) == 'DRIVING' and style != 'descriptive':
                                style = 'concise'
                        except Exception:
                            pass
                        if message and variety_en:
                            # Templates per variant
                            templates = {
                                'urgent_critical': {
                                    'concise': ["Urgent! {label} {side} very close!", "Danger! {label} {side}!"],
                                    'descriptive': ["URGENT! {label} {side}, very close!", "Immediate hazard: {label} {side}!"],
                                },
                                'warning_critical': {
                                    'concise': ["Warning. {label} {side} very close.", "Caution! {label} {side}, close."],
                                    'descriptive': ["Warning! {label} {side}, very close.", "Watch out: {label} {side}, very close."],
                                },
                                'urgent_approach': {
                                    'concise': ["Urgent! {label} fast {side}!", "Fast {label} {side}!"],
                                    'descriptive': ["URGENT! Fast {label} {side} approaching!", "Rapid approach: {label} {side}!"],
                                },
                                'moderate_approach': {
                                    'concise': ["Caution. {label} {side} approaching.", "{label} {side}, closing."],
                                    'descriptive': ["Caution. {label} {side} approaching.", "{label} {side}, getting closer."],
                                },
                                'slow_approach': {
                                    'concise': ["Heads up. {label} slow {side}.", "Slow {label} {side}."],
                                    'descriptive': ["Heads up. {label} slowly approaching {side}.", "Slow approach: {label} {side}."],
                                },
                                'caution': {
                                    'concise': ["Caution! {label} {side}.", "Careful: {label} {side}."],
                                    'descriptive': ["Caution! {label} {side}.", "Watch out: {label} {side}."],
                                },
                                'nearby': {
                                    'concise': ["{label} {side} nearby.", "Nearby: {label} {side}."],
                                    'descriptive': ["{label} nearby {side}.", "Close by: {label} {side}."],
                                },
                                'closing_fast': {
                                    'concise': ["Closing fast! {label} {side}.", "Fast closing: {label} {side}."],
                                    'descriptive': ["Closing fast! {label} {side}.", "Rapid closing: {label} {side}."],
                                },
                                'intent_countdown': {
                                    'concise': ["Incoming {label} {side} in ~{dist}s.", "{label} {side} in ~{dist}s."],
                                    'descriptive': ["Incoming {label} {side} in about {dist} seconds.", "Approach in {dist} seconds: {label} {side}."],
                                },
                            }
                            # Person-specific phrasing overrides for approach-related variants
                            try:
                                ann_cfg_local = config.get('announcements', {})
                                person_phrases_en = bool(ann_cfg_local.get('person_phrases_enabled', True))
                            except Exception:
                                person_phrases_en = True
                            if person_phrases_en and label.lower() == 'person':
                                person_templates = {
                                    'urgent_critical': {
                                        'concise': ["Urgent! Person {side} very close!", "Danger! Person {side}!"],
                                        'descriptive': ["URGENT! Person {side}, very close!", "Immediate hazard: Person {side}!"],
                                    },
                                    'warning_critical': {
                                        'concise': ["Warning. Person {side} very close.", "Caution! Person {side}, close."],
                                        'descriptive': ["Warning! Person {side}, very close.", "Watch out: Person {side}, very close."],
                                    },
                                    'urgent_approach': {
                                        'concise': ["Urgent! Person fast {side} approaching!", "Fast Person {side}!"],
                                        'descriptive': ["URGENT! Fast Person {side} approaching!", "Rapid approach: Person {side}!"],
                                    },
                                    'moderate_approach': {
                                        'concise': ["Caution. Person {side} approaching.", "Person {side}, closing."],
                                        'descriptive': ["Caution. Person {side} approaching.", "Person {side}, getting closer."],
                                    },
                                    'slow_approach': {
                                        'concise': ["Heads up. Person slowly approaching {side}.", "Slow Person {side}."],
                                        'descriptive': ["Heads up. Person slowly approaching {side}.", "Slow approach: Person {side}."],
                                    },
                                    'caution': {
                                        'concise': ["Caution! Person {side}.", "Careful: Person {side}."],
                                        'descriptive': ["Caution! Person {side}.", "Watch out: Person {side}."],
                                    },
                                    'nearby': {
                                        'concise': ["Person {side} nearby.", "Nearby: Person {side}."],
                                        'descriptive': ["Person nearby {side}.", "Close by: Person {side}."],
                                    },
                                    'closing_fast': {
                                        'concise': ["Closing fast! Person {side}.", "Fast closing: Person {side}."],
                                        'descriptive': ["Closing fast! Person {side}.", "Rapid closing: Person {side}."],
                                    },
                                    'intent_countdown': {
                                        'concise': ["Incoming person {side} in ~{dist}s.", "Person {side} in ~{dist}s."],
                                        'descriptive': ["Incoming person {side} in about {dist} seconds.", "Approach in {dist} seconds: person {side}."],
                                    },
                                }
                                # Overlay person-specific templates when available
                                pt_bank = person_templates.get(variant_type, {}).get(style, [])
                                if pt_bank:
                                    templates[variant_type][style] = pt_bank
                            # Rotate through templates to reduce repetition
                            try:
                                if not hasattr(state, '_phrase_rotation'):
                                    state._phrase_rotation = {}
                                key = (variant_type, label.lower(), side_norm, style)
                                bank = templates.get(variant_type, {}).get(style, [])
                                if bank:
                                    idx = (state._phrase_rotation.get(key, -1) + 1) % len(bank)
                                    state._phrase_rotation[key] = idx
                                    # Prepare dist value in meters or seconds for intent
                                    dist_val = ''
                                    try:
                                        if variant_type == 'intent_countdown':
                                            dist_val = f"{round(ttc,1)}"
                                        else:
                                            dm = float(depth_for_announcement or 0.0)
                                            dist_val = f"{round(dm,1)}"
                                    except Exception:
                                        dist_val = ''
                                    message = bank[idx].format(label=label, side=msg_side_phrase, dist=dist_val)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    if not message:
                        # If nothing selected but object is within nearby range, collect for summary
                        try:
                            if depth_for_announcement <= max(nearby_m + 2.0, 6.0):
                                summary_bucket.append({
                                    'label': label.lower(),
                                    'side': side_norm,
                                    'depth_m': float(depth_for_announcement)
                                })
                        except Exception:
                            pass
                        continue  # Not important enough to announce

                    # Variant-level repeat suppression
                    try:
                        vkey = (variant_type, label.lower(), side)
                        vwin = float(config.get('audio', {}).get('message_repeat_window_s', 4.0))
                        # Strengthen suppression for non-hazard informational variants
                        non_hazard_variants = ('nearby', 'slow_approach', 'moderate_approach')
                        if variant_type in non_hazard_variants:
                            vwin = max(vwin, float(config.get('audio', {}).get('info_repeat_window_s', 12.0)))
                        last_vt = state._last_variant_spoken.get(vkey, 0.0)
                        if now - last_vt < vwin:
                            try:
                                with state.lock:
                                    tel = state.telemetry
                                    if variant_type in ('moderate_approach', 'slow_approach'):
                                        tel['approach_repeat_suppressed_ts'].append(now)
                            except Exception:
                                pass
                            continue
                    except Exception:
                        pass

                    # Announce the message
                    # Per-class quota gating: limit repeats within a short window
                    try:
                        ann_cfg = config.get('announcements', {})
                        window_s = float(ann_cfg.get('class_quota_window_s', 4.0))
                        default_max = int(ann_cfg.get('class_quota_default_max', 2))
                        overrides = ann_cfg.get('class_quota_overrides', {}) or {}
                        cls = label.lower()
                        allowed = int(overrides.get(cls, default_max))
                        tel = state.telemetry
                        if cls not in tel['per_class_ts']:
                            tel['per_class_ts'][cls] = deque(maxlen=5000)
                        recent = [t for t in tel['per_class_ts'][cls] if (now - t) <= window_s]
                        if len(recent) >= allowed:
                            # Defer to summary bucket rather than speaking immediately
                            try:
                                summary_bucket.append({'label': cls, 'side': side_norm, 'depth_m': float(depth_for_announcement)})
                            except Exception:
                                pass
                            continue
                    except Exception:
                        pass

                    # Realtime bundling: combine two approach hazards into one concise prompt
                    try:
                        ann_cfg = config.get('announcements', {})
                        bundle_en = bool(ann_cfg.get('bundle_combine_enabled', True))
                        if realtime and bundle_en and announced == 0 and len(scored_detections) >= 2:
                            # Peek next candidate
                            _, det2, spd2, smd2 = scored_detections[1]
                            # Quick side and gating
                            frame_shape = state.frame.shape if state.frame is not None else (CAMH, CAMW, 3)
                            side2 = compute_bearing_side(det2, frame_shape, config)
                            fine_side2 = compute_bearing_fine_side(det2, frame_shape, config)
                            side_norm2 = 'left' if fine_side2 in ('left','slight_left') else ('right' if fine_side2 in ('right','slight_right') else 'ahead')
                            # Second TTC
                            ttc2 = float('inf')
                            try:
                                if (spd2 is not None) and (spd2 > 1e-3):
                                    ttc2 = float((min(smd2, det2.get('depth_m', smd2))) / float(max(spd2, 1e-3)))
                            except Exception:
                                ttc2 = float('inf')
                            # Determine if both are approach variants
                            is_approach1 = variant_type in ('urgent_approach','warning_approach','moderate_approach','slow_approach')
                            is_approach2 = ((spd2 is not None) and (spd2 >= 0.3)) or (ttc2 <= ttc_warning_s)
                            # Quota gating for second
                            ok_quota = True
                            try:
                                ann_cfg2 = config.get('announcements', {})
                                window_s2 = float(ann_cfg2.get('class_quota_window_s', 4.0))
                                default_max2 = int(ann_cfg2.get('class_quota_default_max', 2))
                                overrides2 = ann_cfg2.get('class_quota_overrides', {}) or {}
                                cls2 = str(det2.get('label', '')).lower()
                                allowed2 = int(overrides2.get(cls2, default_max2))
                                tel2 = state.telemetry
                                if cls2 not in tel2['per_class_ts']:
                                    tel2['per_class_ts'][cls2] = deque(maxlen=5000)
                                recent2 = [t for t in tel2['per_class_ts'][cls2] if (now - t) <= window_s2]
                                if len(recent2) >= allowed2:
                                    ok_quota = False
                            except Exception:
                                ok_quota = True
                            if is_approach1 and is_approach2 and ok_quota and (side_norm2 != side_norm):
                                # Build combined concise prompt
                                lbl1 = str(label).lower()
                                lbl2 = str(det2.get('label', '')).lower()
                                combined = f"Two hazards: {lbl1} {side_norm}, {lbl2} {side_norm2}."
                                speak_status(combined)
                                with state.lock:
                                    if not realtime:
                                        state.last_announce_time = now
                                    state.last_announcement_time = now
                                    # Dedup marks for both sides
                                    try:
                                        if not hasattr(state, '_recent_label_side'):
                                            state._recent_label_side = {}
                                        state._recent_label_side[(lbl1, side_norm)] = now
                                        state._recent_label_side[(lbl2, side_norm2)] = now
                                    except Exception:
                                        pass
                                # Telemetry bundle event
                                try:
                                    state.telemetry['events'].append({
                                        'time': now,
                                        'type': 'bundle',
                                        'labels': [lbl1, lbl2],
                                        'sides': [side_norm, side_norm2],
                                        'message': combined
                                    })
                                except Exception:
                                    pass
                                announced += 2
                                continue
                    except Exception:
                        pass
                    # Semantic scene brief (AI summary) when multiple hazards cluster
                    try:
                        ann_cfg = config.get('announcements', {})
                        brief_en = bool(ann_cfg.get('semantic_scene_brief_enabled', True))
                        if hasattr(state, 'scene_brief_enabled'):
                            brief_en = bool(state.scene_brief_enabled)
                        brief_ok = realtime and brief_en and (announced == 0) and (len(scored_detections) >= 3)
                        # Avoid brief during critical/urgent single hazard
                        if brief_ok and variant_type not in ('urgent_critical','urgent_approach'):
                            side_counts = {'left': 0, 'right': 0, 'ahead': 0}
                            labels_seen = {}
                            frame_shape = state.frame.shape if state.frame is not None else (CAMH, CAMW, 3)
                            for i in range(min(5, len(scored_detections))):
                                _, d2, _, _ = scored_detections[i]
                                fs2 = compute_bearing_fine_side(d2, frame_shape, config)
                                sn2 = 'left' if fs2 in ('left','slight_left') else ('right' if fs2 in ('right','slight_right') else 'ahead')
                                side_counts[sn2] += 1
                                lbl2 = str(d2.get('label','')).lower()
                                labels_seen[lbl2] = labels_seen.get(lbl2, 0) + 1
                            dom_side = max(side_counts.items(), key=lambda kv: kv[1])[0]
                            # Corridor hint
                            corr = state.environment_context.get('safe_corridor', {'side': None, 'width': 0.0})
                            corr_side = corr.get('side')
                            corr_hint = f"; safe {corr_side} corridor" if corr_side in ('left','right') else ''
                            # Top labels
                            top = sorted(labels_seen.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
                            lbl_desc = ', '.join([f"{kv[0]}" for kv in top]) if top else 'hazards'
                            brief = f"Scene: {lbl_desc}, mostly {dom_side}{corr_hint}."
                            speak_status(brief)
                            try:
                                with state.lock:
                                    if not realtime:
                                        state.last_announce_time = now
                                    state.last_announcement_time = now
                                    state.telemetry['events'].append({
                                        'time': now,
                                        'type': 'scene_brief',
                                        'labels': list(labels_seen.keys()),
                                        'sides': side_counts,
                                        'message': brief
                                    })
                                announced += 1
                                continue
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Coaching add-on: angle and pacing guidance
                    try:
                        ann_cfg = config.get('announcements', {})
                        coaching_en = bool(ann_cfg.get('coaching_mode_enabled', True))
                        if hasattr(state, 'coaching_enabled'):
                            coaching_en = bool(state.coaching_enabled)
                        if message and coaching_en:
                            # Estimate angle from corridor width
                            corr = state.environment_context.get('safe_corridor', {'side': None, 'width': 0.0})
                            cs = corr.get('side')
                            cw = float(corr.get('width', 0.0) or 0.0)
                            angle = 8 if cw < 0.8 else (12 if cw < 1.2 else 15)
                            pacing = 'now' if variant_type in ('urgent_critical','urgent_approach') else ('steady' if variant_type in ('moderate_approach','warning_approach') else 'smooth')
                            if cs in ('left','right'):
                                if message.endswith('.'):
                                    message = message[:-1]
                                message = f"{message}; Coach: veer {cs} {angle}°, pace {pacing}."
                    except Exception:
                        pass

                    # Temporary confirmation: exercise async speaker in detection step
                    try:
                        msg = message
                        speak_async(msg)
                    except Exception:
                        pass
                    speak_status(message)
                    _record_announcement_state_and_telemetry(label, side_norm, actual_depth, smoothed_depth, speed, importance, message, variant_type, detection, now)

            # If we didn't speak and have multiple low-importance candidates, provide a concise summary
            spoken_this_tick = _emit_summary_if_needed(summary_bucket, now, cooldown_ok) or spoken_this_tick
            # Heartbeat cue when suppressed hazards persist and no speech occurred
            _emit_heartbeat_if_needed(spoken_this_tick, heartbeat_candidates, now)
            # Resolution prompt: area clear after prior warnings (disabled in realtime mode)
            try:
                if (not realtime) and (not spoken_this_tick) and (now >= state._global_silence_until) and state._last_warning_time > 0 and (now - state._last_warning_time >= 5.0) and (now - state._last_resolution >= 8.0) and ((time.time() - state.last_announce_time) >= effective_cd):
                    # Consider clear if no detections under travel-mode warning distance
                    mode = getattr(state, 'travel_mode', 'WALKING')
                    factor = 1.0
                    if str(mode).upper() in ('DRIVING', 'CYCLING'):
                        factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
                    warning_m = WARNING_DISTANCE_M * factor
                    close = [d for d in dets if d.get('depth_m') is not None and d['depth_m'] <= warning_m]
                    if not close:
                        speak_status("Area clear ahead.")
                        with state.lock:
                            state._last_resolution = now
                            state.last_announce_time = now
                            state._last_warning_time = 0.0
                        spoken_this_tick = True
            except Exception:
                pass
        else:
            # Simple fallback mode - focus mainly on person detection
            dets_with_depth = [d for d in dets if d['depth_m'] is not None]
            if not dets_with_depth:
                continue
            
            # Sort by distance (closest first)
            dets_with_depth.sort(key=lambda d: d['depth_m'])
            
            # Check if person exists in the closest detections
            person_detected = False
            for detection in dets_with_depth[:3]:  # Check top 3 closest
                if detection['label'].lower() == 'person' and detection['conf'] >= state.min_announce_conf:
                    # Process person detection
                    phrase = 'Person'
                    win = windows_by_phrase.get(phrase, [])
                    win.append((now, detection['depth_m']))
                    cutoff = now - (1.0 if state.fast_depth else 1.5)
                    win = [w for w in win if w[0] >= cutoff]
                    windows_by_phrase[phrase] = win
                    
                    if len(win) >= 3 and cooldown_ok:
                        t0, d0 = win[0]
                        t1, d1 = win[-1]
                        total_drop = d0 - d1
                        dt = max(1e-3, t1 - t0)
                        rate = total_drop / dt
                        
                        # Check if person is approaching
                        approaching = total_drop > 0.15 and rate > 0.1
                        last_depth = state._last_class_announce.get(f"depth:{phrase}", 1e9)
                        progressed = d1 < (last_depth - 0.15)
                        
                        # Determine message based on distance and movement
                        message = None
                        if d1 <= CRITICAL_DISTANCE_M:
                            message = "Warning. Person very close."
                        elif d1 <= WARNING_DISTANCE_M and (approaching or progressed):
                            message = "Caution. Person approaching."
                        elif d1 <= 4.0 and len(win) >= 5:  # Informative for nearby person
                            message = "Person nearby."
                        
                        # Make announcement if needed
                        if message:
                            speak_status(message)
                            with state.lock:
                                state.last_announce_time = now
                                state.last_announcement_time = now
                                state._last_class_announce[f"depth:{phrase}"] = d1
                            person_detected = True
                            break  # Announce only one person
                
            # If no person announced, check for other important objects using simplified scoring
            if not person_detected and cooldown_ok:
                scored = []
                for d in dets_with_depth:
                    label = d['label'].lower()
                    if d['conf'] < state.min_announce_conf:
                        continue
                    pref = state.class_prefs.get(label)
                    if pref is None or d['depth_m'] > pref['max_m']:
                        continue
                    
                    # Simple scoring based on priority and distance
                    priority = pref['priority']
                    depth_score = max(0.0, (pref['max_m'] - d['depth_m']) / pref['max_m'])
                    score = priority * 2.0 + depth_score * 2.0 + d['conf']
                    scored.append((score, pref['phrase'], d))
                
                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    best_phrase, best_det = scored[0][1], scored[0][2]
                    
                    # Check distance thresholds for simplified announcement
                    depth = best_det['depth_m']
                    if depth <= CRITICAL_DISTANCE_M:
                        message = f"Warning. {best_phrase} very close."
                        speak_status(message, priority=10)
                    elif depth <= WARNING_DISTANCE_M:
                        message = f"Caution. {best_phrase} nearby."
                        speak_status(message, priority=8)
                    
                    # Make announcement if needed
                    if message:
                        last_time_cls = state._last_class_announce.get(best_phrase, 0.0)
                        if now - last_time_cls >= state.class_cooldown_s:
                            speak_status(message)
                            with state.lock:
                                state.last_announce_time = now
                                state.last_announcement_time = now
                                state._last_class_announce[best_phrase] = now


def voice_listener_thread():
    # Simple stub: watch command_queue; integrate real STT externally
    while state.running and state.enable_voice_listen:
        try:
            cmd = state.command_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        cmd_l = cmd.lower().strip()
        handled = False
        now = time.time()
        if cmd_l.startswith('navigate to '):
            destination = cmd_l.replace('navigate to ', '').strip()
            if destination:
                try:
                    # Open directions according to provider
                    if state.provider == 'google':
                        url = (
                            f'https://www.google.com/maps/dir/?api=1&destination={destination.replace(" ", "+")}'
                            f'&travelmode={state.travel_mode.lower()}'
                        )
                        subprocess.Popen(['open', url])
                    else:
                        subprocess.Popen(['open', f'maps://?daddr={destination}'])
                    speak_status(f"Starting directions to {destination}")
                    try:
                        with state.lock:
                            state.telemetry['voice_handled_ts'].append(now)
                        handled = True
                    except Exception:
                        pass
                except Exception:
                    pass
        elif cmd_l.startswith('search '):
            query = cmd_l.replace('search ', '').strip()
            if query:
                try:
                    if state.provider == 'google':
                        url = f'https://www.google.com/maps/search/?api=1&query={query.replace(" ", "+")}+near+me'
                        subprocess.Popen(['open', url])
                    else:
                        # Apple Maps approximate search
                        subprocess.Popen(['open', f'maps://?q={query}'])
                    speak_status(f"Searching for {query} nearby")
                    try:
                        with state.lock:
                            state.telemetry['voice_handled_ts'].append(now)
                        handled = True
                    except Exception:
                        pass
                except Exception:
                    pass
        elif cmd_l in ('use google maps', 'switch to google maps'):
            state.provider = 'google'
            speak_status('Switched to Google Maps')
            try:
                with state.lock:
                    state.telemetry['voice_handled_ts'].append(now)
                handled = True
            except Exception:
                pass
        elif cmd_l in ('use apple maps', 'switch to apple maps'):
            state.provider = 'apple'
            speak_status('Switched to Apple Maps')
            try:
                with state.lock:
                    state.telemetry['voice_handled_ts'].append(now)
                handled = True
            except Exception:
                pass
        elif cmd_l in ('walking mode', 'use walking'):
            state.travel_mode = 'WALKING'
            speak_status('Navigation mode set to walking')
            try:
                with state.lock:
                    state.telemetry['voice_handled_ts'].append(now)
                handled = True
            except Exception:
                pass
        elif cmd_l in ('driving mode', 'use driving'):
            state.travel_mode = 'DRIVING'
            speak_status('Navigation mode set to driving')
            try:
                with state.lock:
                    state.telemetry['voice_handled_ts'].append(now)
                handled = True
            except Exception:
                pass
        elif cmd_l in ('quieter', 'be quieter'):
            try:
                with state.lock:
                    # Reduce volume gently
                    v = float(getattr(state, '_speech_volume', 1.0))
                    state._speech_volume = max(0.1, round(v - 0.1, 2))
                    state.announce_cooldown_s = min(10.0, state.announce_cooldown_s * 1.5)
                    # brief hush window
                    state._global_silence_until = time.time() + 2.0
                speak_status('Okay, I will be quieter.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l in ('louder', 'be louder'):
            try:
                with state.lock:
                    v = float(getattr(state, '_speech_volume', 1.0))
                    state._speech_volume = min(1.0, round(v + 0.1, 2))
                speak_status('Volume increased.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l in ('mute', 'mute audio'):
            try:
                with state.lock:
                    state._speech_volume = 0.0
                speak_status('Muted.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l in ('unmute', 'unmute audio'):
            try:
                with state.lock:
                    if float(getattr(state, '_speech_volume', 1.0)) <= 0.0:
                        state._speech_volume = 0.6
                speak_status('Unmuted.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l.startswith('silence '):
            # e.g., "silence 10" in seconds
            try:
                secs = float(cmd_l.replace('silence', '').strip().split()[0])
            except Exception:
                secs = 2.0
            try:
                with state.lock:
                    secs = max(0.0, min(secs, 300.0))
                    state._global_silence_until = time.time() + secs
                speak_status(f'Silencing hints for {int(secs)} seconds.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l in ('unsilence', 'resume'):
            try:
                with state.lock:
                    state._global_silence_until = 0.0
                speak_status('Resuming hints.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l in ('focus center', 'focus on center'):
            try:
                with state.lock:
                    state.roi_ratio = 0.88
                    state._focus_center_until = time.time() + 30.0
                speak_status('Focusing center for 30 seconds.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l in ('crowd mode on', 'enable crowd mode', 'silent crowd on'):
            try:
                with state.lock:
                    state.context_mode = 'silent_crowd'
                speak_status('Crowd mode enabled.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l in ('crowd mode off', 'disable crowd mode', 'silent crowd off'):
            try:
                with state.lock:
                    state.context_mode = 'normal'
                speak_status('Crowd mode disabled.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass

        # TTS provider selection - local only
        elif cmd_l in ('use local tts', 'switch to local tts'):
            try:
                with state.lock:
                    config.setdefault('audio', {})['tts_provider'] = 'local'
                    state._ext_tts = None
                speak_status('Using local system voice.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass

        # AI phrasing toggles
        elif cmd_l in ('ai phrasing on', 'enable ai phrasing', 'use openai phrasing'):
            try:
                with state.lock:
                    ann = config.setdefault('announcements', {})
                    ann['phrasing_enabled'] = True
                    if 'openai' in cmd_l:
                        ann['phrasing_provider'] = 'openai'
                    from providers import get_phrasing_provider
                    state._phrasing_provider = get_phrasing_provider(config)
                speak_status('AI phrasing enabled.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l in ('ai phrasing off', 'disable ai phrasing'):
            try:
                with state.lock:
                    ann = config.setdefault('announcements', {})
                    ann['phrasing_enabled'] = False
                    state._phrasing_provider = None
                speak_status('AI phrasing disabled.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass

        # Coaching and brief toggles
        elif cmd_l in ('coach on', 'coaching on', 'enable coach'):
            try:
                with state.lock:
                    state.coaching_enabled = True
                speak_status('Coaching enabled.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l in ('coach off', 'coaching off', 'disable coach'):
            try:
                with state.lock:
                    state.coaching_enabled = False
                speak_status('Coaching disabled.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l in ('brief on', 'enable brief', 'scene brief on'):
            try:
                with state.lock:
                    state.scene_brief_enabled = True
                speak_status('Scene brief enabled.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l in ('brief off', 'disable brief', 'scene brief off'):
            try:
                with state.lock:
                    state.scene_brief_enabled = False
                speak_status('Scene brief disabled.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass

        # Style profile adjustments
        elif cmd_l in ('concise mode', 'shorter'):
            try:
                with state.lock:
                    state.ai_style_profile = 'concise'
                speak_status('Using concise phrasing.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l in ('descriptive mode', 'longer'):
            try:
                with state.lock:
                    state.ai_style_profile = 'descriptive'
                speak_status('Using descriptive phrasing.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
                # Persist preference
                try:
                    import json, os
                    prefs_path = os.path.join('logs', 'prefs.json')
                    os.makedirs('logs', exist_ok=True)
                    data = {'style_profile': 'descriptive'}
                    with open(prefs_path, 'w') as f:
                        json.dump(data, f)
                except Exception:
                    pass
            except Exception:
                pass
        elif cmd_l in ('concise mode', 'shorter'):
            try:
                with state.lock:
                    state.ai_style_profile = 'concise'
                speak_status('Using concise phrasing.')
                try:
                    with state.lock:
                        state.telemetry['voice_handled_ts'].append(now)
                    handled = True
                except Exception:
                    pass
                # Persist preference
                try:
                    import json, os
                    prefs_path = os.path.join('logs', 'prefs.json')
                    os.makedirs('logs', exist_ok=True)
                    data = {'style_profile': 'concise'}
                    with open(prefs_path, 'w') as f:
                        json.dump(data, f)
                except Exception:
                    pass
            except Exception:
                pass

        # Record unknown voice command if nothing matched
        try:
            if not handled:
                with state.lock:
                    state.telemetry['voice_unknown_ts'].append(now)
        except Exception:
            pass


def stt_mic_thread():
    """
    Optional microphone Speech-to-Text feeder for voice_listener_thread.
    If 'speech_recognition' is available, listens on default mic and puts
    recognized phrases into state.command_queue. Controlled by state.enable_voice_listen.
    """
    try:
        import speech_recognition as sr
    except Exception:
        logging.info('SpeechRecognition not available; use /voice to feed commands.')
        return
    r = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except Exception:
        logging.warning('Microphone unavailable; STT disabled.')
        return
    # Calibrate for ambient noise
    try:
        with mic as source:
            try:
                r.adjust_for_ambient_noise(source, duration=0.5)
            except Exception:
                pass
    except Exception:
        pass
    try:
        speak_status('Listening for voice commands...')
    except Exception:
        pass
    while True:
        try:
            if not getattr(state, 'enable_voice_listen', False):
                time.sleep(0.2)
                continue
            with mic as source:
                audio = r.listen(source, timeout=3, phrase_time_limit=6)
            cmd = ''
            try:
                # Use Google recognizer if available; failures are ignored
                cmd = r.recognize_google(audio)
            except Exception:
                cmd = ''
            if cmd:
                try:
                    # Sanitize and truncate overly long commands from mic
                    cmd = (cmd or '').strip()
                    try:
                        max_len = int(os.environ.get('THIRD_EYE_VOICE_CMD_MAX_LEN', '120'))
                    except Exception:
                        max_len = 120
                    if len(cmd) > max_len:
                        cmd = cmd[:max_len]
                    state.command_queue.put_nowait(cmd)
                    # Telemetry: record received voice command from mic
                    try:
                        with state.lock:
                            state.telemetry['voice_received_ts'].append(time.time())
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            time.sleep(0.5)

class _ShortcutsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            path = parsed.path
            # Alias healthz to health for standard checks
            if path == '/healthz':
                path = '/health'
            # Convenience: treat root as dashboard for easier access
            if path in ('', '/'):
                path = '/dashboard'
            # Authorization and rate limiting for sensitive endpoints
            try:
                client_ip = self.client_address[0]
            except Exception:
                client_ip = 'unknown'
            try:
                prod_gate = bool(getattr(state, '_shortcuts_prod_gate', False))
            except Exception:
                prod_gate = False
            sensitive = path in ('/metrics','/dashboard','/ota')
            try:
                require_auth = bool(getattr(state, '_shortcuts_require_token', False))
            except Exception:
                require_auth = False
            # Public paths should remain accessible for service checks
            public_paths = ('/ready','/health')
            if prod_gate and path not in public_paths:
                require_auth = True
            if sensitive:
                logging.info(f'Access {path} from {client_ip}')
            # Rate limiting for sensitive endpoints
            if not self._rate_limit_ok(path, client_ip):
                logging.warning(f'Rate limited {path} from {client_ip}')
                self._too_many('too many requests')
                return
            # Authorization
            try:
                if require_auth:
                    if not self._is_authorized(qs):
                        logging.warning(f'Unauthorized {path} from {client_ip}')
                        self._unauthorized('unauthorized')
                        return
            except Exception:
                logging.exception('Auth check error')
                self._unauthorized('unauthorized')
                return
            # Simple routing
            if path == '/toggle':
                feature = (qs.get('feature', [''])[0] or '').lower()
                state_val = (qs.get('state', [''])[0] or '').lower()
                val = state_val in ('on', 'true', '1')
                if feature == 'announce':
                    state.enable_announce = val
                elif feature == 'voice':
                    state.enable_voice_listen = val
                    try:
                        if val:
                            # Start voice listener and mic STT if not already running
                            try:
                                t_v = state.threads.get('voice_listener')
                                if t_v is None or not t_v.is_alive():
                                    t_v = threading.Thread(target=voice_listener_thread, daemon=True)
                                    t_v.start()
                                    state.threads['voice_listener'] = t_v
                            except Exception:
                                threading.Thread(target=voice_listener_thread, daemon=True).start()
                            try:
                                t_s = state.threads.get('stt')
                                if t_s is None or not t_s.is_alive():
                                    t_s = threading.Thread(target=stt_mic_thread, daemon=True)
                                    t_s.start()
                                    state.threads['stt'] = t_s
                            except Exception:
                                threading.Thread(target=stt_mic_thread, daemon=True).start()
                            speak_status('Voice mode on')
                        else:
                            speak_status('Voice mode off')
                    except Exception:
                        pass
                elif feature == 'ai':
                    state.ai_decider = val
                elif feature == 'context':
                    state.context_aware = val
                elif feature == 'realtime':
                    state.realtime_mode = val
                elif feature == 'tts_provider':
                    provider = (qs.get('value', [''])[0] or '').strip().lower()
                    if provider in ('local', 'elevenlabs'):
                        try:
                            with state.lock:
                                cfg = config.setdefault('audio', {})
                                cfg['tts_provider'] = provider
                                if provider == 'local':
                                    state._ext_tts = None
                                    # Reset backend to local default
                                    state._speech_backend = 'say'
                                else:
                                    try:
                                        from providers import get_tts_provider
                                        state._ext_tts = get_tts_provider(config)
                                    except Exception:
                                        state._ext_tts = None
                            # Persist preference
                            try:
                                import json, os
                                prefs_path = os.path.join('logs', 'prefs.json')
                                os.makedirs('logs', exist_ok=True)
                                data = {'tts_provider': provider}
                                try:
                                    # Merge with existing prefs
                                    if os.path.exists(prefs_path):
                                        with open(prefs_path, 'r') as f:
                                            prev = json.load(f)
                                            if isinstance(prev, dict):
                                                prev.update(data)
                                                data = prev
                                except Exception:
                                    pass
                                with open(prefs_path, 'w') as f:
                                    json.dump(data, f)
                            except Exception:
                                pass
                            if provider == 'local':
                                speak_status('Using local system voice.')
                            elif provider == 'elevenlabs':
                                speak_status('Using ElevenLabs for speech.')
                            self._ok(f"tts_provider set to {provider}")
                        except Exception as e:
                            self._bad(f'error: {e}')
                    else:
                        self._bad('invalid tts provider')
                    return
                elif feature == 'phrasing':
                    try:
                        with state.lock:
                            ann = config.setdefault('announcements', {})
                            ann['phrasing_enabled'] = val
                            provider = (qs.get('value', [''])[0] or qs.get('provider', [''])[0] or '').strip().lower()
                            if provider:
                                ann['phrasing_provider'] = provider
                            # Initialize or clear provider
                            try:
                                if ann['phrasing_enabled']:
                                    from providers import get_phrasing_provider
                                    state._phrasing_provider = get_phrasing_provider(config)
                                else:
                                    state._phrasing_provider = None
                            except Exception:
                                state._phrasing_provider = None
                        # Persist preference
                        try:
                            import json, os
                            prefs_path = os.path.join('logs', 'prefs.json')
                            os.makedirs('logs', exist_ok=True)
                            data = {'phrasing_enabled': bool(val)}
                            if provider:
                                data['phrasing_provider'] = provider
                            try:
                                if os.path.exists(prefs_path):
                                    with open(prefs_path, 'r') as f:
                                        prev = json.load(f)
                                        if isinstance(prev, dict):
                                            prev.update(data)
                                            data = prev
                            except Exception:
                                pass
                            with open(prefs_path, 'w') as f:
                                json.dump(data, f)
                        except Exception:
                            pass
                        speak_status('AI phrasing enabled.' if val else 'AI phrasing disabled.')
                        self._ok(json.dumps({'phrasing_enabled': bool(val), 'phrasing_provider': ann.get('phrasing_provider', 'builtin')}))
                    except Exception as e:
                        self._bad(f'error: {e}')
                    return
                elif feature == 'roi':
                    try:
                        # Toggle ROI enabled/disabled
                        with state.lock:
                            state.roi_enabled = bool(val)
                            # Optional ratio update via 'ratio' or 'value'
                            ratio_str = (qs.get('ratio', [''])[0] or qs.get('value', [''])[0] or '').strip()
                            if ratio_str:
                                try:
                                    r = float(ratio_str)
                                    state.roi_ratio = max(0.5, min(1.0, r))
                                except Exception:
                                    pass
                            # Optional full-frame cadence via 'full_every_n'
                            fen_str = (qs.get('full_every_n', [''])[0] or '').strip()
                            if fen_str:
                                try:
                                    n = int(fen_str)
                                    state.roi_full_every_n = max(1, min(20, n))
                                except Exception:
                                    pass
                        body = {
                            'roi_enabled': bool(getattr(state, 'roi_enabled', True)),
                            'roi_ratio': float(getattr(state, 'roi_ratio', 0.8)),
                            'roi_full_every_n': int(getattr(state, 'roi_full_every_n', 4)),
                        }
                        self._ok(json.dumps(body))
                    except Exception as e:
                        self._bad(f'error: {e}')
                    return
                self._ok(f"{feature} set to {val}")
            elif path == '/mode':
                travel = (qs.get('travel', [''])[0] or '').upper()
                if travel in ('WALKING', 'DRIVING'):
                    state.travel_mode = travel
                    self._ok(f"travel mode {travel}")
                else:
                    self._bad('invalid travel mode')
            elif path == '/provider':
                provider = (qs.get('map', [''])[0] or '').lower()
                if provider in ('apple', 'google'):
                    state.provider = provider
                    self._ok(f"provider {provider}")
                else:
                    self._bad('invalid provider')
            elif path == '/haptic':
                level = (qs.get('level', ['nearby'])[0] or 'nearby')
                try:
                    perform_haptic(level)
                except Exception:
                    pass
                self._ok(f"haptic {level}")
            elif path == '/imu':
                # Update IMU readings from query
                try:
                    ax = float(qs.get('ax', ['0'])[0])
                except Exception:
                    ax = 0.0
                try:
                    ay = float(qs.get('ay', ['0'])[0])
                except Exception:
                    ay = 0.0
                try:
                    az = float(qs.get('az', ['0'])[0])
                except Exception:
                    az = 0.0
                try:
                    gx = float(qs.get('gx', ['0'])[0])
                except Exception:
                    gx = 0.0
                try:
                    gy = float(qs.get('gy', ['0'])[0])
                except Exception:
                    gy = 0.0
                try:
                    gz = float(qs.get('gz', ['0'])[0])
                except Exception:
                    gz = 0.0
                with state.lock:
                    state.imu['user_acceleration'] = [ax, ay, az]
                    state.imu['rotation_rate'] = [gx, gy, gz]
                    state.imu['last_update'] = time.time()
                self._ok('imu updated')
            elif path == '/announce':
                msg = (qs.get('msg', [''])[0] or '').strip()
                if msg:
                    try:
                        speak_async(msg)
                    except Exception:
                        pass
                    speak_status(msg)
                    self._ok('spoken')
                else:
                    self._bad('missing msg')
            elif path == '/voice':
                # Feed a voice command into the command queue (for external STT)
                cmd = (qs.get('cmd', [''])[0] or '').strip()
                if not cmd:
                    self._bad('missing cmd')
                else:
                    # Enforce a max command length to prevent abuse
                    try:
                        max_len = int(os.environ.get('THIRD_EYE_VOICE_CMD_MAX_LEN', '120'))
                    except Exception:
                        max_len = 120
                    if len(cmd) > max_len:
                        self._bad('cmd too long')
                    else:
                        try:
                            state.command_queue.put_nowait(cmd)
                            # Telemetry: record received voice command
                            try:
                                with state.lock:
                                    state.telemetry['voice_received_ts'].append(time.time())
                            except Exception:
                                pass
                            self._ok('queued')
                        except Exception:
                            self._bad('command queue full')
            elif path == '/audio':
                # Live controls for audio: volume, earcons, and quiet intervals
                action = (qs.get('action', [''])[0] or '').strip().lower()
                try:
                    with state.lock:
                        if action == 'quieter':
                            v = float(getattr(state, '_speech_volume', 1.0))
                            v = max(0.1, round(v - 0.1, 2))
                            state._speech_volume = v
                            speak_status('Okay, I will be quieter.')
                            self._ok(json.dumps({'speech_volume': v}))
                        elif action == 'louder':
                            v = float(getattr(state, '_speech_volume', 1.0))
                            v = min(1.0, round(v + 0.1, 2))
                            state._speech_volume = v
                            speak_status('Volume increased.')
                            self._ok(json.dumps({'speech_volume': v}))
                        elif action == 'mute':
                            state._speech_volume = 0.0
                            speak_status('Muted.')
                            self._ok(json.dumps({'speech_volume': 0.0}))
                        elif action == 'unmute':
                            prev = float(getattr(state, '_speech_volume', 1.0))
                            if prev <= 0.0:
                                state._speech_volume = 0.6
                            speak_status('Unmuted.')
                            self._ok(json.dumps({'speech_volume': float(getattr(state, '_speech_volume', 0.6))}))
                        elif action == 'set_volume':
                            val_str = (qs.get('value', [''])[0] or '').strip()
                            try:
                                val = float(val_str)
                                val = min(1.0, max(0.0, val))
                                state._speech_volume = val
                                self._ok(json.dumps({'speech_volume': val}))
                            except Exception:
                                self._bad('invalid value')
                        elif action == 'set_earcon_volume':
                            val_str = (qs.get('value', [''])[0] or '').strip()
                            try:
                                val = float(val_str)
                                val = min(1.0, max(0.0, val))
                                state._earcon_volume = val
                                self._ok(json.dumps({'earcon_volume': val}))
                            except Exception:
                                self._bad('invalid value')
                        elif action == 'enable_earcons':
                            state._earcons_enabled = True
                            self._ok(json.dumps({'earcons_enabled': True}))
                        elif action == 'disable_earcons':
                            state._earcons_enabled = False
                            self._ok(json.dumps({'earcons_enabled': False}))
                        elif action == 'silence':
                            secs_str = (qs.get('secs', [''])[0] or '').strip()
                            try:
                                secs = float(secs_str or '0.0')
                                secs = max(0.0, min(secs, 300.0))
                                state._global_silence_until = time.time() + secs
                                self._ok(json.dumps({'global_silence_until': state._global_silence_until, 'silence_secs': secs}))
                            except Exception:
                                self._bad('invalid secs')
                        elif action == 'unsilence':
                            state._global_silence_until = 0.0
                            self._ok(json.dumps({'global_silence_until': state._global_silence_until}))
                        else:
                            self._bad('unknown action')
                except Exception as e:
                    self._bad(f'error: {e}')
            elif path == '/audio/voice':
                # Return current TTS provider and voice selection
                try:
                    with state.lock:
                        aud = config.get('audio', {})
                        provider = aud.get('tts_provider', config.get('tts_provider', 'local'))
                        voice_id = aud.get('voice_id', '')
                        self._ok(json.dumps({'provider': provider, 'voice_id': voice_id}))
                except Exception as e:
                    self._bad(f'error: {e}')
            elif path == '/permissions':
                # Trigger immediate permission check
                try:
                    ok = _permission_check_once()
                    self._ok(f"permissions ok={ok}")
                except Exception:
                    self._bad('permission check failed')
            elif path == '/metrics':
                # Return telemetry and performance metrics as JSON
                try:
                    now = time.time()
                    # Verbose extras toggle (gated by auth when enabled)
                    verbose_param = (qs.get('verbose', ['0'])[0] or '').lower() in ('1','true','yes')
                    allow_verbose = True
                    try:
                        need_auth = bool(getattr(state, '_shortcuts_require_token', False)) or bool(getattr(state, '_shortcuts_prod_gate', False))
                        if need_auth:
                            allow_verbose = self._is_authorized(qs)
                    except Exception:
                        allow_verbose = False
                    verbose = verbose_param and allow_verbose
                    with state.lock:
                        tel = state.telemetry
                        def count_since(ts_list, window_s):
                            cutoff = now - window_s
                            return sum(1 for t in ts_list if t >= cutoff)
                        # Window for event/importance counts
                        win_s = float(getattr(state, 'metrics_window_seconds', 900.0) or 900.0)
                        uptime = max(0.0, now - float(tel.get('start_time', now)))
                        # Compute recent events with opt-in redaction
                        try:
                            include_recent = bool(config.get('logging', {}).get('telemetry_include_recent_events', True))
                        except Exception:
                            include_recent = True
                        events_recent = []
                        if include_recent and verbose:
                            events_recent = list(tel.get('events', []))[-20:]
                            try:
                                redact = config.get('logging', {}).get('telemetry_redact_fields', [])
                                if isinstance(redact, list) and redact:
                                    for ev in events_recent:
                                        try:
                                            for k in redact:
                                                if k in ev:
                                                    ev.pop(k, None)
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                        # Prepare stable defaults to ensure a consistent shape
                        metrics = {
                            'uptime_s': 0.0,
                            'announce_total': 0,
                            'critical_total': 0,
                            'suppressed_total': 0,
                            'summary_total': 0,
                            'announce_rate_60s': 0.0,
                            'critical_rate_60s': 0.0,
                            'suppressed_rate_60s': 0.0,
                            'summary_rate_60s': 0.0,
                            # Voice command totals and rates
                            'voice_received_total': 0,
                            'voice_handled_total': 0,
                            'voice_unknown_total': 0,
                            'voice_received_rate_60s': 0.0,
                            'voice_handled_rate_60s': 0.0,
                            'voice_unknown_rate_60s': 0.0,
                            # Approach-tier totals and rates
                            'moderate_approach_total': 0,
                            'slow_approach_total': 0,
                            'moderate_approach_rate_60s': 0.0,
                            'slow_approach_rate_60s': 0.0,
                            'moderate_left_rate_60s': 0.0,
                            'moderate_right_rate_60s': 0.0,
                            'slow_left_rate_60s': 0.0,
                            'slow_right_rate_60s': 0.0,
                            'approach_repeat_suppressed_total': 0,
                            'approach_repeat_suppressed_rate_60s': 0.0,
                            'fps': float(getattr(state, 'last_fps', 0.0) or 0.0),
                            'quality': float(state.performance_stats.get('quality_level', 1.0) or 1.0),
                            'cpu_avg': 0.0,
                            'mem_avg': 0.0,
                            # Audio rolling averages (ms)
                            'tts_queue_avg_ms_10': 0.0,
                            'tts_speak_avg_ms_10': 0.0,
                            'tts_queue_avg_ms_50': 0.0,
                            'tts_speak_avg_ms_50': 0.0,
                            # Current audio state
                            'audio_queue_size': int(audio_queue.qsize()),
                            'speech_backend': str(getattr(state, '_speech_backend', 'say')),
                            'speech_volume': float(getattr(state, '_speech_volume', 1.0) or 1.0),
                            'earcon_volume': float(getattr(state, '_earcon_volume', 0.8) or 0.8),
                            'earcons_enabled': bool(getattr(state, '_earcons_enabled', True)),
                            'global_silence_remaining_s': max(0.0, float(getattr(state, '_global_silence_until', 0.0) or 0.0) - now),
                            'clear_path_persist': bool(state.environment_context.get('clear_path', {}).get('persist', False)),
                            'clear_path_corridor_frac': float(state.environment_context.get('clear_path', {}).get('corridor_frac', 0.0) or 0.0),
                            'travel_mode': str(getattr(state, 'travel_mode', 'WALKING')),
                            'provider': str(getattr(state, 'provider', 'apple')),
                            'context_aware': bool(getattr(state, 'context_aware', True)),
                            'context_mode': str(getattr(state, 'context_mode', 'normal')),
                            'importance_threshold': float(getattr(state, 'min_importance_score', float(config.get('announcements', {}).get('min_importance_score', 0.6)))),
                            'ai_decider': bool(getattr(state, 'ai_decider', True)),
                            'degraded_mode': bool(getattr(state, 'degraded_mode', False)),
                            'mock_sensor': bool(getattr(state, 'mock_sensor', False)),
                            # Placeholders for derived metrics
                            'per_class': {},
                            'event_type_counts': {},
                            'importance_counts': {},
                            # Queue latency histograms (in milliseconds)
                            'queue_latencies': {
                                'tts_queue': [],
                                'tts_speak': [],
                                'audio_queue': []
                            },
                            # Per-class history for trendlines
                            'per_class_history': {},
                            # Recent events (may be redacted)
                            'events_recent': events_recent
                        }
                        # Fill with live telemetry where available
                        metrics['uptime_s'] = uptime
                        metrics['announce_total'] = len(tel.get('announce_ts', []))
                        metrics['critical_total'] = len(tel.get('critical_ts', []))
                        metrics['suppressed_total'] = len(tel.get('suppressed_ts', []))
                        metrics['summary_total'] = len(tel.get('summary_ts', []))
                        metrics['announce_rate_60s'] = count_since(tel.get('announce_ts', []), 60.0) / 60.0
                        metrics['critical_rate_60s'] = count_since(tel.get('critical_ts', []), 60.0) / 60.0
                        metrics['suppressed_rate_60s'] = count_since(tel.get('suppressed_ts', []), 60.0) / 60.0
                        metrics['summary_rate_60s'] = count_since(tel.get('summary_ts', []), 60.0) / 60.0
                        # Voice metrics
                        metrics['voice_received_total'] = len(tel.get('voice_received_ts', []))
                        metrics['voice_handled_total'] = len(tel.get('voice_handled_ts', []))
                        metrics['voice_unknown_total'] = len(tel.get('voice_unknown_ts', []))
                        metrics['voice_received_rate_60s'] = count_since(tel.get('voice_received_ts', []), 60.0) / 60.0
                        metrics['voice_handled_rate_60s'] = count_since(tel.get('voice_handled_ts', []), 60.0) / 60.0
                        metrics['voice_unknown_rate_60s'] = count_since(tel.get('voice_unknown_ts', []), 60.0) / 60.0
                        # Feedback metrics
                        metrics['feedback_total'] = len(tel.get('feedback_ts', []))
                        metrics['feedback_rate_60s'] = count_since(tel.get('feedback_ts', []), 60.0) / 60.0
                        metrics['moderate_approach_total'] = len(tel.get('moderate_approach_ts', []))
                        metrics['slow_approach_total'] = len(tel.get('slow_approach_ts', []))
                        metrics['moderate_approach_rate_60s'] = count_since(tel.get('moderate_approach_ts', []), 60.0) / 60.0
                        metrics['slow_approach_rate_60s'] = count_since(tel.get('slow_approach_ts', []), 60.0) / 60.0
                        metrics['moderate_left_rate_60s'] = count_since(tel.get('moderate_left_ts', []), 60.0) / 60.0
                        metrics['moderate_right_rate_60s'] = count_since(tel.get('moderate_right_ts', []), 60.0) / 60.0
                        metrics['slow_left_rate_60s'] = count_since(tel.get('slow_left_ts', []), 60.0) / 60.0
                        metrics['slow_right_rate_60s'] = count_since(tel.get('slow_right_ts', []), 60.0) / 60.0
                        metrics['approach_repeat_suppressed_total'] = len(tel.get('approach_repeat_suppressed_ts', []))
                        metrics['approach_repeat_suppressed_rate_60s'] = count_since(tel.get('approach_repeat_suppressed_ts', []), 60.0) / 60.0
                        # CPU/memory rolling averages
                        if state.performance_stats.get('cpu_usage'):
                            metrics['cpu_avg'] = float(np.mean(state.performance_stats['cpu_usage'][-10:]))
                        if state.performance_stats.get('memory_usage'):
                            metrics['mem_avg'] = float(np.mean(state.performance_stats['memory_usage'][-10:]))
                        # Audio rolling averages (ms)
                        if state.performance_stats.get('tts_queue_latencies'):
                            metrics['tts_queue_avg_ms_10'] = float(np.mean(state.performance_stats['tts_queue_latencies'][-10:]) * 1000.0)
                            metrics['tts_queue_avg_ms_50'] = float(np.mean(state.performance_stats['tts_queue_latencies'][-50:]) * 1000.0)
                        if state.performance_stats.get('tts_speak_latencies'):
                            metrics['tts_speak_avg_ms_10'] = float(np.mean(state.performance_stats['tts_speak_latencies'][-10:]) * 1000.0)
                            metrics['tts_speak_avg_ms_50'] = float(np.mean(state.performance_stats['tts_speak_latencies'][-50:]) * 1000.0)
                        # Per-class metrics
                        metrics['per_class'] = {k: {
                            'total': len(v),
                            'rate_60s': count_since(v, 60.0) / 60.0
                        } for k, v in tel.get('per_class_ts', {}).items()}
                        # Event type/importance counters
                        metrics['event_type_counts'] = {k: (count_since(v, win_s) if isinstance(v, (deque, list)) else int(v)) for k, v in tel.get('event_type_counts', {}).items()}
                        metrics['importance_counts'] = {k: (count_since(v, win_s) if isinstance(v, (deque, list)) else int(v)) for k, v in tel.get('importance_counts', {}).items()}
                        # Queue latency histograms (in milliseconds)
                        metrics['queue_latencies'] = {
                            'tts_queue': list(tel.get('queue_latencies', {}).get('tts_queue', [])),
                            'tts_speak': list(tel.get('queue_latencies', {}).get('tts_speak', [])),
                            'audio_queue': list(tel.get('queue_latencies', {}).get('audio_queue', []))
                        }
                        # Per-class history for trendlines
                        metrics['per_class_history'] = {k: list(v) for k, v in tel.get('per_class_history', {}).items()}
                        # Provider and circuit-breaker status (TTS and phrasing)
                        try:
                            # TTS provider name from config
                            tts_provider = str(config.get('audio', {}).get('tts_provider', 'local') or 'local')
                        except Exception:
                            tts_provider = 'local'
                        try:
                            # Phrasing toggles and provider
                            ann_cfg = config.get('announcements', {})
                            phrasing_enabled = bool(ann_cfg.get('phrasing_enabled', False))
                            phrasing_provider = str(ann_cfg.get('phrasing_provider', 'builtin') or 'builtin')
                        except Exception:
                            phrasing_enabled = False
                            phrasing_provider = 'builtin'
                        # Circuit breaker snapshots
                        tts_cb = dict(getattr(state, '_tts_cb', {}) or {})
                        phr_cb = dict(getattr(state, '_phr_cb', {}) or {})
                        # Compute cooldown remaining seconds for open breakers
                        def _cooldown_remaining(cb_obj):
                            try:
                                if bool(cb_obj.get('open', False)):
                                    cd = float(cb_obj.get('cooldown_s', 60.0) or 60.0)
                                    opened_at = float(cb_obj.get('opened_at', 0.0) or 0.0)
                                    return max(0.0, cd - (now - opened_at))
                            except Exception:
                                pass
                            return 0.0
                        metrics['tts_provider'] = tts_provider
                        metrics['phrasing_enabled'] = phrasing_enabled
                        metrics['phrasing_provider'] = phrasing_provider
                        metrics['tts_cb'] = {
                            'open': bool(tts_cb.get('open', False)),
                            'fail_count': int(tts_cb.get('fail_count', 0) or 0),
                            'threshold': int(tts_cb.get('threshold', 3) or 3),
                            'cooldown_s': float(tts_cb.get('cooldown_s', 60.0) or 60.0),
                            'cooldown_remaining_s': float(_cooldown_remaining(tts_cb)),
                        }
                        metrics['phr_cb'] = {
                            'open': bool(phr_cb.get('open', False)),
                            'fail_count': int(phr_cb.get('fail_count', 0) or 0),
                            'threshold': int(phr_cb.get('threshold', 3) or 3),
                            'cooldown_s': float(phr_cb.get('cooldown_s', 60.0) or 60.0),
                            'cooldown_remaining_s': float(_cooldown_remaining(phr_cb)),
                        }
                        if verbose:
                            events_sample = list(tel.get('events', []))[-50:]
                            try:
                                redact = config.get('logging', {}).get('telemetry_redact_fields', [])
                                if isinstance(redact, list) and redact:
                                    for ev in events_sample:
                                        try:
                                            for k in redact:
                                                if k in ev:
                                                    ev.pop(k, None)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            metrics['events_sample'] = events_sample
                    self._json(metrics)
                except Exception:
                    self._bad('metrics error')
            elif path == '/ota':
                # Admin OTA controls: status, check now, defer
                try:
                    action = (qs.get('action', ['status'])[0] or 'status').strip().lower()
                    with state.lock:
                        upd = config.get('updates', {})
                        enabled = bool(upd.get('ota_enabled', False))
                        auto_apply = bool(upd.get('auto_apply', True))
                        base_url = str(upd.get('base_url', '') or '')
                        interval = float(upd.get('check_interval_s', 900.0))
                        defer_until = float(getattr(state, '_ota_defer_until_ts', 0.0) or 0.0)
                        force_pending = bool(getattr(state, '_ota_force_check', False))
                        last_check = float(getattr(state, '_ota_last_check_ts', 0.0) or 0.0)
                    if action == 'status':
                        body = {
                            'enabled': enabled,
                            'auto_apply': auto_apply,
                            'base_url': base_url,
                            'check_interval_s': interval,
                            'defer_until_ts': defer_until,
                            'force_pending': force_pending,
                            'last_check_ts': last_check,
                        }
                        return self._json(body)
                    elif action == 'check':
                        # Queue an immediate manifest check (optionally apply)
                        apply_str = (qs.get('apply', ['1'])[0] or '1').strip().lower()
                        apply_now = apply_str in ('1','true','yes','on')
                        with state.lock:
                            setattr(state, '_ota_force_check', True)
                            setattr(state, '_ota_force_apply', bool(apply_now))
                        try:
                            speak_status('OTA check queued')
                        except Exception:
                            pass
                        return self._ok('queued')
                    elif action == 'defer':
                        try:
                            secs = float((qs.get('secs', ['600'])[0] or '600'))
                        except Exception:
                            secs = 600.0
                        secs = max(1.0, min(secs, 86400.0))
                        until = time.time() + secs
                        with state.lock:
                            setattr(state, '_ota_defer_until_ts', until)
                        try:
                            speak_status('OTA deferred')
                        except Exception:
                            pass
                        return self._ok(json.dumps({'defer_until_ts': until, 'defer_secs': secs}))
                    elif action == 'auto':
                        # Toggle auto apply setting
                        val = (qs.get('value', [''])[0] or '').strip().lower()
                        if val in ('enable','on','true','1'):
                            with state.lock:
                                config.setdefault('updates', {})['auto_apply'] = True
                            return self._ok('auto_apply enabled')
                        elif val in ('disable','off','false','0'):
                            with state.lock:
                                config.setdefault('updates', {})['auto_apply'] = False
                            return self._ok('auto_apply disabled')
                        else:
                            return self._bad('invalid value')
                    else:
                        return self._bad('unknown action')
                except Exception:
                    self._bad('ota error')
            elif path == '/ready':
                try:
                    with state.lock:
                        ready_flag = bool(getattr(state, 'ready', False))
                        core = ['yolo', 'depth']
                        threads_ok = all((state.threads.get(n) is not None) and state.threads[n].is_alive() for n in core)
                        running = bool(getattr(state, 'running', False))
                    body = {'ready': bool(ready_flag and threads_ok and running)}
                    self.send_response(200 if body['ready'] else 503)
                    self._apply_security_headers()
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(json.dumps(body).encode('utf-8'))
                except Exception:
                    self._bad('readiness error')
            elif path == '/health':
                try:
                    now = time.time()
                    with state.lock:
                        threads = {name: (t.is_alive() if t is not None else False) for name, t in state.threads.items()}
                        required = ['camera','yolo','depth','watchdog','shortcuts']
                        last_yolo = float(getattr(state, 'last_yolo_time', 0.0) or 0.0)
                        last_depth = float(getattr(state, '_last_depth_time', 0.0) or 0.0)
                        last_frame = float(getattr(state, '_last_frame_time', 0.0) or 0.0)
                        yolo_stall_s = float(config.get('watchdog', {}).get('yolo_stall_s', 2.5))
                        depth_stall_s = float(config.get('watchdog', {}).get('depth_stall_s', 2.5))
                        cam_stall_s = float(config.get('watchdog', {}).get('camera_stall_s', 2.5))
                        stall = {
                            'yolo': (now - last_yolo) > yolo_stall_s,
                            'depth': (now - last_depth) > depth_stall_s,
                            'camera': (now - last_frame) > cam_stall_s,
                        }
                        ok_threads = all(threads.get(n, False) for n in required)
                        min_fps = float(config.get('performance', {}).get('min_fps', 12.0))
                        fps = float(getattr(state, 'last_fps', 0.0) or 0.0)
                        ok = bool(state.running and ok_threads and (not any(stall.values())) and (fps >= min_fps))
                        health = {
                            'ok': ok,
                            'running': bool(state.running),
                            'threads': threads,
                            'last_fps': fps,
                            'min_fps': min_fps,
                            'last_yolo_age_s': max(0.0, now - last_yolo),
                            'last_depth_age_s': max(0.0, now - last_depth),
                            'last_frame_age_s': max(0.0, now - last_frame),
                            'stall': stall,
                            'restart_times': {k: float(v) for k, v in getattr(state, '_last_restart', {}).items()},
                            'degraded_mode': bool(getattr(state, 'degraded_mode', False)),
                            'mock_sensor': bool(getattr(state, 'mock_sensor', False)),
                        }
                        # Verbose extras on demand
                        verbose = (qs.get('verbose', ['0'])[0] or '').lower() in ('1','true','yes')
                        if verbose:
                            # Queue depths and buffers
                            extras = {
                                'queues': {
                                    'audio_q': int(audio_queue.qsize()),
                                    'preproc_q': int(state.preproc_queue.qsize()) if getattr(state, 'use_preproc_queue', False) else 0,
                                    'preproc_q_max': int(getattr(state.preproc_queue, 'maxsize', 0)) if getattr(state, 'use_preproc_queue', False) else 0,
                                    'command_q': int(state.command_queue.qsize()) if hasattr(state, 'command_queue') else 0,
                                },
                                'detections': {
                                    'current': int(len(state.detections) if state.detections is not None else 0),
                                    'tracker': int(len(state.tracker_detections) if state.tracker_detections is not None else 0),
                                },
                                'imu_age_s': max(0.0, now - float(state.imu.get('last_update', 0.0) or 0.0)),
                                'perf': {
                                    'cpu_avg': float(np.mean(state.performance_stats['cpu_usage'][-10:])) if state.performance_stats.get('cpu_usage') else 0.0,
                                    'mem_avg': float(np.mean(state.performance_stats['memory_usage'][-10:])) if state.performance_stats.get('memory_usage') else 0.0,
                                    'quality': float(state.performance_stats.get('quality_level', 1.0) or 1.0),
                                    # Latency averages (ms)
                                    'avg_yolo_inf_ms': float(np.mean(state.performance_stats['yolo_inference_times'][-20:]) * 1000.0) if state.performance_stats.get('yolo_inference_times') else 0.0,
                                    'avg_depth_inf_ms': float(np.mean(state.performance_stats['depth_inference_times'][-20:]) * 1000.0) if state.performance_stats.get('depth_inference_times') else 0.0,
                                    'avg_depth_total_ms': float(np.mean(state.performance_stats['depth_total_times'][-20:]) * 1000.0) if state.performance_stats.get('depth_total_times') else 0.0,
                                    'avg_cam_capture_ms': float(np.mean(state.performance_stats['camera_capture_times'][-20:]) * 1000.0) if state.performance_stats.get('camera_capture_times') else 0.0,
                                    'avg_cam_preproc_ms': float(np.mean(state.performance_stats['camera_preproc_times'][-20:]) * 1000.0) if state.performance_stats.get('camera_preproc_times') else 0.0,
                                    'avg_tts_queue_ms': float(np.mean(state.performance_stats['tts_queue_latencies'][-20:]) * 1000.0) if state.performance_stats.get('tts_queue_latencies') else 0.0,
                                    'avg_tts_speak_ms': float(np.mean(state.performance_stats['tts_speak_latencies'][-20:]) * 1000.0) if state.performance_stats.get('tts_speak_latencies') else 0.0,
                                },
                                'audio': {
                                    'speech_backend': str(getattr(state, '_speech_backend', 'say')),
                                    'speech_volume': float(getattr(state, '_speech_volume', 1.0) or 1.0),
                                    'earcon_volume': float(getattr(state, '_earcon_volume', 0.8) or 0.8),
                                    'earcons_enabled': bool(getattr(state, '_earcons_enabled', True)),
                                    'global_silence_until': float(getattr(state, '_global_silence_until', 0.0) or 0.0),
                                    'global_silence_remaining_s': max(0.0, float(getattr(state, '_global_silence_until', 0.0) or 0.0) - now),
                                    'enqueue_rate_limit_s': float(config.get('audio', {}).get('enqueue_rate_limit_s', 0.8)),
                                    'max_queue_size': int(config.get('audio', {}).get('max_queue_size', 2)),
                                    'dedup_window_s': float(config.get('audio', {}).get('dedup_window_s', 2.0))
                                },
                                'skips': {
                                    'camera_skips': int(state.performance_stats.get('camera_skips', 0) or 0),
                                    'yolo_skips': int(state.performance_stats.get('yolo_skips', 0) or 0),
                                    'depth_skips': int(state.performance_stats.get('depth_skips', 0) or 0),
                                },
                                'device': dict(getattr(state, 'device_info', {}) or {}),
                                'thresholds': {
                                    'yolo_stall_s': yolo_stall_s,
                                    'depth_stall_s': depth_stall_s,
                                    'camera_stall_s': cam_stall_s,
                                    'min_fps': min_fps,
                                },
                                'intervals': {
                                    'yolo_interval_s': float(getattr(state, 'yolo_interval', 0.1) or 0.1),
                                    'depth_min_interval_s': float(getattr(state, 'depth_min_interval', 0.04) or 0.04),
                                }
                            }
                            health.update(extras)
                    self._json(health)
                except Exception:
                    self._bad('health error')
            elif path == '/display.jpg':
                # Serve the latest display JPEG for live preview
                try:
                    if not self._is_authorized(qs):
                        return self._unauthorized()
                    with state.lock:
                        payload = getattr(state, '_display_jpeg', None)
                        ts = float(getattr(state, '_display_jpeg_time', 0.0) or 0.0)
                    if not payload:
                        self.send_response(404)
                        self._apply_security_headers()
                        self.send_header('Content-Type', 'text/plain')
                        self.end_headers()
                        try:
                            self.wfile.write(b'no frame available')
                        except Exception:
                            pass
                    else:
                        self.send_response(200)
                        self._apply_security_headers()
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Cache-Control', 'no-store, max-age=0')
                        try:
                            from email.utils import formatdate
                            if ts > 0:
                                self.send_header('Last-Modified', formatdate(ts, usegmt=True))
                        except Exception:
                            pass
                        self.end_headers()
                        try:
                            self.wfile.write(payload)
                        except Exception:
                            pass
                except Exception:
                    self._bad('display error')
            elif path == '/preview':
                # Serve the latest camera JPEG (alias of display) for easier embedding
                try:
                    if not self._is_authorized(qs):
                        return self._unauthorized()
                    with state.lock:
                        payload = getattr(state, '_display_jpeg', None)
                        ts = float(getattr(state, '_display_jpeg_time', 0.0) or 0.0)
                    if not payload:
                        self.send_response(404)
                        self._apply_security_headers()
                        self.send_header('Content-Type', 'text/plain')
                        self.end_headers()
                        try:
                            self.wfile.write(b'no frame available')
                        except Exception:
                            pass
                    else:
                        self.send_response(200)
                        self._apply_security_headers()
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Cache-Control', 'no-store, max-age=0')
                        try:
                            from email.utils import formatdate
                            if ts > 0:
                                self.send_header('Last-Modified', formatdate(ts, usegmt=True))
                        except Exception:
                            pass
                        self.end_headers()
                        try:
                            self.wfile.write(payload)
                        except Exception:
                            pass
                except Exception:
                    self._bad('preview error')
            elif path == '/dashboard':
                # Simple auto-refresh dashboard
                try:
                    html = """<!doctype html><html><head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Third Eye Metrics</title>
<style>body{font-family:system-ui,Arial;padding:16px;color:#222;background:#fafafa}
.grid{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:12px}
.card{background:#fff;border:1px solid #eee;border-radius:8px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,0.05)}
h1{margin:0 0 12px 0;font-size:20px} h2{margin:0 0 8px 0;font-size:16px}
.mono{font-family:Menlo,monospace}
.spark svg{display:block;width:100%;height:40px}
.spark path{stroke-linejoin:round}
.small{font-size:12px;color:#666}
.histogram{display:flex;align-items:end;height:60px;gap:2px;margin:8px 0}
.histogram .bar{background:#2a8;min-width:3px;border-radius:1px 1px 0 0}
.trendline{height:30px;margin:4px 0}
.metric-row{display:flex;justify-content:space-between;margin:2px 0}
</style>
</head><body>
<h1>Third Eye Runtime Dashboard</h1>
<div id=\"meta\" class=\"mono\"></div>
<div id=\"status\" class=\"small\"></div>
<div class=\"grid\">
  <div class=\"card\"><h2>Announcements</h2><div id=\"ann\"></div><div id=\"ann_spark\" class=\"spark\"></div></div>
  <div class=\"card\"><h2>Suppressions</h2><div id=\"sup\"></div><div id=\"sup_spark\" class=\"spark\"></div></div>
  <div class=\"card\"><h2>Event Types</h2><div id=\"event_types\"></div></div>
  <div class=\"card\"><h2>Event Importance</h2><div id=\"importance\"></div></div>
  <div class=\"card\"><h2>Queue Latencies</h2><div id=\"queue_latencies\"></div></div>
  <div class=\"card\"><h2>Top Classes</h2><div id=\"top_classes\"></div></div>
  <div class=\"card\"><h2>Events</h2><div id=\"ev\" class=\"mono small\"></div></div>
  <div class=\"card\"><h2>System</h2><div id=\"sys\"></div></div>
  <div class=\"card\"><h2>Clear Path</h2><div id=\"cp\"></div></div>
</div>
<script>
let annHist=[], supHist=[];
function lineChart(elId, data, {color='#2a8', width=160, height=40}={}){
  const el=document.getElementById(elId);
  if(!el){return}
  if(!data || data.length===0){el.innerHTML='';return}
  const max=Math.max(...data), min=Math.min(...data);
  const step=data.length>1 ? width/(data.length-1) : width;
  let d='';
  for(let i=0;i<data.length;i++){
    const x=i*step;
    const y=(max===min)? height/2 : (height - ((data[i]-min)/(max-min))*height);
    d += (i===0? 'M':' L') + x + ',' + y;
  }
  el.innerHTML = `<svg width="${width}" height="${height}"><path d="${d}" stroke="${color}" fill="none" stroke-width="1.5"/></svg>`;
}
function histogram(elId, data, maxBars=20){
  const el=document.getElementById(elId);
  if(!el || !data || data.length===0){el.innerHTML='<div class="small">No data</div>';return}
  const max=Math.max(...data);
  if(max===0){el.innerHTML='<div class="small">No latency data</div>';return}
  const bins=Math.min(maxBars, Math.max(5, Math.ceil(Math.sqrt(data.length))));
  const binSize=max/bins;
  const counts=new Array(bins).fill(0);
  for(const v of data){
    const bin=Math.min(bins-1, Math.floor(v/binSize));
    counts[bin]++;
  }
  const maxCount=Math.max(...counts);
  const bars=counts.map((count,i)=>{
    const height=maxCount>0 ? (count/maxCount)*100 : 0;
    const label=`${(i*binSize).toFixed(0)}-${((i+1)*binSize).toFixed(0)}ms`;
    return `<div class="bar" style="height:${height}%" title="${label}: ${count} samples"></div>`;
  }).join('');
  el.innerHTML = `<div class="histogram">${bars}</div><div class="small">Latency distribution (${data.length} samples)</div>`;
}
function sanitizeId(s){return String(s).replace(/[^a-zA-Z0-9]/g,'_')}
function getEl(id){return document.getElementById(id)}
function setHtml(id, html){const el=getEl(id); if(el) el.innerHTML=html}
function setText(id, text){const el=getEl(id); if(el) el.textContent=text}
function formatTelemetry(n){return typeof n === 'number' ? n.toFixed(2) : n}
function renderKeyValueList(elId, obj){
  const el=getEl(elId);
  if(!el){return}
  const keys=Object.keys(obj||{});
  el.innerHTML = keys.length ? keys.map(k=>`<div class=\"metric-row\"><span>${k}:</span><span>${obj[k]}</span></div>`).join('') : '<div class=\"small\">No data</div>';
}
function formatEvent(e){
  const ts = new Date((e.time||0)*1000).toLocaleTimeString();
  const tag = e.type || 'event';
  const lbl = e.label ? ` [${e.label}${e.side?':' + e.side:''}]` : '';
  const imp = (typeof e.importance==='number') ? ` imp:${formatTelemetry(e.importance)}` : '';
  const depth = (typeof e.depth_m==='number') ? ` d:${formatTelemetry(e.depth_m)}m` : '';
  const speed = (typeof e.speed_mps==='number') ? ` v:${formatTelemetry(e.speed_mps)}m/s` : '';
  const msg = e.message || '';
  return `<div>${ts} — ${tag}${lbl}${imp}${depth}${speed}: ${msg}</div>`;
}
let retryMs = 1000; const maxRetryMs = 10000;
function showStatus(type, msg){
  const bg = type==='ok' ? '#eef7ee' : '#fff2f2';
  const fg = type==='ok' ? '#0a4' : '#900';
  setHtml('status', `<div class=\"small\" style=\"padding:4px 6px;border-radius:4px;background:${bg};color:${fg}\">${msg}</div>`);
}
function scheduleNext(ms){ setTimeout(refresh, ms); }
async function refresh(){
  try{
    const params = new URLSearchParams(window.location.search);
    const tok = params.get('token');
    const mUrl = tok ? `/metrics?verbose=1&token=${encodeURIComponent(tok)}` : '/metrics?verbose=1';
    const hUrl = tok ? `/health?token=${encodeURIComponent(tok)}` : '/health';
    const [mResp, hResp] = await Promise.all([
      fetch(mUrl),
      fetch(hUrl)
    ]);
    if(!mResp.ok || !hResp.ok){ throw new Error(`HTTP ${mResp.status}/${hResp.status}`); }
    const m = await mResp.json();
    const h = await hResp.json();
    const defNum = (v,d=0)=> (typeof v==='number' && !Number.isNaN(v) ? v : d);
    const defStr = (v,d='')=> (typeof v==='string' ? v : d);
    const defBool = (v,d=false)=> (typeof v==='boolean' ? v : d);
    const safeArr = (v)=> (Array.isArray(v) ? v : []);
    const safeObj = (v)=> (v && typeof v==='object' ? v : {});

    // Meta
    setText('meta', `Mode: ${defStr(m.travel_mode,'WALKING')} | Provider: ${defStr(m.provider,'apple')} | Uptime: ${formatTelemetry(defNum(m.uptime_s,0))}s`);
    // Announcement counters
    const announce_total = defNum(m.announce_total, 0);
    const announce_rate = defNum(m.announce_rate_60s, 0);
    const critical_total = defNum(m.critical_total, 0);
    const critical_rate = defNum(m.critical_rate_60s, 0);
    const suppressed_total = defNum(m.suppressed_total, 0);
    const suppressed_rate = defNum(m.suppressed_rate_60s, 0);
    const summary_total = defNum(m.summary_total, 0);
    const summary_rate = defNum(m.summary_rate_60s, 0);
    setHtml('ann', `Total: ${announce_total}<br/>Rate(60s): ${formatTelemetry(announce_rate)} /s<br/>Critical Total: ${critical_total}<br/>Critical Rate: ${formatTelemetry(critical_rate)} /s`);
    setHtml('sup', `Suppressed Total: ${suppressed_total}<br/>Rate(60s): ${formatTelemetry(suppressed_rate)} /s<br/>Summary Total: ${summary_total}<br/>Summary Rate: ${formatTelemetry(summary_rate)} /s`);
    annHist.push(announce_rate); if(annHist.length>50) annHist.shift();
    supHist.push(suppressed_rate); if(supHist.length>50) supHist.shift();
    lineChart('ann_spark', annHist, {color:'#2a8', width:160, height:40});
    lineChart('sup_spark', supHist, {color:'#e67', width:160, height:40});

    // System status
    const fps = defNum(m.fps, 0.0);
    const quality = defNum(m.quality, 1.0);
    const cpu = defNum(m.cpu_avg, 0.0);
    const mem = defNum(m.mem_avg, 0.0);
    setHtml('sys', `Health: <span style=\"padding:2px 6px;border-radius:4px;background:${defBool(h.ok)?'#d4f8d4':'#ffd7d7'};color:${defBool(h.ok)?'#0a4':'#900'}\">${defBool(h.ok)?'Healthy':'Unhealthy'}</span><br/>FPS: ${formatTelemetry(fps)}<br/>Quality: ${formatTelemetry(quality)}<br/>CPU avg: ${formatTelemetry(cpu)}%<br/>MEM avg: ${formatTelemetry(mem)}%<br/>Audio Q: ${defNum(m.audio_queue_size,0)}`);
    setHtml('cp', `Persist: ${defBool(m.clear_path_persist,false)}<br/>Corridor frac: ${formatTelemetry(defNum(m.clear_path_corridor_frac,0))}`);

    // Event type and importance counters
    renderKeyValueList('event_types', safeObj(m.event_type_counts));
    renderKeyValueList('importance', safeObj(m.importance_counts));

    // Queue latency histograms
    const queueLatencies = safeObj(m.queue_latencies);
    let queueHtml = '';
    for (const [queueName, latencies] of Object.entries(queueLatencies)){
      const arr = safeArr(latencies);
      if (arr.length > 0) {
        const avg = arr.reduce((a,b) => a+b, 0) / arr.length;
        queueHtml += `<div><strong>${queueName.replace('_', ' ')}</strong> (avg: ${formatTelemetry(avg)}ms)</div>`;
        queueHtml += `<div id=\"hist_${queueName}\" class=\"histogram\"></div>`;
      }
    }
    setHtml('queue_latencies', queueHtml || '<div class=\"small\">No queue data</div>');
    // Render histograms
    for (const [queueName, latencies] of Object.entries(queueLatencies)) {
      const arr = safeArr(latencies);
      if (arr.length > 0) {
        histogram(`hist_${queueName}`, arr, 10);
      }
    }
    
    // Top classes with trendlines
    const pcs = safeObj(m.per_class);
    const pcHistory = safeObj(m.per_class_history);
    const topClasses = Object.entries(pcs)
      .sort((a,b)=> (defNum(b[1]?.rate_60s,0))-(defNum(a[1]?.rate_60s,0)))
      .slice(0,5);
    let topClassesHtml = '';
    for (const [className, data] of topClasses){
      const id = sanitizeId(className);
      const dTotal = defNum(data.total, 0);
      const dRate = defNum(data.rate_60s, 0);
      topClassesHtml += `<div class=\"metric-row\">\n        <div><strong>${className}</strong><br/>\n        <span class=\"small\">total: ${dTotal}, rate: ${formatTelemetry(dRate)}/s</span></div>\n        <div class=\"trendline\" id=\"trend_${id}\"></div>\n      </div>`;
    }
    setHtml('top_classes', topClassesHtml || '<div class=\"small\">No class data</div>');
    // Render trendlines
    for (const [className] of topClasses){
      const id = sanitizeId(className);
      const history = safeArr(pcHistory[className]);
      if (history.length > 0) {
        lineChart(`trend_${id}`, history, {color:'#2a8', width:80, height:25});
      }
    }
    // Recent events and verbose sample
    const evs = safeArr(m.events_recent);
    const sample = safeArr(m.events_sample);
    const recentHtml = evs.slice(-10).map(e=>formatEvent(e)).join('');
    const sampleHtml = sample.length
      ? `<div style=\"margin-top:8px;\"><strong>Sample last ${Math.min(sample.length,50)} events</strong><br/>` + sample.slice(-50).map(s=>formatEvent(s)).join('') + '</div>'
      : '';
    setHtml('ev', (recentHtml || '<div>No events yet.</div>') + sampleHtml);

    showStatus('ok', 'Live');
    retryMs = 1000;
    scheduleNext(1000);
  }catch(e){
    console.error('Dashboard refresh error', e);
    showStatus('error', `Error refreshing: ${e && e.message ? e.message : e}`);
    retryMs = Math.min(retryMs * 2, maxRetryMs);
    scheduleNext(retryMs);
  }
}
refresh();
</script>
</body></html>"""
                    self._html(html)
                except Exception:
                    self._bad('dashboard error')
            elif path == '/feed':
                # Return recent announcement feed entries for client-side speech
                try:
                    try:
                        since_str = (qs.get('since', [''])[0] or '').strip()
                        since = float(since_str) if since_str else 0.0
                    except Exception:
                        since = 0.0
                    try:
                        limit_str = (qs.get('limit', [''])[0] or '').strip()
                        limit = int(limit_str) if limit_str else 20
                    except Exception:
                        limit = 20
                    limit = max(1, min(limit, 100))
                    with state.lock:
                        feed = list(getattr(state, '_announce_feed', []) or [])
                    if since > 0.0:
                        feed = [e for e in feed if float(e.get('time', 0.0) or 0.0) > since]
                    if len(feed) > limit:
                        feed = feed[-limit:]
                    return self._json({'items': feed, 'count': len(feed)})
                except Exception:
                    self._bad('feed error')
            elif path == '/config/alerts':
                # Return stored alert preferences and resolved active preferences
                try:
                    # Optionally include raw stored prefs
                    include_stored = (qs.get('stored', ['1'])[0] or '1').strip().lower() in ('1','true','yes')
                    # Resolve active preferences for current position
                    try:
                        active = _get_active_alert_prefs()
                    except Exception:
                        active = {}
                    body = {'active': active}
                    if include_stored:
                        try:
                            body['stored'] = _read_alert_prefs()
                        except Exception:
                            body['stored'] = {}
                    return self._json(body)
                except Exception:
                    return self._bad('alerts error')
            else:
                self._bad('unknown endpoint')
        except Exception:
            self._bad('error')

    def log_message(self, format, *args):
        # Silence default HTTP server logs
        return

    def _unauthorized(self, msg='unauthorized'):
        self.send_response(401)
        self._apply_security_headers()
        self.send_header('Content-Type', 'text/plain')
        # Hint acceptable auth scheme to clients
        self.send_header('WWW-Authenticate', 'Bearer')
        self.end_headers()
        try:
            hint = ''
            try:
                require = bool(getattr(state, '_shortcuts_require_token', False))
            except Exception:
                require = False
            if require:
                hint = ' (use Authorization: Bearer <token>; set THIRD_EYE_TOKEN)'
            self.wfile.write((str(msg) + hint).encode('utf-8'))
        except Exception:
            pass

    def _is_authorized(self, qs=None):
        try:
            require = bool(getattr(state, '_shortcuts_require_token', False))
        except Exception:
            require = False
        if not require:
            return True
        try:
            expected = str(getattr(state, '_shortcuts_token', '') or '')
        except Exception:
            expected = ''
        try:
            prod_gate = bool(getattr(state, '_shortcuts_prod_gate', False))
        except Exception:
            prod_gate = False
        current_secret = str(getattr(state, '_shortcuts_token_secret_current', '') or '')
        next_secret = str(getattr(state, '_shortcuts_token_secret_next', '') or '')
        # If no credentials configured at all, deny
        if (expected == '') and (current_secret == '') and (next_secret == ''):
            return False

        # Prefer Authorization: Bearer
        try:
            auth = str(self.headers.get('Authorization', '') or '')
            if auth.startswith('Bearer '):
                bearer = auth[len('Bearer '):].strip()
                # Accept exact match to static expected token
                if expected and (bearer == expected):
                    return True
                # Attempt JWT HS256 verification with rotation secrets
                try:
                    import base64, hmac, hashlib, json, time
                    parts = bearer.split('.')
                    if len(parts) == 3:
                        h_b64, p_b64, s_b64 = parts
                        def _b64url_decode(s: str) -> bytes:
                            s = s + '=' * (-len(s) % 4)
                            return base64.urlsafe_b64decode(s.encode('ascii'))
                        signed_input = f"{h_b64}.{p_b64}".encode('ascii')
                        payload = json.loads(_b64url_decode(p_b64).decode('utf-8'))
                        exp = float(payload.get('exp', 0) or 0)
                        if exp <= 0 or time.time() >= exp:
                            return False
                        sig = _b64url_decode(s_b64)
                        for secret in (current_secret, next_secret):
                            if not secret:
                                continue
                            calc = hmac.new(secret.encode('utf-8'), signed_input, hashlib.sha256).digest()
                            if hmac.compare_digest(calc, sig):
                                return True
                except Exception:
                    pass
            else:
                # Non-Bearer methods: allowed only when not in production gate
                if not prod_gate:
                    # Token scheme
                    if auth.startswith('Token '):
                        if expected and (auth[len('Token '):].strip() == expected):
                            return True
                    # Basic with password==token
                    if auth.startswith('Basic '):
                        import base64
                        try:
                            creds = base64.b64decode(auth.split(' ', 1)[1]).decode('utf-8')
                            if ':' in creds:
                                pwd = creds.split(':', 1)[1]
                                if expected and (pwd == expected):
                                    return True
                        except Exception:
                            pass
        except Exception:
            pass
        # Custom header allowed only outside production gate
        try:
            if not prod_gate:
                xtok = str(self.headers.get('X-ThirdEye-Token', '') or '')
                if expected and (xtok == expected):
                    return True
        except Exception:
            pass
        # Query string param allowed only outside production gate
        try:
            if (not prod_gate) and (qs is not None):
                qtok = str((qs.get('token', [''])[0] or ''))
                if expected and (qtok == expected):
                    return True
        except Exception:
            pass
        return False

    def _ok(self, msg):
        self.send_response(200)
        self._apply_security_headers()
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        try:
            self.wfile.write(msg.encode('utf-8'))
        except Exception:
            pass

    def _bad(self, msg):
        self.send_response(400)
        self._apply_security_headers()
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        try:
            self.wfile.write(msg.encode('utf-8'))
        except Exception:
            pass

    def _json(self, data):
        try:
            payload = json.dumps(data).encode('utf-8')
        except Exception:
            payload = b'{}'
        self.send_response(200)
        self._apply_security_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        try:
            self.wfile.write(payload)
        except Exception:
            pass

    def _html(self, html):
        self.send_response(200)
        self._apply_security_headers()
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        try:
            self.wfile.write(html.encode('utf-8'))
        except Exception:
            pass

    def do_OPTIONS(self):
        try:
            self.send_response(204)
            self._apply_security_headers()
            self.end_headers()
        except Exception:
            pass

    def do_POST(self):
        try:
            url = urlparse(self.path)
            ip = self.client_address[0]
            if not self._rate_limit_ok(url.path, ip):
                return self._too_many('rate limited')
            if not self._is_authorized():
                return self._unauthorized()
            # Read payload
            try:
                length = int(self.headers.get('Content-Length', '0') or '0')
            except Exception:
                length = 0
            try:
                raw = self.rfile.read(length) if length > 0 else b''
            except Exception:
                raw = b''
            ctype = str(self.headers.get('Content-Type', 'application/json') or 'application/json').split(';')[0].strip()
            data = {}
            if ctype == 'application/json':
                try:
                    data = json.loads(raw.decode('utf-8')) if raw else {}
                except Exception:
                    data = {}
            # Endpoint: user feedback reports
            if url.path == '/feedback':
                try:
                    now = time.time()
                    # Normalize fields
                    ftype = str(data.get('type', 'general') or 'general').lower()
                    label = str(data.get('label', '') or '')
                    comment = str(data.get('comment', '') or '')
                    severity = float(data.get('severity', 0.0) or 0.0)
                    side = str(data.get('side', '') or '').lower()
                    dist = float(data.get('distance_m', 0.0) or 0.0)
                    context = str(data.get('context', '') or '').lower()
                    action = str(data.get('action', '') or '')
                    entry = {
                        'ts': now,
                        'type': ftype,
                        'label': label,
                        'comment': comment,
                        'severity': severity,
                        'side': side,
                        'distance_m': dist,
                        'context': context,
                        'action': action,
                        'ip': ip
                    }
                    # Append to feedback log
                    try:
                        os.makedirs('logs', exist_ok=True)
                        with open(os.path.join('logs', 'feedback.jsonl'), 'a') as f:
                            f.write(json.dumps(entry) + "\n")
                    except Exception:
                        pass
                    # Telemetry counters
                    try:
                        with state.lock:
                            state.telemetry['feedback_ts'].append(now)
                            ft = state.telemetry.get('feedback_type_counts', {})
                            dq = ft.get(ftype)
                            if not isinstance(dq, deque):
                                dq = deque(maxlen=5000)
                                ft[ftype] = dq
                            dq.append(now)
                            state.telemetry['feedback_type_counts'] = ft
                    except Exception:
                        pass
                    # Optional adaptive tuning: mark nav stats on hazard feedback
                    try:
                        if ftype in ('recognition_inaccuracy','hazard_missed','false_positive'):
                            nav = state.nav_weight_stats
                            decay = float(config.get('blind_navigation', {}).get('tuning_window_s', 300.0))
                            # Increment appropriate counters for analysis
                            if ftype == 'hazard_missed':
                                nav['path_obs_count'] = nav.get('path_obs_count', 0.0) + 1.0
                            elif ftype == 'false_positive':
                                nav['overhead_obs_count'] = nav.get('overhead_obs_count', 0.0) + 1.0
                            elif ftype == 'recognition_inaccuracy':
                                nav['trip_obs_count'] = nav.get('trip_obs_count', 0.0) + 1.0
                    except Exception:
                        pass
                    return self._ok('ok')
                except Exception:
                    return self._bad('error')
            # Endpoint: push external depth (and optional RGB)
            if url.path == '/depth/push':
                try:
                    now = time.time()
                    # Decode depth from PNG/JPEG base64
                    depth_b64 = data.get('depth_png_b64') or data.get('depth_b64')
                    if not depth_b64:
                        return self._bad('missing depth_png_b64')
                    try:
                        buf = np.frombuffer(base64.b64decode(depth_b64), dtype=np.uint8)
                        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                    except Exception:
                        img = None
                    if img is None:
                        return self._bad('invalid depth image')
                    # Convert to meters according to config and input hints
                    try:
                        units_hint = str(data.get('units', '') or '').lower()
                    except Exception:
                        units_hint = ''
                    try:
                        net_units = str(config.get('depth', {}).get('network_units', 'meters')).lower()
                        net_scale = float(config.get('depth', {}).get('network_scale', 1.0))
                    except Exception:
                        net_units = 'meters'
                        net_scale = 1.0
                    depth_m = None
                    try:
                        if img.dtype == np.uint16:
                            # Assume millimeters unless units hint/config say meters
                            u = (units_hint or net_units or 'millimeters')
                            if 'meter' in u:
                                depth_m = img.astype(np.float32) * (1.0/1.0) * float(net_scale)
                            else:
                                depth_m = img.astype(np.float32) * (1.0/1000.0) * float(net_scale)
                        elif img.dtype == np.float32 or img.dtype == np.float64:
                            depth_m = img.astype(np.float32) * float(net_scale)
                        else:
                            # 8-bit image likely encoded depth; apply scale, assume meters 0-255 -> 0-10m when scale set
                            depth_m = img.astype(np.float32) * float(net_scale)
                    except Exception:
                        depth_m = None
                    if depth_m is None:
                        return self._bad('unsupported depth format')
                    # Optional RGB frame
                    rgb_b64 = data.get('rgb_jpeg_b64') or data.get('rgb_png_b64') or None
                    rgb = None
                    if rgb_b64:
                        try:
                            rbuf = np.frombuffer(base64.b64decode(rgb_b64), dtype=np.uint8)
                            rgb = cv2.imdecode(rbuf, cv2.IMREAD_COLOR)
                        except Exception:
                            rgb = None
                    with state.lock:
                        state._external_depth_m = depth_m
                        state._external_depth_time = now
                        if rgb is not None:
                            state._external_rgb = rgb
                            state._external_rgb_time = now
                        state._use_network_depth = True
                    return self._ok('ok')
                except Exception:
                    return self._bad('error')
            # Endpoint: push external RGB only
            elif url.path == '/frame/push':
                try:
                    now = time.time()
                    rgb_b64 = data.get('rgb_jpeg_b64') or data.get('rgb_png_b64') or None
                    if not rgb_b64:
                        return self._bad('missing rgb_jpeg_b64')
                    try:
                        rbuf = np.frombuffer(base64.b64decode(rgb_b64), dtype=np.uint8)
                        rgb = cv2.imdecode(rbuf, cv2.IMREAD_COLOR)
                    except Exception:
                        rgb = None
                    if rgb is None:
                        return self._bad('invalid rgb image')
                    with state.lock:
                        state._external_rgb = rgb
                        state._external_rgb_time = now
                    return self._ok('ok')
                except Exception:
                    return self._bad('error')
            # Endpoint: update announcements personalization
            elif url.path == '/config/announcements':
                try:
                    ann = config.setdefault('announcements', {})
                    tone = str(data.get('tone_style', ann.get('tone_style', 'supportive')) or 'supportive')
                    verbosity = str(data.get('verbosity_level', ann.get('verbosity_level', 'standard')) or 'standard')
                    if tone.lower() in ('neutral','supportive'):
                        ann['tone_style'] = tone.lower()
                    if verbosity.lower() in ('brief','standard','detailed'):
                        ann['verbosity_level'] = verbosity.lower()
                    # Optional: update summary interval
                    if 'summary_interval_s' in data:
                        try:
                            ann['summary_interval_s'] = float(data.get('summary_interval_s'))
                        except Exception:
                            pass
                    # Optional: adaptive cooldown toggle
                    if 'adaptive_cooldown' in data:
                        ann['adaptive_cooldown'] = bool(data.get('adaptive_cooldown'))
                    # Persist preferences
                    try:
                        os.makedirs('logs', exist_ok=True)
                        prefs_path = os.path.join('logs', 'prefs.json')
                        pref = {'tone_style': ann.get('tone_style'), 'verbosity_level': ann.get('verbosity_level')}
                        if os.path.exists(prefs_path):
                            with open(prefs_path, 'r') as f:
                                prev = json.load(f)
                                if isinstance(prev, dict):
                                    prev.update(pref)
                                    pref = prev
                        with open(prefs_path, 'w') as f:
                            json.dump(pref, f)
                    except Exception:
                        pass
                    speak_status(f"Settings updated: tone {ann.get('tone_style')}, verbosity {ann.get('verbosity_level')}")
                    return self._ok(json.dumps({'tone_style': ann.get('tone_style'), 'verbosity_level': ann.get('verbosity_level'), 'summary_interval_s': ann.get('summary_interval_s'), 'adaptive_cooldown': ann.get('adaptive_cooldown')}))
                except Exception as e:
                    return self._bad(f'error: {e}')
            # Endpoint: set TTS voice id for external provider
            elif url.path == '/audio/voice':
                try:
                    aud = config.setdefault('audio', {})
                    voice_id = str(data.get('id', data.get('voice_id', aud.get('voice_id', ''))) or '')
                    aud['voice_id'] = voice_id
                    # Persist preference
                    try:
                        os.makedirs('logs', exist_ok=True)
                        prefs_path = os.path.join('logs', 'prefs.json')
                        pref = {'voice_id': voice_id}
                        if os.path.exists(prefs_path):
                            with open(prefs_path, 'r') as f:
                                prev = json.load(f)
                                if isinstance(prev, dict):
                                    prev.update(pref)
                                    pref = prev
                        with open(prefs_path, 'w') as f:
                            json.dump(pref, f)
                    except Exception:
                        pass
                    speak_status('Voice updated.')
                    return self._ok(json.dumps({'voice_id': voice_id}))
                except Exception as e:
                    return self._bad(f'error: {e}')
            # Endpoint: ingest local crowd-sourced hints
            elif url.path == '/hints/push':
                try:
                    # Accept either a single hint or an array under 'hints'
                    items = []
                    obj = data.get('hint')
                    arr = data.get('hints')
                    if isinstance(arr, list):
                        items = [it for it in arr if isinstance(it, dict)]
                    elif isinstance(obj, dict):
                        items = [obj]
                    else:
                        # Flatten any dicts directly under payload
                        if isinstance(data, dict):
                            maybe = data.get('items') or data.get('data') or []
                            if isinstance(maybe, list):
                                items = [it for it in maybe if isinstance(it, dict)]
                    if not items:
                        return self._bad('missing hints')
                    # Validate and normalize
                    out = []
                    for h in items:
                        try:
                            typ = str(h.get('type', '') or '').strip().lower()
                            if typ not in ('crosswalk','intersection'):
                                continue
                            lat = float(h.get('lat'))
                            lon = float(h.get('lon'))
                            entry = {
                                'type': typ,
                                'lat': lat,
                                'lon': lon,
                                'source': str(h.get('source','local') or 'local'),
                                'created_ts': float(h.get('created_ts', time.time()) or time.time()),
                            }
                            # Optional: expires_ts or max_age_s hint
                            if 'expires_ts' in h:
                                try:
                                    entry['expires_ts'] = float(h.get('expires_ts'))
                                except Exception:
                                    pass
                            out.append(entry)
                        except Exception:
                            continue
                    if not out:
                        return self._bad('no valid hints')
                    # Persist each push as its own JSON file for simpler rotation
                    try:
                        os.makedirs(os.path.join('cache','hints'), exist_ok=True)
                    except Exception:
                        pass
                    fname = os.path.join('cache','hints', f"local_{int(time.time())}_{random.randint(1000,9999)}.json")
                    try:
                        with open(fname, 'w') as f:
                            json.dump(out, f)
                    except Exception:
                        return self._bad('persist error')
                    # Optional telemetry for admin visibility
                    try:
                        with state.lock:
                            tel = state.telemetry
                            dq = tel.get('local_hints_push_ts')
                            if not isinstance(dq, deque):
                                dq = deque(maxlen=1000)
                                tel['local_hints_push_ts'] = dq
                            dq.append(time.time())
                    except Exception:
                        pass
                    try:
                        speak_status('Hints received')
                    except Exception:
                        pass
                    return self._ok(json.dumps({'stored': len(out)}))
                except Exception:
                    return self._bad('error')
            # Endpoint: alert preferences (global and geofenced)
            elif url.path == '/config/alerts':
                try:
                    # Load current stored preferences
                    try:
                        prefs = _read_alert_prefs()
                    except Exception:
                        prefs = {}
                    if not isinstance(prefs, dict):
                        prefs = {}
                    # Upsert global prefs
                    g = data.get('global')
                    if isinstance(g, dict):
                        cur = prefs.get('global', {}) if isinstance(prefs.get('global'), dict) else {}
                        upd = {}
                        if 'mute_all' in g:
                            try:
                                upd['mute_all'] = bool(g.get('mute_all'))
                            except Exception:
                                pass
                        if 'urgency' in g and isinstance(g.get('urgency'), list):
                            try:
                                upd['urgency'] = [str(x).lower() for x in g.get('urgency') if str(x).lower() in ('critical','warning','info')]
                            except Exception:
                                pass
                        if 'types' in g and isinstance(g.get('types'), list):
                            try:
                                # Limit to known types; accept unknowns as strings
                                upd['types'] = [str(x).lower() for x in g.get('types')]
                            except Exception:
                                pass
                        if 'language' in g and isinstance(g.get('language'), dict):
                            lang = {}
                            try:
                                code = str(g['language'].get('code', '') or '')
                                if code:
                                    lang['code'] = code
                            except Exception:
                                pass
                            try:
                                voice = str(g['language'].get('voice', '') or '')
                                if voice:
                                    lang['voice'] = voice
                            except Exception:
                                pass
                            if lang:
                                upd['language'] = lang
                        if 'sensitivity' in g:
                            try:
                                val = float(g.get('sensitivity'))
                                upd['sensitivity'] = float(max(0.2, min(val, 2.0)))
                            except Exception:
                                pass
                        if 'adaptive' in g:
                            try:
                                upd['adaptive'] = bool(g.get('adaptive'))
                            except Exception:
                                pass
                        cur.update(upd)
                        prefs['global'] = cur
                    # Geofence operations: upsert/replace/remove
                    mode = str(data.get('mode', 'upsert') or 'upsert').strip().lower()
                    # Delete by name shortcut
                    del_name = str(data.get('delete_name', data.get('remove_name', '')) or '').strip()
                    if del_name:
                        try:
                            arr = prefs.get('geofences', [])
                            if isinstance(arr, list):
                                prefs['geofences'] = [gf for gf in arr if str(gf.get('name','')) != del_name]
                        except Exception:
                            pass
                    # Upsert from provided geofence or geofences
                    items = []
                    if isinstance(data.get('geofence'), dict):
                        items = [data.get('geofence')]
                    elif isinstance(data.get('geofences'), list):
                        items = [x for x in data.get('geofences') if isinstance(x, dict)]
                    # Normalize and validate items
                    norm = []
                    for it in items:
                        try:
                            name = str(it.get('name', '') or '').strip()
                            center = it.get('center', {}) or {}
                            lat = float(center.get('lat'))
                            lon = float(center.get('lon'))
                            radius_m = float(it.get('radius_m', 150.0))
                            radius_m = float(max(10.0, min(radius_m, 2000.0)))
                            gprefs = it.get('prefs', {}) or {}
                            entry = {
                                'name': name or f"zone_{int(time.time())}",
                                'center': {'lat': lat, 'lon': lon},
                                'radius_m': radius_m,
                                'prefs': {}
                            }
                            # Accept same keys as global
                            if isinstance(gprefs, dict):
                                if 'mute_all' in gprefs:
                                    try:
                                        entry['prefs']['mute_all'] = bool(gprefs.get('mute_all'))
                                    except Exception:
                                        pass
                                if 'urgency' in gprefs and isinstance(gprefs.get('urgency'), list):
                                    try:
                                        entry['prefs']['urgency'] = [str(x).lower() for x in gprefs.get('urgency') if str(x).lower() in ('critical','warning','info')]
                                    except Exception:
                                        pass
                                if 'types' in gprefs and isinstance(gprefs.get('types'), list):
                                    try:
                                        entry['prefs']['types'] = [str(x).lower() for x in gprefs.get('types')]
                                    except Exception:
                                        pass
                                if 'language' in gprefs and isinstance(gprefs.get('language'), dict):
                                    lang = {}
                                    try:
                                        code = str(gprefs['language'].get('code', '') or '')
                                        if code:
                                            lang['code'] = code
                                    except Exception:
                                        pass
                                    try:
                                        voice = str(gprefs['language'].get('voice', '') or '')
                                        if voice:
                                            lang['voice'] = voice
                                    except Exception:
                                        pass
                                    if lang:
                                        entry['prefs']['language'] = lang
                                if 'sensitivity' in gprefs:
                                    try:
                                        val = float(gprefs.get('sensitivity'))
                                        entry['prefs']['sensitivity'] = float(max(0.2, min(val, 2.0)))
                                    except Exception:
                                        pass
                                if 'adaptive' in gprefs:
                                    try:
                                        entry['prefs']['adaptive'] = bool(gprefs.get('adaptive'))
                                    except Exception:
                                        pass
                            norm.append(entry)
                        except Exception:
                            continue
                    # Apply mode
                    if norm:
                        if mode == 'replace':
                            prefs['geofences'] = norm
                        else:
                            arr = prefs.get('geofences', [])
                            if not isinstance(arr, list):
                                arr = []
                            # Upsert by name
                            by_name = {str(x.get('name','')): x for x in arr}
                            for entry in norm:
                                by_name[entry['name']] = entry
                            prefs['geofences'] = list(by_name.values())
                    # Persist
                    try:
                        _write_alert_prefs(prefs)
                    except Exception:
                        pass
                    # Compute active to return
                    try:
                        active = _get_active_alert_prefs()
                    except Exception:
                        active = {}
                    try:
                        speak_status('Alert preferences updated')
                    except Exception:
                        pass
                    return self._json({'stored': prefs, 'active': active})
                except Exception as e:
                    return self._bad(f'error: {e}')
            else:
                return self._bad('unknown endpoint')
        except Exception:
            return self._bad('error')

    def _apply_security_headers(self):
        try:
            origin = self.headers.get('Origin')
            allow_origin = 'null'
            cors_enabled = bool(getattr(state, '_shortcuts_cors_enabled', True))
            cors_allowlist = getattr(state, '_shortcuts_cors_allowlist', ['http://localhost', 'http://127.0.0.1'])
            if cors_enabled and origin and any(str(origin).startswith(x) for x in cors_allowlist):
                allow_origin = origin
            self.send_header('Access-Control-Allow-Origin', allow_origin)
            self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Authorization, Content-Type, X-ThirdEye-Token')
            self.send_header('Access-Control-Max-Age', '600')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            # Privacy: prevent caching of sensitive responses
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Pragma', 'no-cache')
            # CSP to constrain dashboard assets
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
            # Disable powerful browser features
            self.send_header('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
            self.send_header('X-Permitted-Cross-Domain-Policies', 'none')
            # Add HSTS if HTTPS is enabled (behind proxy or configured)
            try:
                proto = str(self.headers.get('X-Forwarded-Proto', '') or '').lower()
            except Exception:
                proto = ''
            https_enabled = bool(getattr(state, '_shortcuts_https_enabled', False))
            if proto == 'https' or https_enabled:
                self.send_header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
        except Exception:
            pass

    def _too_many(self, msg='too many requests'):
        try:
            self.send_response(429)
            self._apply_security_headers()
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(str(msg).encode('utf-8'))
        except Exception:
            pass

    def _rate_limit_ok(self, path, ip):
        try:
            limits = getattr(state, '_rate_limits', None)
            if limits is None:
                limits = {
                    'window': 10.0,
                    'limits': {
                        '/metrics': {'limit': 10},
                        '/dashboard': {'limit': 5},
                    },
                    'store': {},
                }
                setattr(state, '_rate_limits', limits)
            window = float(limits.get('window', 10.0))
            rule = limits.get('limits', {}).get(path, {'limit': 20})
            limit = int(rule.get('limit', 20))
            now = time.time()
            # Apply backoff penalty for repeat offenders
            penalties = limits.setdefault('penalties', {})
            pen_key = (ip, path)
            pen = penalties.get(pen_key)
            if pen and now < float(pen.get('until', 0.0)):
                limit = max(3, int(limit // 3))
            store = limits['store'].setdefault(ip, {})
            bucket = store.get(path)
            if not bucket or (now - bucket.get('start', 0.0)) > window:
                store[path] = {'start': now, 'count': 1}
                return True
            bucket['count'] = int(bucket.get('count', 0)) + 1
            if bucket['count'] <= limit:
                return True
            # Set a short penalty window on limit breach
            penalties[pen_key] = {'until': now + 60.0}
            return False
        except Exception:
            return True


def shortcuts_server_thread():
    # Start a simple HTTP server for Apple Shortcuts integration
    try:
        port = int(config.get('announcements', {}).get('shortcuts_port', 8765))
    except Exception:
        port = 8765
    # Determine bind address and token requirement
    try:
        bind_addr = str(config.get('announcements', {}).get('shortcuts_bind', '127.0.0.1') or '127.0.0.1')
    except Exception:
        bind_addr = '127.0.0.1'
    try:
        token = str(os.environ.get('THIRD_EYE_TOKEN', config.get('announcements', {}).get('shortcuts_token', '')) or '')
    except Exception:
        token = ''
    try:
        require_token = bind_addr not in ('127.0.0.1', '::1', 'localhost')
    except Exception:
        require_token = False
    # Allow explicit override to require authentication even on loopback
    try:
        force_req_env = str(os.environ.get('THIRD_EYE_REQUIRE_TOKEN', '') or '')
        if force_req_env.strip().lower() in ('1','true','yes','on'):
            require_token = True
    except Exception:
        pass
    try:
        with state.lock:
            state._shortcuts_require_token = bool(require_token)
            state._shortcuts_token = token
            # JWT HMAC secrets for Bearer token verification and rotation
            try:
                state._shortcuts_token_secret_current = str(os.environ.get('THIRD_EYE_TOKEN_SECRET', '') or '')
                state._shortcuts_token_secret_next = str(os.environ.get('THIRD_EYE_TOKEN_SECRET_NEXT', '') or '')
            except Exception:
                state._shortcuts_token_secret_current = ''
                state._shortcuts_token_secret_next = ''
            # Production gating: require auth for sensitive endpoints when enabled
            try:
                prod_env = os.environ.get('THIRD_EYE_SHORTCUTS_PROD', '')
                prod_gate = bool(prod_env) and prod_env.strip() not in ('0','false','False')
            except Exception:
                prod_gate = False
            state._shortcuts_prod_gate = bool(prod_gate)
            # HTTPS indicator (for HSTS behind proxy)
            try:
                https_env = os.environ.get('THIRD_EYE_SHORTCUTS_HTTPS', '')
                state._shortcuts_https_enabled = https_env.strip() not in ('', '0', 'false', 'False')
            except Exception:
                state._shortcuts_https_enabled = False
            # CORS policy
            try:
                cors_enabled_env = os.environ.get('THIRD_EYE_CORS_ENABLED', '1')
                state._shortcuts_cors_enabled = cors_enabled_env.strip() not in ('0','false','False')
            except Exception:
                state._shortcuts_cors_enabled = True
            try:
                origins_env = os.environ.get('THIRD_EYE_SHORTCUTS_CORS_ORIGINS', '')
                if origins_env:
                    allowlist = [o.strip() for o in origins_env.split(',') if o.strip()]
                else:
                    allowlist = [
                        'http://localhost',
                        'http://127.0.0.1',
                        f'http://{bind_addr}:{port}',
                    ]
                state._shortcuts_cors_allowlist = allowlist
            except Exception:
                state._shortcuts_cors_allowlist = ['http://localhost','http://127.0.0.1']
            # Rate limits
            try:
                rl_metrics = int(os.environ.get('THIRD_EYE_RATE_LIMIT_METRICS', '10'))
            except Exception:
                rl_metrics = 10
            try:
                rl_dashboard = int(os.environ.get('THIRD_EYE_RATE_LIMIT_DASHBOARD', '5'))
            except Exception:
                rl_dashboard = 5
            try:
                rl_voice = int(os.environ.get('THIRD_EYE_RATE_LIMIT_VOICE', '8'))
            except Exception:
                rl_voice = 8
            try:
                rl_push_depth = int(os.environ.get('THIRD_EYE_RATE_LIMIT_PUSH_DEPTH', '30'))
            except Exception:
                rl_push_depth = 30
            try:
                rl_push_frame = int(os.environ.get('THIRD_EYE_RATE_LIMIT_PUSH_FRAME', '60'))
            except Exception:
                rl_push_frame = 60
            try:
                rl_feedback = int(os.environ.get('THIRD_EYE_RATE_LIMIT_FEEDBACK', '30'))
            except Exception:
                rl_feedback = 30
            state._rate_limits = {
                'window': 10.0,
                'limits': {
                    '/metrics': {'limit': rl_metrics},
                    '/dashboard': {'limit': rl_dashboard},
                    '/display.jpg': {'limit': int(os.environ.get('THIRD_EYE_RATE_LIMIT_DISPLAY', '8'))},
                    '/preview': {'limit': int(os.environ.get('THIRD_EYE_RATE_LIMIT_PREVIEW', '8'))},
                    '/voice': {'limit': rl_voice},
                    '/depth/push': {'limit': rl_push_depth},
                    '/frame/push': {'limit': rl_push_frame},
                    '/feedback': {'limit': rl_feedback},
                    '/config/announcements': {'limit': 20},
                    '/config/alerts': {'limit': 20},
                    '/audio/voice': {'limit': 20},
                    '/hints/push': {'limit': 30},
                    '/ota': {'limit': 10},
                },
                'store': {},
                'penalties': {},
            }
    except Exception:
        pass
    try:
        # Allow fast restarts without TIME_WAIT bind errors
        try:
            from http.server import HTTPServer as _HTTPServerClass
            _HTTPServerClass.allow_reuse_address = True
        except Exception:
            pass
        try:
            logging.info(f"Attempting Shortcuts bind on {bind_addr}:{port}")
        except Exception:
            pass
        server = HTTPServer((bind_addr, port), _ShortcutsHandler)
        # Optional TLS enablement for secure transport
        try:
            cert_path = str(os.environ.get('THIRD_EYE_TLS_CERT', config.get('announcements', {}).get('shortcuts_tls_cert', '')) or '')
            key_path = str(os.environ.get('THIRD_EYE_TLS_KEY', config.get('announcements', {}).get('shortcuts_tls_key', '')) or '')
            if cert_path and key_path and os.path.exists(cert_path) and os.path.exists(key_path):
                import ssl
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                try:
                    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
                except Exception:
                    pass
                try:
                    ctx.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20')
                except Exception:
                    pass
                ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
                server.socket = ctx.wrap_socket(server.socket, server_side=True)
                try:
                    with state.lock:
                        state._shortcuts_https_enabled = True
                except Exception:
                    pass
                try:
                    logging.info('Shortcuts TLS enabled')
                except Exception:
                    pass
            else:
                # Enforce auth when serving on non-loopback without TLS
                if bind_addr not in ('127.0.0.1', '::1', 'localhost'):
                    try:
                        with state.lock:
                            state._shortcuts_require_token = True
                    except Exception:
                        pass
                    try:
                        logging.warning('Shortcuts server without TLS on non-loopback; authentication enforced.')
                    except Exception:
                        pass
        except Exception as _tls_err:
            try:
                logging.warning(f'TLS setup error: {_tls_err}')
            except Exception:
                pass
        # Ensure the server loop does not block indefinitely
        try:
            server.timeout = float(config.get('announcements', {}).get('shortcuts_timeout_s', 1.0))
        except Exception:
            server.timeout = 1.0
        # Initialize baseline telemetry before serving any requests
        try:
            with state.lock:
                tel = state.telemetry
                if 'start_time' not in tel:
                    tel['start_time'] = time.time()
                tel.setdefault('events', [])
                tel.setdefault('announce_ts', [])
                tel.setdefault('critical_ts', [])
                tel.setdefault('suppressed_ts', [])
                tel.setdefault('summary_ts', [])
                tel.setdefault('moderate_approach_ts', [])
                tel.setdefault('slow_approach_ts', [])
                tel.setdefault('moderate_left_ts', []),
                tel.setdefault('moderate_right_ts', []),
                tel.setdefault('slow_left_ts', []),
                tel.setdefault('slow_right_ts', []),
                tel.setdefault('approach_repeat_suppressed_ts', [])
                tel.setdefault('per_class_ts', {})
                tel.setdefault('per_class_history', {})
                tel.setdefault('event_type_counts', {})
                tel.setdefault('importance_counts', {})
                tel.setdefault('queue_latencies', {
                    'tts_queue': deque(maxlen=50),
                    'tts_speak': deque(maxlen=50),
                    'audio_queue': deque(maxlen=50)
                })
                tel.setdefault('degraded_mode', False)
                tel.setdefault('mock_sensor', bool(getattr(state, 'mock_sensor', False)))
        except Exception:
            pass
        # Gate server-bound voice announcement behind config; default off
        try:
            if bool(config.get('audio', {}).get('announce_server_bound', False)):
                speak_status(f'Shortcuts server on {bind_addr}:{port}')
        except Exception:
            pass
        try:
            logging.info(f"Shortcuts server bound: {bind_addr}:{port}")
        except Exception:
            pass
        while state.running:
            try:
                server.handle_request()
            except Exception as e:
                logging.warning(f"Shortcuts request error: {e}")
                time.sleep(0.1)
        try:
            server.server_close()
        except Exception:
            pass
    except Exception as e:
        try:
            msg = str(e)
            if 'Address already in use' in msg or 'Errno 48' in msg or 'Errno 98' in msg:
                logging.warning(f"Shortcuts server bind conflict on {bind_addr}:{port}: {e}. If another runtime is running, stop it or change 'announcements.shortcuts_port'.")
            else:
                logging.warning(f"Shortcuts server error: {e}")
        except Exception:
            logging.warning(f"Shortcuts server error: {e}")


def imu_listener_thread():
    # Optional UDP listener for IMU JSON packets from external phone apps
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        port = int(config.get('announcements', {}).get('imu_udp_port', 5391))
        try:
            sock.bind(('0.0.0.0', port))
        except OSError as e:
            # Provide a clear hint when the port is busy
            logging.warning(f"IMU UDP bind failed on {port}: {e}. If another process is using this port, stop it or set 'announcements.imu_udp_port' to a free port.")
            try:
                sock.close()
            except Exception:
                pass
            return
        sock.settimeout(1.0)
        logging.info(f"IMU UDP listener on {port}")
    except Exception as e:
        logging.warning(f"IMU UDP bind failed: {e}")
        return
    try:
        while state.running:
            try:
                data, _addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception:
                break
            try:
                payload = json.loads(data.decode('utf-8'))
                ax = float(payload.get('ax', 0.0))
                ay = float(payload.get('ay', 0.0))
                az = float(payload.get('az', 0.0))
                gx = float(payload.get('gx', 0.0))
                gy = float(payload.get('gy', 0.0))
                gz = float(payload.get('gz', 0.0))
                with state.lock:
                    state.imu['user_acceleration'] = [ax, ay, az]
                    state.imu['rotation_rate'] = [gx, gy, gz]
                    state.imu['last_update'] = time.time()
            except Exception:
                continue
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _permission_check_once():
    # Check camera liveness via last frame update and thread state
    try:
        t_cam = state.threads.get('camera')
        last = float(getattr(state, '_last_frame_time', 0.0) or 0.0)
        alive = bool(t_cam and t_cam.is_alive())
        recent = (time.time() - last) < 2.0
        cam_ok = alive and recent
    except Exception:
        cam_ok = False
    # Debounce warning announcements to avoid spamming logs/tasks
    if not cam_ok:
        now = time.time()
        try:
            start_time = float(state.telemetry.get('start_time', now) or now)
        except Exception:
            start_time = now
        uptime_s = max(0.0, now - start_time)
        # Minimum interval between permission warnings (configurable)
        min_interval = float(config.get('announcements', {}).get('permission_warn_min_interval_s', 60.0))
        last_warn = float(getattr(state, '_last_permission_warn_ts', 0.0) or 0.0)
        should_warn = (uptime_s >= 10.0) and ((now - last_warn) >= min_interval)
        if should_warn:
            speak_status('Camera not updating. Please check permission.')
            try:
                perform_haptic('warning')
            except Exception:
                pass
            try:
                setattr(state, '_last_permission_warn_ts', now)
            except Exception:
                pass
    return cam_ok


def permission_check_thread():
    # Periodically monitor camera permission/liveness and announce issues
    interval = float(config.get('announcements', {}).get('permission_check_s', 5.0))
    while state.running:
        try:
            _permission_check_once()
        except Exception:
            pass
        time.sleep(interval)


def draw_ui(frame):
    h, w = frame.shape[:2]
    disp = frame
    depth_vis = None
    depth_map_m = None
    dets = []
    with state.lock:
        if state.depth_vis is not None:
            depth_vis = state.depth_vis.copy()
        if state.depth_map_m is not None:
            depth_map_m = state.depth_map_m
        dets = list(state.detections)

    # Align depth visualization to the display frame (size, channels, dtype)
    if depth_vis is not None:
        try:
            # Resize to match frame resolution if needed
            if depth_vis.shape[:2] != (h, w):
                depth_vis = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_NEAREST)

            # Match channels: convert grayscale to BGR if mixing with a 3-channel frame
            if disp.ndim == 3 and (depth_vis.ndim == 2 or (depth_vis.ndim == 3 and depth_vis.shape[2] == 1)):
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            elif disp.ndim == 2 and depth_vis.ndim == 3 and depth_vis.shape[2] == 3:
                # If the frame is grayscale but depth_vis is BGR, convert frame to BGR for overlay consistency
                disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

            # Match dtype for cv2.addWeighted
            if depth_vis.dtype != disp.dtype:
                depth_vis = depth_vis.astype(disp.dtype)
        except Exception:
            # On any conversion failure, fall back to using the original frame without overlay
            depth_vis = None

    if state.show_depth and depth_vis is not None:
        disp = depth_vis
    elif state.show_overlay and depth_vis is not None:
        disp = cv2.addWeighted(disp, 0.7, depth_vis, 0.3, 0)

    # Limit drawing to top-K most relevant detections to reduce clutter
    if dets:
        scored = []
        for d in dets:
            depth_score = 0.0
            if d.get('depth_m') is not None:
                # nearer gets higher score
                depth_score = max(0.0, 20.0 - d['depth_m'])
            conf_score = d.get('conf', 0.0) * 10.0
            priority = state.class_prefs.get(d.get('label', ''), {'priority': 1})['priority'] * 5.0
            total = depth_score + conf_score + priority
            scored.append((total, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        dets = [d for _, d in scored[:state.top_k_draw]]

    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        color = (0, 255, 0)
        thickness = 2
        
        # Calculate speed for this object
        track_id = d.get('track_id')
        speed = 0.0
        if track_id and track_id in state.object_tracks:
            track_data = state.object_tracks[track_id]
            speed = calculate_object_speed(track_data, time.time())
        
        if d['depth_m'] is not None:
            # Travel-mode-aware thresholds for UI coloring
            mode = getattr(state, 'travel_mode', 'WALKING')
            factor = 1.0
            if str(mode).upper() in ('DRIVING', 'CYCLING'):
                factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
            critical_m = CRITICAL_DISTANCE_M * factor
            warning_m = WARNING_DISTANCE_M * factor
            if d['depth_m'] <= critical_m:
                color = (0, 0, 255)
                thickness = 3
            elif d['depth_m'] <= warning_m:
                color = (0, 165, 255)
                thickness = 2
            elif speed >= FAST_SPEED_THRESHOLD:
                color = (0, 255, 255)  # Cyan for fast objects
                thickness = 3
        
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, thickness)
        # Draw labels only when overlay text is enabled
        if bool(getattr(state, 'show_overlay', False)):
            label = f"{d['label']} {d['conf']:.2f}"
            if d['depth_m'] is not None:
                label += f" {d['depth_m']:.1f}m"
            if speed > 0.5:  # Only show speed if significant
                label += f" {speed:.1f}m/s"
            cv2.putText(disp, label, (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Performance stats - Fixed FPS calculation
    if bool(getattr(state, 'show_overlay', False)):
        fps_color = (0, 255, 0) if state.last_fps >= 20 else (0, 255, 255) if state.last_fps >= 15 else (0, 0, 255)
        cv2.putText(disp, f"FPS: {state.last_fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)
    
    # CPU and Memory usage
    if bool(getattr(state, 'show_overlay', False)) and state.performance_stats['cpu_usage']:
        avg_cpu = np.mean(state.performance_stats['cpu_usage'][-10:])
        avg_mem = np.mean(state.performance_stats['memory_usage'][-10:])
        cv2.putText(disp, f"CPU: {avg_cpu:.1f}% MEM: {avg_mem:.1f}%", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Quality level
    if bool(getattr(state, 'show_overlay', False)):
        quality = state.performance_stats['quality_level']
        quality_color = (0, 255, 0) if quality >= 0.9 else (0, 255, 255) if quality >= 0.7 else (0, 0, 255)
        cv2.putText(disp, f"Quality: {quality:.1f} Depth:[{state.min_m:.1f},{state.max_m:.1f}]", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
    
    # Processing time
    if bool(getattr(state, 'show_overlay', False)) and state.performance_stats['processing_times']:
        avg_proc_time = np.mean(state.performance_stats['processing_times'][-10:])
        cv2.putText(disp, f"Process: {avg_proc_time*1000:.1f}ms", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # System status
    if bool(getattr(state, 'show_overlay', False)):
        cv2.putText(disp, f"Announce:{'ON' if state.enable_announce else 'OFF'} AI:{'ON' if state.ai_decider else 'OFF'} Voice:{'ON' if state.enable_voice_listen else 'OFF'}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Navigation provider/mode
        cv2.putText(disp, f"Nav:{state.provider}/{state.travel_mode}", (10, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 2)
    # Travel-mode adjusted thresholds overlay and adaptive params
    try:
        mode = getattr(state, 'travel_mode', 'WALKING')
        factor = 1.0
        if str(mode).upper() in ('DRIVING', 'CYCLING'):
            factor = 1.4 if str(mode).upper() == 'DRIVING' else 1.25
        critical_m = CRITICAL_DISTANCE_M * factor
        warning_m = WARNING_DISTANCE_M * factor
        nearby_m = NEARBY_DISTANCE_M * factor
        if bool(getattr(state, 'show_overlay', False)):
            cv2.putText(disp, f"Dist: crit {critical_m:.1f}m warn {warning_m:.1f}m near {nearby_m:.1f}m", (10, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 255, 220), 2)
            cv2.putText(disp, f"YOLO:{YOLO_IMG_SIZE} det:{MAX_DET} ROI:{state.roi_ratio:.2f}", (10, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 255), 2)
            # Debug overlay for letterbox/shape mapping
            dbg = None
            try:
                with state.lock:
                    dbg = getattr(state, 'debug_meta', None)
            except Exception:
                dbg = None
            if dbg:
                try:
                    img0 = dbg.get('img0_shape')
                    img1 = dbg.get('img1_shape')
                    ratio = dbg.get('lb_ratio')
                    pad = dbg.get('lb_pad')
                    if img0 and img1:
                        cv2.putText(disp, f"img0:{int(img0[0])}x{int(img0[1])} img1:{int(img1[0])}x{int(img1[1])}", (10, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 2)
                    if pad is not None:
                        try:
                            pw = float(pad[0]) if isinstance(pad, (tuple, list)) else float(pad)
                            ph = float(pad[1]) if isinstance(pad, (tuple, list)) else 0.0
                        except Exception:
                            pw, ph = 0.0, 0.0
                        r = ratio if isinstance(ratio, (int, float)) else (ratio if ratio is not None else 1.0)
                        if isinstance(r, (tuple, list)):
                            try:
                                r = float(r[0])
                            except Exception:
                                r = 1.0
                        cv2.putText(disp, f"lb_ratio:{float(r):.3f} lb_pad:({pw:.3f},{ph:.3f})", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 2)
                except Exception:
                    pass
    except Exception:
        pass
    return disp


def watchdog_thread():
    """Monitor core threads for stalls and request controlled restarts.

    Tracks liveness timestamps (YOLO, depth, camera, announcements, audio),
    compares against configured stall thresholds, and restarts threads as
    needed with minimum intervals to avoid flapping. Emits logs and updates
    telemetry.

    Returns:
        None. Runs while `state.running` is True.
    """
    """Monitor YOLO and depth threads; restart on stall and log stall durations."""
    yolo_stall_s = float(config.get('watchdog', {}).get('yolo_stall_s', 2.5))
    depth_stall_s = float(config.get('watchdog', {}).get('depth_stall_s', 2.5))
    announce_stall_s = float(config.get('watchdog', {}).get('announce_stall_s', 3.0))
    audio_stall_s = float(config.get('watchdog', {}).get('audio_stall_s', 6.0))
    min_restart_gap = float(config.get('watchdog', {}).get('min_restart_interval_s', 5.0))
    # Periodic performance logging
    _last_perf_log = 0.0
    while state.running:
        time.sleep(0.5)
        now = time.time()
        # Log performance improvements every 5 seconds
        try:
            if (now - _last_perf_log) > float(config.get('logging', {}).get('perf_log_interval_s', 5.0)):
                log_performance_improvements()
                _last_perf_log = now
        except Exception:
            pass
        # YOLO health
        try:
            last_yolo = float(getattr(state, 'last_yolo_time', 0.0) or 0.0)
            t_yolo = state.threads.get('yolo')
            stalled_yolo = (now - last_yolo) > yolo_stall_s
            needs_restart = False
            if t_yolo is None or (t_yolo and (not t_yolo.is_alive())):
                needs_restart = True
            elif stalled_yolo and (now - state._last_restart.get('yolo', 0.0)) > min_restart_gap:
                needs_restart = True
                system_logger.info(f"YOLO stall detected: {now - last_yolo:.2f}s since last activity. Restarting thread.")
            if needs_restart and state._yolo_tuple is not None:
                # Signal thread to exit gracefully if alive
                with state.lock:
                    state.restart_request['yolo'] = True
                # Give it a moment to exit
                for _ in range(10):
                    if state.threads.get('yolo') and state.threads['yolo'].is_alive():
                        time.sleep(0.1)
                    else:
                        break
                # Clear restart flag before launching new thread to avoid race
                with state.lock:
                    state.restart_request['yolo'] = False
                # Start new YOLO thread
                yolo, device, ensemble_models = state._yolo_tuple
                new_t = threading.Thread(target=yolo_thread, args=(yolo, device, ensemble_models), daemon=True)
                new_t.start()
                with state.lock:
                    state.threads['yolo'] = new_t
                    state._last_restart['yolo'] = now
        except Exception as e:
            system_logger.info(f"Watchdog YOLO error: {e}")
        # Depth health
        try:
            last_depth = float(getattr(state, '_last_depth_time', 0.0) or 0.0)
            t_depth = state.threads.get('depth')
            stalled_depth = (now - last_depth) > depth_stall_s
            needs_restart = False
            if t_depth is None or (t_depth and (not t_depth.is_alive())):
                needs_restart = True
            elif stalled_depth and (now - state._last_restart.get('depth', 0.0)) > min_restart_gap:
                needs_restart = True
                system_logger.info(f"Depth stall detected: {now - last_depth:.2f}s since last activity. Restarting thread.")
            if needs_restart and state._depth_pipe is not None:
                # Signal thread to exit gracefully if alive
                with state.lock:
                    state.restart_request['depth'] = True
                for _ in range(10):
                    if state.threads.get('depth') and state.threads['depth'].is_alive():
                        time.sleep(0.1)
                    else:
                        break
                # Clear restart flag before launching new thread to avoid race
                with state.lock:
                    state.restart_request['depth'] = False
                # Start new Depth thread
                new_t = threading.Thread(target=depth_thread, args=(state._depth_pipe,), daemon=True)
                new_t.start()
                with state.lock:
                    state.threads['depth'] = new_t
                    state._last_restart['depth'] = now
        except Exception as e:
            system_logger.info(f"Watchdog Depth error: {e}")
        # Shortcuts server health: restart if the thread dies
        try:
            t_short = state.threads.get('shortcuts')
            if (t_short is None or (not t_short.is_alive())) and ((now - state._last_restart.get('shortcuts', 0.0)) > min_restart_gap):
                new_t = threading.Thread(target=shortcuts_server_thread, daemon=True)
                new_t.start()
                with state.lock:
                    state.threads['shortcuts'] = new_t
                    state._last_restart['shortcuts'] = now
        except Exception as e:
            system_logger.info(f"Watchdog Shortcuts error: {e}")
        # Camera health: attempt self-recovery on stall
        try:
            last_frame = float(getattr(state, '_last_frame_time', 0.0) or 0.0)
            cam_stall_s = float(config.get('watchdog', {}).get('camera_stall_s', 2.5))
            stalled_camera = (now - last_frame) > cam_stall_s
            needs_restart = False
            t_cam = state.threads.get('camera')
            if t_cam is None or (t_cam and (not t_cam.is_alive())):
                needs_restart = True
            elif stalled_camera and ((now - state._last_restart.get('camera', 0.0)) > min_restart_gap):
                needs_restart = True
                system_logger.info(f"Camera stall detected: {now - last_frame:.2f}s since last frame. Restarting thread.")
            if needs_restart:
                # Signal thread to exit gracefully if alive
                with state.lock:
                    state.restart_request['camera'] = True
                for _ in range(20):
                    if state.threads.get('camera') and state.threads['camera'].is_alive():
                        time.sleep(0.1)
                    else:
                        break
                # Clear restart flag before launching new thread to avoid race
                with state.lock:
                    state.restart_request['camera'] = False
                # Start new Camera thread
                new_t = threading.Thread(target=camera_thread, daemon=True)
                new_t.start()
                with state.lock:
                    state.threads['camera'] = new_t
                    state._last_restart['camera'] = now
        except Exception as e:
            system_logger.info(f"Watchdog Camera error: {e}")

        # Announcement thread health: restart if the thread dies or stalls
        try:
            t_ann = state.threads.get('announcement')
            last_tick = float(getattr(state, '_last_announce_tick', 0.0) or 0.0)
            stalled = (now - last_tick) > announce_stall_s
            needs_restart = False
            if t_ann is None or (t_ann and (not t_ann.is_alive())):
                needs_restart = True
            elif stalled and ((now - state._last_restart.get('announcement', 0.0)) > min_restart_gap):
                needs_restart = True
                system_logger.info(f"Announcement stall detected: {now - last_tick:.2f}s since last tick. Restarting thread.")
            if needs_restart:
                new_t = threading.Thread(target=announcement_thread, daemon=True)
                new_t.start()
                with state.lock:
                    state.threads['announcement'] = new_t
                    state._last_restart['announcement'] = now
        except Exception as e:
            system_logger.info(f"Watchdog Announcement error: {e}")

        # Audio worker health: restart if the thread dies
        try:
            async_mode = bool(config.get('audio', {}).get('async_mode', False))
        except Exception:
            async_mode = False
        try:
            if async_mode:
                # In async mode, skip thread-based restart; monitor tick only for logging
                last_tick = float(getattr(state, '_last_audio_tick', 0.0) or 0.0)
                stalled = (now - last_tick) > audio_stall_s
                if stalled:
                    system_logger.info(f"Audio async stall: {now - last_tick:.2f}s since last tick. Attempting async worker reschedule.")
                    try:
                        if (now - float(state._last_restart.get('audio_async', 0.0) or 0.0)) > min_restart_gap:
                            # Ensure queue exists
                            if getattr(state, '_async_audio_queue', None) is None and getattr(state, '_async_audio_loop', None) is not None:
                                try:
                                    fut_q = asyncio.run_coroutine_threadsafe(_make_async_audio_queue(), state._async_audio_loop)
                                    q = fut_q.result(timeout=2.0)
                                except Exception:
                                    q = None
                                with state.lock:
                                    state._async_audio_queue = q
                            # Reschedule async worker task
                            if getattr(state, '_async_audio_loop', None) is not None:
                                try:
                                    fut = asyncio.run_coroutine_threadsafe(async_audio_worker(), state._async_audio_loop)
                                    with state.lock:
                                        state.threads['audio_async_task'] = fut
                                        state._last_restart['audio_async'] = now
                                except Exception:
                                    # Fallback: start threaded worker if reschedule fails, guard against duplicates
                                    try:
                                        existing = state.threads.get('audio')
                                        if existing is None or (existing and (not existing.is_alive())):
                                            t_audio = threading.Thread(target=audio_worker, daemon=True)
                                            t_audio.start()
                                            with state.lock:
                                                state.threads['audio'] = t_audio
                                                state._last_restart['audio'] = now
                                        else:
                                            # Nudge existing worker by preempting any ongoing speech
                                            try:
                                                with state.lock:
                                                    state._preempt_now_ts = now
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                    except Exception:
                        pass
            else:
                t_audio = state.threads.get('audio')
                last_tick = float(getattr(state, '_last_audio_tick', 0.0) or 0.0)
                stalled = (now - last_tick) > audio_stall_s
                needs_restart = False
                if t_audio is None or (t_audio and (not t_audio.is_alive())):
                    needs_restart = True
                elif stalled and ((now - state._last_restart.get('audio', 0.0)) > min_restart_gap):
                    # Only restart on stall if queue has items; avoid restarts during idle
                    if not audio_queue.empty():
                        needs_restart = True
                        system_logger.info(f"Audio stall detected: {now - last_tick:.2f}s without progress and queue not empty. Restarting thread.")
                if needs_restart:
                    # Only start a new worker if none exists or it died
                    if t_audio is None or (t_audio and (not t_audio.is_alive())):
                        new_t = threading.Thread(target=audio_worker, daemon=True)
                        new_t.start()
                        with state.lock:
                            state.threads['audio'] = new_t
                            state._last_restart['audio'] = now
                    else:
                        # For stall while alive, avoid duplicate threads; preempt to unstick
                        try:
                            with state.lock:
                                state._preempt_now_ts = now
                        except Exception:
                            pass
        except Exception as e:
            system_logger.info(f"Watchdog Audio error: {e}")

        # Periodic resource sampling (GPU memory etc.)
        try:
            safe_execute('resource_sample', sample_resources, fallback=None, level='debug', rate_limit_s=2.0)
        except Exception:
            pass


def log_telemetry(event_data):
    """Log a telemetry event to the events list."""
    try:
        if not isinstance(event_data, dict):
            return
        
        # Add timestamp if not present
        if 'time' not in event_data:
            event_data['time'] = time.time()
        
        with state.lock:
            if 'events' not in state.telemetry:
                state.telemetry['events'] = []
            state.telemetry['events'].append(event_data)
            
            # Keep only recent events to prevent memory bloat
            max_events = 1000
            if len(state.telemetry['events']) > max_events:
                state.telemetry['events'] = state.telemetry['events'][-max_events:]
    except Exception:
        pass


def _snapshot_telemetry():
    """Build a compact telemetry snapshot for longitudinal analysis."""
    try:
        with state.lock:
            tel = state.telemetry
            now = time.time()
            uptime = max(0.0, now - float(tel.get('start_time', now)))
            per_class_totals = {k: len(v) for k, v in tel.get('per_class_ts', {}).items()}
            try:
                include_recent = bool(config.get('logging', {}).get('telemetry_include_recent_events', True))
            except Exception:
                include_recent = True
            events_recent = []
            if include_recent:
                events_recent = list(tel.get('events', []))[-20:]
                try:
                    redact = config.get('logging', {}).get('telemetry_redact_fields', [])
                    if isinstance(redact, list) and redact:
                        for ev in events_recent:
                            try:
                                for k in redact:
                                    if k in ev:
                                        ev.pop(k, None)
                            except Exception:
                                pass
                except Exception:
                    pass
            snap = {
                'ts': now,
                'uptime_s': uptime,
                'liveness': {
                    'last_yolo_ts': float(getattr(state, 'last_yolo_time', 0.0) or 0.0),
                    'last_depth_ts': float(getattr(state, '_last_depth_time', 0.0) or 0.0),
                    'last_frame_ts': float(getattr(state, '_last_frame_time', 0.0) or 0.0),
                    'last_success_yolo_ts': float(getattr(state, 'last_success_yolo_ts', 0.0) or 0.0),
                    'last_success_depth_ts': float(getattr(state, 'last_success_depth_ts', 0.0) or 0.0),
                },
                'counts': {
                    'announce_total': len(tel.get('announce_ts', [])),
                    'critical_total': len(tel.get('critical_ts', [])),
                    'suppressed_total': len(tel.get('suppressed_ts', [])),
                    'summary_total': len(tel.get('summary_ts', [])),
                    'moderate_total': len(tel.get('moderate_approach_ts', [])),
                    'slow_total': len(tel.get('slow_approach_ts', [])),
                    'repeat_suppressed_total': len(tel.get('approach_repeat_suppressed_ts', [])),
                    'moderate_left_total': len(tel.get('moderate_left_ts', [])),
                    'moderate_right_total': len(tel.get('moderate_right_ts', [])),
                    'slow_left_total': len(tel.get('slow_left_ts', [])),
                    'slow_right_total': len(tel.get('slow_right_ts', [])),
                },
                'per_class_totals': per_class_totals,
                'events_recent': events_recent,
            }
        return snap
    except Exception as e:
        try:
            logging.warning(f'telemetry snapshot error: {e}')
        except Exception:
            pass
        return None


def telemetry_persist_thread():
    """Periodically persist telemetry snapshots to JSONL for long-term analysis."""
    out_file = str(config.get('logging', {}).get('telemetry_file', 'logs/telemetry.jsonl'))
    try:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    except Exception:
        pass
    try:
        interval = float(config.get('logging', {}).get('telemetry_interval_s', 30.0))
    except Exception:
        interval = 30.0
    while state.running:
        try:
            enabled = bool(config.get('logging', {}).get('telemetry_enabled', True))
        except Exception:
            enabled = True
        if not enabled:
            time.sleep(interval)
            continue
        snap = _snapshot_telemetry()
        if snap is not None:
            try:
                with open(out_file, 'a') as f:
                    json.dump(snap, f)
                    f.write('\n')
            except Exception as e:
                try:
                    logging.warning(f'telemetry persist error: {e}')
                except Exception:
                    pass
        time.sleep(interval)


def cloud_sync_thread():
    """Non-blocking optional cloud sync: uploads telemetry snapshots and fetches map updates and hints.
    Runs entirely off the critical path; failures are logged and ignored.
    """
    # Read config with safe defaults
    try:
        cloud = config.get('cloud', {})
    except Exception:
        cloud = {}
    base_url = str(cloud.get('base_url', '') or '')
    token = str(cloud.get('auth_token', '') or '')
    try:
        interval = float(cloud.get('sync_interval_s', 60.0))
    except Exception:
        interval = 60.0
    upload_recent = bool(cloud.get('upload_recent_events', True))
    # Prepare headers
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    # Map and hints cache directories
    map_cache_dir = os.path.join('cache', 'map')
    try:
        os.makedirs(map_cache_dir, exist_ok=True)
    except Exception:
        pass
    hints_cache_dir = os.path.join('cache', 'hints')
    try:
        os.makedirs(hints_cache_dir, exist_ok=True)
    except Exception:
        pass
    while state.running:
        try:
            enabled = bool(cloud.get('enabled', False))
        except Exception:
            enabled = False
        if (not enabled) or (not base_url):
            time.sleep(interval)
            continue
        # Build telemetry snapshot; honor redaction settings and upload_recent flag
        snap = _snapshot_telemetry()
        if snap is not None and (not upload_recent):
            try:
                snap.pop('events_recent', None)
            except Exception:
                pass
        # Upload telemetry
        if snap is not None:
            try:
                import requests
                url = base_url.rstrip('/') + '/api/telemetry'
                requests.post(url, headers=headers, data=json.dumps(snap), timeout=2.0)
            except Exception as e:
                try:
                    logging.debug(f'cloud telemetry upload failed: {e}')
                except Exception:
                    pass
        # Optional: fetch map updates near current position (no impact on core loop)
        try:
            pos = getattr(state, 'positioning', {})
            lat = pos.get('lat')
            lon = pos.get('lon')
            if (lat is not None) and (lon is not None):
                import requests
                mu = base_url.rstrip('/') + f'/api/map/updates?lat={lat}&lon={lon}'
                r = requests.get(mu, headers=headers, timeout=2.0)
                if r.status_code == 200:
                    fname = os.path.join(map_cache_dir, f'updates_{int(time.time())}.json')
                    try:
                        with open(fname, 'wb') as f:
                            f.write(r.content)
                    except Exception:
                        pass
                # Crowd-sourced hints (crosswalks/intersections) near position
                try:
                    if bool(cloud.get('hints_enabled', True)):
                        hu = base_url.rstrip('/') + f'/api/hints?lat={lat}&lon={lon}'
                        hr = requests.get(hu, headers=headers, timeout=2.0)
                        if hr.status_code == 200 and hr.content:
                            hname = os.path.join(hints_cache_dir, f'hints_{int(time.time())}.json')
                            try:
                                with open(hname, 'wb') as f:
                                    f.write(hr.content)
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(interval)


def ota_update_thread():
    """Optional OTA updates for models/config. Safe, staged downloads and hot-reload of YOLO.
    - Checks a remote manifest; if newer YOLO weights available, download, verify, and restart YOLO.
    - Entirely non-blocking; errors are logged and ignored.
    """
    try:
        updates = config.get('updates', {})
    except Exception:
        updates = {}
    base_url = str(updates.get('base_url', '') or '')
    try:
        interval = float(updates.get('check_interval_s', 900.0))
    except Exception:
        interval = 900.0
    auto_apply = bool(updates.get('auto_apply', True))
    token = str(updates.get('auth_token', '') or '')
    headers = {'Accept': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    models_dir = 'models'
    try:
        os.makedirs(models_dir, exist_ok=True)
    except Exception:
        pass
    def _sha256_file(path):
        try:
            import hashlib
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None
    while state.running:
        # Admin controls: defer window and force checks
        try:
            now = time.time()
        except Exception:
            now = time.time()
        try:
            defer_until = float(getattr(state, '_ota_defer_until_ts', 0.0) or 0.0)
        except Exception:
            defer_until = 0.0
        if defer_until > now:
            try:
                time.sleep(min(interval, max(1.0, defer_until - now)))
            except Exception:
                time.sleep(1.0)
            continue
        try:
            force_check = bool(getattr(state, '_ota_force_check', False))
        except Exception:
            force_check = False
        try:
            force_apply = bool(getattr(state, '_ota_force_apply', True))
        except Exception:
            force_apply = True
        try:
            enabled = bool(updates.get('ota_enabled', False))
        except Exception:
            enabled = False
        # Allow forced check regardless of auto_apply; still require base_url
        if (not enabled or not auto_apply) and (not force_check):
            time.sleep(interval)
            continue
        try:
            import requests
            murl = base_url.rstrip('/') + '/ota/manifest.json'
            r = requests.get(murl, headers=headers, timeout=3.0)
            if r.status_code == 200:
                man = r.json()
            else:
                man = None
        except Exception as e:
            man = None
            try:
                logging.debug(f'ota manifest fetch failed: {e}')
            except Exception:
                pass
        try:
            with state.lock:
                setattr(state, '_ota_last_check_ts', time.time())
        except Exception:
            pass
        if not man:
            time.sleep(interval)
            # Clear force flag even on failure to prevent tight loops
            if force_check:
                try:
                    with state.lock:
                        setattr(state, '_ota_force_check', False)
                except Exception:
                    pass
            continue
        # Manifest fields: { yolo_weights_url, yolo_sha256, version }
        yurl = str(man.get('yolo_weights_url', '') or '')
        ysha = str(man.get('yolo_sha256', '') or '')
        if not yurl:
            time.sleep(interval)
            if force_check:
                try:
                    with state.lock:
                        setattr(state, '_ota_force_check', False)
                except Exception:
                    pass
            continue
        # Compare local weights
        try:
            current_weights = str(config.get('yolo', {}).get('weights', 'yolov8n.pt'))
        except Exception:
            current_weights = 'yolov8n.pt'
        local_path = os.path.join(models_dir, os.path.basename(yurl)) if ('/' in yurl) else os.path.join(models_dir, yurl)
        need_download = True
        try:
            if os.path.exists(local_path) and ysha:
                need_download = (_sha256_file(local_path) != ysha)
        except Exception:
            need_download = True
        if need_download:
            try:
                import requests
                rr = requests.get(yurl, headers=headers, timeout=10.0)
                if rr.status_code == 200:
                    tmp = local_path + '.tmp'
                    with open(tmp, 'wb') as f:
                        f.write(rr.content)
                    ok = True
                    if ysha:
                        ok = (_sha256_file(tmp) == ysha)
                    if ok:
                        os.replace(tmp, local_path)
                    else:
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
                        raise RuntimeError('sha256 mismatch for downloaded weights')
                else:
                    raise RuntimeError(f'HTTP {rr.status_code}')
            except Exception as e:
                try:
                    logging.warning(f'ota download failed: {e}')
                except Exception:
                    pass
                time.sleep(interval)
                continue
        # If the downloaded file differs from current weights, swap and restart YOLO
        try:
            if os.path.abspath(local_path) != os.path.abspath(current_weights):
                # Load new YOLO model and update runtime tuple; reuse device and ensemble settings
                from ultralytics import YOLO as _Y
                dev = getattr(state, '_yolo_tuple', (None, select_device(), []))[1]
                new_yolo = _Y(local_path)
                try:
                    new_yolo.fuse(); new_yolo.model.eval()
                except Exception:
                    pass
                if auto_apply or force_apply:
                    with state.lock:
                        try:
                            ens = getattr(state, '_yolo_tuple', (None, dev, []))[2]
                        except Exception:
                            ens = []
                        state._yolo_tuple = (new_yolo, dev, ens)
                        # Update config and global for future loads
                        try:
                            config['yolo']['weights'] = local_path
                        except Exception:
                            pass
                        try:
                            globals()['YOLO_WEIGHTS'] = local_path
                        except Exception:
                            pass
                        # Signal restart; watchdog will relaunch yolo_thread with new tuple
                        state.restart_request['yolo'] = True
                    # Give yolo_thread time to exit; watchdog will clear flag and relaunch
                    time.sleep(2.0)
        except Exception as e:
            try:
                logging.warning(f'ota apply failed: {e}')
            except Exception:
                pass
        # Clear one-shot force flags
        if force_check:
            try:
                with state.lock:
                    setattr(state, '_ota_force_check', False)
            except Exception:
                pass
        time.sleep(interval)


def main():
    # Enable runtime optimizations
    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    logging.info('Loading models...')
    try:
        yolo, device, depth_pipe, ensemble_models = load_models()
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        return
    # Store model references for watchdog restarts
    try:
        with state.lock:
            state._yolo_tuple = (yolo, device, ensemble_models)
            state._depth_pipe = depth_pipe
    except Exception:
        pass
    logging.info(f'YOLO device: {device}')
    logging.info(f'Ensemble models: {len(ensemble_models) if ensemble_models else 0}')
    
    # Initialize optimized frame processor for faster YOLO-Depth interconnection
    try:
        init_optimized_processor(device)
        logging.info('Optimized frame processor initialized')
    except Exception as e:
        logging.warning(f'Failed to initialize optimized processor: {e}')
    
    logging.info('Starting threads...')
    try:
        # Start camera and YOLO threads
        try:
            t_cam = threading.Thread(target=camera_thread, daemon=True)
            t_cam.start()
            state.threads['camera'] = t_cam
        except Exception as e:
            logging.warning(f'Camera thread could not start: {e}')
        t_yolo = threading.Thread(target=yolo_thread, args=(yolo, device, ensemble_models), daemon=True)
        t_yolo.start()
        state.threads['yolo'] = t_yolo

        # Initialize audio backend and start audio workers (async or sync)
        try:
            init_tts_backend()
        except Exception:
            pass
        try:
            async_mode = bool(config.get('audio', {}).get('async_mode', False))
        except Exception:
            async_mode = False
        if async_mode:
            # Create and start a dedicated asyncio event loop for audio
            try:
                loop = asyncio.new_event_loop()
                def _run_loop(l):
                    try:
                        asyncio.set_event_loop(l)
                        l.run_forever()
                    except Exception:
                        pass
                t_loop = threading.Thread(target=_run_loop, args=(loop,), daemon=True)
                t_loop.start()
                with state.lock:
                    state._async_audio_loop = loop
                    state.threads['audio_loop'] = t_loop
            except Exception:
                loop = None
            # Create async audio queue and schedule async worker; fallback to sync worker if needed
            try:
                if loop is not None:
                    fut_q = asyncio.run_coroutine_threadsafe(_make_async_audio_queue(), loop)
                    q = fut_q.result(timeout=2.0)
                    with state.lock:
                        state._async_audio_queue = q
                    fut = asyncio.run_coroutine_threadsafe(async_audio_worker(), loop)
                    with state.lock:
                        state.threads['audio_async_task'] = fut
                else:
                    raise RuntimeError('async audio loop unavailable')
            except Exception:
                # Fallback to synchronous audio worker
                try:
                    t_audio = threading.Thread(target=audio_worker, daemon=True)
                    t_audio.start()
                    with state.lock:
                        state.threads['audio'] = t_audio
                except Exception as e:
                    logging.warning(f'Audio worker fallback failed: {e}')
        else:
            try:
                t_audio = threading.Thread(target=audio_worker, daemon=True)
                t_audio.start()
                with state.lock:
                    state.threads['audio'] = t_audio
            except Exception as e:
                logging.warning(f'Audio worker could not start: {e}')

        # Start announcement thread (respects enable_announce flag)
        try:
            t_ann = threading.Thread(target=announcement_thread, daemon=True)
            t_ann.start()
            with state.lock:
                state.threads['announcement'] = t_ann
        except Exception as e:
            logging.warning(f'Announcement thread could not start: {e}')

        # Start watchdog to monitor and restart threads on stall/death
        try:
            t_watch = threading.Thread(target=watchdog_thread, daemon=True)
            t_watch.start()
            with state.lock:
                state.threads['watchdog'] = t_watch
        except Exception as e:
            logging.warning(f'Watchdog thread could not start: {e}')

        # Start provider monitor and telemetry persistence threads
        try:
            t_prov = threading.Thread(target=_provider_monitor_worker, daemon=True)
            t_prov.start()
            with state.lock:
                state.threads['provider_monitor'] = t_prov
        except Exception:
            pass
        try:
            t_tel = threading.Thread(target=telemetry_persist_thread, daemon=True)
            t_tel.start()
            with state.lock:
                state.threads['telemetry_persist'] = t_tel
        except Exception:
            pass
        
        # Mark system as ready once core threads have started
        with state.lock:
            state.ready = True
        # Audio self-test disabled
    except Exception as e:
        logging.error(f"Thread startup error: {e}")
        return

    try:
        frame_interval = 1.0 / max(1, TARGET_FPS)
        last_display_time = time.time()
        
        while state.running:
            loop_start = time.time()
            frame = None
            with state.lock:
                if state.frame is not None:
                    frame = state.frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Strict 25 FPS display control
            current_time = time.time()
            if current_time - last_display_time < frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            display = draw_ui(frame)
            # Cache JPEG for web preview regardless of headless mode
            try:
                ok, buf = cv2.imencode('.jpg', display, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    with state.lock:
                        state._display_jpeg = buf.tobytes()
                        state._display_jpeg_time = time.time()
            except Exception:
                pass
            if not HEADLESS:
                cv2.imshow('Real-time YOLO + DepthAnythingV2', display)
            last_display_time = current_time
            
            if not HEADLESS:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    state.running = False
                elif key == ord('d'):
                    state.show_depth = not state.show_depth
                    state.show_overlay = False
                elif key == ord('o'):
                    state.show_overlay = not state.show_overlay
                    state.show_depth = False
                elif key == ord('a'):
                    with state.lock:
                        state.enable_announce = not state.enable_announce
                    if state.enable_announce:
                        speak_status('Announcements enabled')
                    else:
                        speak_status('Announcements disabled')
                elif key == ord('v'):
                    with state.lock:
                        state.enable_voice_listen = not state.enable_voice_listen
                    if state.enable_voice_listen:
                        speak_status('Voice mode on')
                        try:
                            t_v = state.threads.get('voice_listener')
                            if t_v is None or not t_v.is_alive():
                                t_v = threading.Thread(target=voice_listener_thread, daemon=True)
                                t_v.start()
                                state.threads['voice_listener'] = t_v
                        except Exception:
                            threading.Thread(target=voice_listener_thread, daemon=True).start()
                        try:
                            t_s = state.threads.get('stt')
                            if t_s is None or not t_s.is_alive():
                                t_s = threading.Thread(target=stt_mic_thread, daemon=True)
                                t_s.start()
                                state.threads['stt'] = t_s
                        except Exception:
                            threading.Thread(target=stt_mic_thread, daemon=True).start()
                    else:
                        speak_status('Voice mode off')
                elif key == ord('c'):
                    with state.lock:
                        state.ai_decider = not state.ai_decider
                    speak_status('AI announcer on' if state.ai_decider else 'AI announcer off')
                elif key == ord('x'):
                    with state.lock:
                        state.context_aware = not state.context_aware
                    speak_status('Context awareness on' if state.context_aware else 'Context awareness off')
                elif key == ord('i'):
                    with state.lock:
                        state.min_importance_score = min(1.0, state.min_importance_score + 0.1)
                    speak_status(f'Importance threshold {state.min_importance_score:.1f}')
                elif key == ord('k'):
                    with state.lock:
                        state.min_importance_score = max(0.1, state.min_importance_score - 0.1)
                    speak_status(f'Importance threshold {state.min_importance_score:.1f}')
                elif key == ord('q'):
                    with state.lock:
                        # Toggle quality levels
                        quality_levels = config["performance"]["quality_levels"]
                        current_quality = state.performance_stats['quality_level']
                        current_idx = quality_levels.index(current_quality) if current_quality in quality_levels else 3
                        next_idx = (current_idx + 1) % len(quality_levels)
                        state.performance_stats['quality_level'] = quality_levels[next_idx]
                    speak_status(f'Quality level {state.performance_stats["quality_level"]:.1f}')
                elif key == ord('p'):
                    # Print performance stats
                    logging.info(f"\n=== Performance Stats ===")
                    logging.info(f"FPS: {state.last_fps:.1f}")
                    logging.info(f"Quality: {state.performance_stats['quality_level']:.1f}")
                    if state.performance_stats['cpu_usage']:
                        logging.info(f"CPU: {np.mean(state.performance_stats['cpu_usage'][-10:]):.1f}%")
                    if state.performance_stats['memory_usage']:
                        logging.info(f"Memory: {np.mean(state.performance_stats['memory_usage'][-10:]):.1f}%")
                    if state.performance_stats['processing_times']:
                        logging.info(f"Processing: {np.mean(state.performance_stats['processing_times'][-10:])*1000:.1f}ms")
                    logging.info("========================")
                elif key == ord('m'):
                    with state.lock:
                        state.provider = 'google' if state.provider == 'apple' else 'apple'
                    speak_status(f"Maps provider {state.provider}")
                elif key == ord('f'):
                    with state.lock:
                        state.fast_depth = not state.fast_depth
                    speak_status('Fast depth on' if state.fast_depth else 'Fast depth off')
                elif key == ord('['):
                    with state.lock:
                        state.max_m = max(state.min_m + 0.1, state.max_m - 0.5)
                elif key == ord(']'):
                    with state.lock:
                        state.max_m = min(100.0, state.max_m + 0.5)
                elif key == ord(';'):
                    with state.lock:
                        state.min_m = max(0.05, state.min_m - 0.05)
                elif key == ord('\''):
                    with state.lock:
                        state.min_m = min(state.max_m - 0.05, state.min_m + 0.05)
                elif key == ord('1'):
                    with state.lock:
                        state.depth_alpha = 0.4
                elif key == ord('2'):
                    with state.lock:
                        state.depth_alpha = 0.6
                elif key == ord('3'):
                    with state.lock:
                        state.depth_alpha = 0.8
            # Additional FPS control if needed
            elapsed = time.time() - loop_start
            if elapsed < frame_interval * 0.8:  # Only sleep if we have time
                time.sleep(frame_interval - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        # Engage centralized shutdown routine for clean exit
        try:
            shutdown_runtime(timeout_s=2.0)
        except Exception:
            # Fallback to simple flag if shutdown routine fails
            state.running = False


def handle_exit(signum, frame):
    logging.info("Received exit signal, shutting down...")
    state.running = False

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Third Eye real-time runtime')
    parser.add_argument('--headless', action='store_true', help='Run without UI window (no imshow)')
    parser.add_argument('--no-announce', action='store_true', help='Disable audio announcements')
    args = parser.parse_args()

    # Apply CLI overrides
    if args.no_announce:
        with state.lock:
            state.enable_announce = False

    # Monkey-patch a headless flag via config to use inside main loop
    HEADLESS = args.headless

    main()
    if not HEADLESS:
        cv2.destroyAllWindows()
    time.sleep(0.5)

def shutdown_runtime(timeout_s: float = 2.0):
    """Cleanly stop threads and close resources to avoid hangs on exit.

    - Sets `state.running = False` to signal loops to exit.
    - Joins all threads in `state.threads` with a timeout.
    - Best-effort cleanup for UI windows.
    """
    try:
        with state.lock:
            state.running = False
    except Exception:
        # Proceed even if lock acquisition fails
        state.running = False

    # Join known threads to ensure clean exit
    try:
        threads = dict(getattr(state, 'threads', {}) or {})
        for name, t in threads.items():
            try:
                if isinstance(t, threading.Thread) and t.is_alive():
                    t.join(timeout=timeout_s)
            except Exception:
                pass
    except Exception:
        pass

    # Close UI windows if any remain
    try:
        if not globals().get('HEADLESS', True):
            cv2.destroyAllWindows()
    except Exception:
        pass

# --- Hints fusion helpers ---
def _haversine_m(lat1, lon1, lat2, lon2):
    try:
        from math import radians, sin, cos, sqrt, atan2
        R = 6371000.0
        dlat = radians((lat2 or 0.0) - (lat1 or 0.0))
        dlon = radians((lon2 or 0.0) - (lon1 or 0.0))
        a = sin(dlat/2)**2 + cos(radians(lat1 or 0.0))*cos(radians(lat2 or 0.0))*sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    except Exception:
        return 1e9

def _read_cached_hints(max_age_s=1800):
    hints = []
    try:
        dirp = os.path.join('cache','hints')
        if not os.path.isdir(dirp):
            return hints
        now = int(time.time())
        for name in sorted(os.listdir(dirp), reverse=True):
            if not name.endswith('.json'):
                continue
            fp = os.path.join(dirp, name)
            try:
                st = os.stat(fp)
                if (now - int(st.st_mtime)) > int(max_age_s):
                    continue
                with open(fp,'rb') as f:
                    import json
                    data = json.loads(f.read().decode('utf-8',errors='ignore') or '{}')
                    if isinstance(data, dict):
                        # normalize single
                        items = data.get('hints') or data.get('items') or []
                        if isinstance(items, dict):
                            items = [items]
                        for it in items:
                            if isinstance(it, dict):
                                hints.append(it)
                    elif isinstance(data, list):
                        hints.extend([it for it in data if isinstance(it, dict)])
            except Exception:
                continue
    except Exception:
        pass
    return hints

def _fuse_hints_into_scene_tags(tags):
    try:
        # Respect local-first: only add hints when local detection is weak
        cfg = config.get('cloud', {})
        if not bool(cfg.get('hints_fusion_enabled', True)):
            return tags
        pos = getattr(state, 'positioning', {})
        lat = pos.get('lat')
        lon = pos.get('lon')
        if (lat is None) or (lon is None):
            return tags
        radius_m = float(cfg.get('hints_radius_m', 60.0))
        max_age_s = int(cfg.get('hints_max_age_s', 1800))
        crowd_tags = set(tags)
        have_crosswalk = 'crosswalk' in crowd_tags
        have_intersection = 'intersection' in crowd_tags
        # If we already detected locally, keep as-is
        if have_crosswalk and have_intersection:
            return tags
        hints = _read_cached_hints(max_age_s=max_age_s)
        nearby_types = set()
        for h in hints:
            try:
                hl = float(h.get('lat'))
                ho = float(h.get('lon'))
                typ = str(h.get('type','')).lower().strip()
                if not typ:
                    continue
                dist = _haversine_m(lat, lon, hl, ho)
                if dist <= radius_m:
                    nearby_types.add(typ)
            except Exception:
                continue
        # Add only what is missing
        if ('crosswalk' in nearby_types) and not have_crosswalk:
            crowd_tags.append('crosswalk')
        if ('intersection' in nearby_types) and not have_intersection:
            crowd_tags.append('intersection')
        return crowd_tags
    except Exception:
        return tags

# --- Alert preferences helpers ---
def _read_alert_prefs() -> dict:
    try:
        base = os.path.join(os.getcwd(), 'cache', 'prefs')
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, 'alerts.json')
        if os.path.isfile(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {
        'global': {
            'urgency_enabled': ['critical', 'warning', 'info'],
            'types_enabled': {'critical': True, 'warning': True, 'clear': True, 'summary': True, 'system': True},
            'sensitivity_scale': 1.0,
            'language': str(config.get('audio', {}).get('tts_language', '') or ''),
            'mute_all': False
        },
        'geofences': []
    }

def _write_alert_prefs(prefs: dict) -> bool:
    try:
        base = os.path.join(os.getcwd(), 'cache', 'prefs')
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, 'alerts.json')
        with open(path, 'w') as f:
            json.dump(prefs, f, indent=2)
        return True
    except Exception:
        return False

def _get_active_alert_prefs() -> dict:
    """Resolve active alert preferences, merging global with most specific geofence (if any)."""
    prefs = _read_alert_prefs()
    out = dict(prefs.get('global', {}))
    try:
        pos = dict(getattr(state, 'positioning', {}) or {})
        lat = pos.get('lat'); lon = pos.get('lon')
        fences = list(prefs.get('geofences', []) or [])
        best = None; best_dist = float('inf')
        if (lat is not None) and (lon is not None):
            for gf in fences:
                try:
                    glat = float(gf.get('lat')); glon = float(gf.get('lon'))
                    rad = float(gf.get('radius_m', 80.0) or 80.0)
                    dist = _haversine_m(lat, lon, glat, glon)
                    if dist <= rad and dist < best_dist:
                        best = gf; best_dist = dist
                except Exception:
                    pass
        if best:
            for k, v in best.items():
                if k in ('lat','lon','radius_m'):
                    continue
                out[k] = v
    except Exception:
        pass
    return out

def _queue_get_latest(q: queue.Queue):
    """Drain a Queue non-blocking and return the latest item, or None.

    Helps synchronize producer-consumer pipelines by preferring freshness
    when consumers lag behind temporarily.
    """
    latest = None
    try:
        while True:
            latest = q.get_nowait()
    except Exception:
        pass
    return latest

