import cv2
import numpy as np
import time
import threading
import queue
import signal
import os
import logging
from logging.handlers import RotatingFileHandler
import argparse
from typing import List

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import simpleaudio as sa
except Exception:
    sa = None


CONFIG = {
    # Detection thresholds
    "min_conf": 0.35,
    "blocked_center_band_frac": 0.33,  # central band width fraction
    "blocked_near_area_frac": 0.20,     # bbox area fraction considered near/large
    # Announcement controls
    "enqueue_rate_limit_s": 1.0,
    "message_dedup_window_s": 3.0,
    "red_alert_repeat_interval_s": 2.5,
    "global_silence_after_red_s": 2.0,
    # Audio
    "tts_enabled": True,
    "sound_enabled": True,
    "alarm_wav_path": "static/alarm.wav",
    "speech_volume": 0.9,
    # Camera
    "camera_id": 0,
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 20,
    "debug_visualize": False,
    # Logging
    "log_file": "logs/improved_announcer.log",
    "log_level": "INFO",
}


def init_logger():
    logger = logging.getLogger("improved_announcer")
    logger.setLevel(getattr(logging, CONFIG["log_level"], logging.INFO))
    os.makedirs(os.path.dirname(CONFIG["log_file"]), exist_ok=True)
    handler = RotatingFileHandler(CONFIG["log_file"], maxBytes=1_000_000, backupCount=3)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


LOGGER = init_logger()


class Detector:
    def __init__(self):
        self.model = None
        self.running = False

    def load_model(self):
        # Integration point: load your model here (YOLO/TensorFlow/etc.)
        # Example:
        # self.model = yolov8.load(weights)
        raise NotImplementedError("load_model must be implemented with your detection model.")

    def model_infer(self, frame):
        # Integration point: run inference and return detections
        # Expected return: list of dicts {label, conf, bbox:(x1,y1,x2,y2)}
        # Optional extra fields: depth_m
        raise NotImplementedError("model_infer must be implemented with your detection model.")

    @staticmethod
    def estimate_direction(bbox, frame_width):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        third = frame_width / 3.0
        if cx < third:
            return "left"
        elif cx > (2 * third):
            return "right"
        else:
            return "center"

    @staticmethod
    def is_blocking_path(bbox, frame_shape):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        # Central band gate
        band_w = w * CONFIG["blocked_center_band_frac"]
        band_x1 = (w - band_w) / 2.0
        band_x2 = band_x1 + band_w
        cx = (x1 + x2) / 2.0
        center_ok = (band_x1 <= cx <= band_x2)
        # Near/large by area fraction
        area = (x2 - x1) * (y2 - y1)
        frame_area = w * h
        near_ok = (area / frame_area) >= CONFIG["blocked_near_area_frac"]
        return center_ok and near_ok


class Announcer:
    def __init__(self):
        self.queue = queue.Queue(maxsize=64)
        self.running = False
        self.last_message_ts = 0.0
        self.last_key_ts = {}
        self.global_silence_until = 0.0
        self.red_alert_active = False
        self.red_alert_last_ts = 0.0
        self._tts_engine = None
        self._alarm = None
        self._speech_thread = threading.Thread(target=self._speech_worker, name="SpeechWorker", daemon=True)

    def init_audio(self):
        if CONFIG["tts_enabled"] and pyttsx3 is not None:
            try:
                self._tts_engine = pyttsx3.init()
                # Volume in [0,1]
                try:
                    self._tts_engine.setProperty("volume", float(CONFIG["speech_volume"]))
                except Exception:
                    pass
            except Exception as e:
                LOGGER.warning(f"TTS init failed: {e}")
                self._tts_engine = None
        if CONFIG["sound_enabled"] and sa is not None:
            try:
                if os.path.exists(CONFIG["alarm_wav_path"]):
                    self._alarm = sa.WaveObject.from_wave_file(CONFIG["alarm_wav_path"])
            except Exception as e:
                LOGGER.warning(f"Alarm WAV load failed: {e}")
                self._alarm = None

    def start(self):
        self.running = True
        self.init_audio()
        self._speech_thread.start()

    def stop(self):
        self.running = False
        try:
            self.queue.put_nowait({"type": "__shutdown__"})
        except Exception:
            pass
        self._speech_thread.join(timeout=2.0)
        try:
            if self._tts_engine is not None:
                self._tts_engine.stop()
        except Exception:
            pass

    def enqueue(self, message, level="info", key=None):
        now = time.time()
        if now < self.global_silence_until:
            return
        # Global rate limit
        if (now - self.last_message_ts) < CONFIG["enqueue_rate_limit_s"]:
            return
        # Dedup by key within window
        if key is not None:
            last = self.last_key_ts.get(key, 0.0)
            if (now - last) < CONFIG["message_dedup_window_s"]:
                return
            self.last_key_ts[key] = now
        self.last_message_ts = now
        try:
            self.queue.put_nowait({"type": "say", "message": message, "level": level})
        except Exception:
            pass

    def set_red_alert(self, active: bool):
        self.red_alert_active = active
        if active:
            self.global_silence_until = 0.0
        else:
            self.global_silence_until = time.time() + CONFIG["global_silence_after_red_s"]

    def _play_alarm(self):
        if self._alarm is not None:
            try:
                self._alarm.play()
            except Exception:
                pass

    def _speak(self, text):
        if self._tts_engine is None:
            LOGGER.info(f"SAY: {text}")
            return
        try:
            self._tts_engine.say(text)
            self._tts_engine.runAndWait()
        except Exception as e:
            LOGGER.warning(f"TTS speak failed: {e}")

    def _speech_worker(self):
        while self.running:
            try:
                item = self.queue.get(timeout=0.5)
            except Exception:
                item = None
            now = time.time()
            # Red-alert repeating
            if self.red_alert_active and (now - self.red_alert_last_ts) >= CONFIG["red_alert_repeat_interval_s"]:
                self.red_alert_last_ts = now
                self._play_alarm()
                self._speak("Red alert! Path blocked ahead.")
            if not item:
                continue
            if item.get("type") == "__shutdown__":
                break
            if item.get("type") == "say":
                msg = item.get("message", "")
                level = item.get("level", "info")
                if level == "red":
                    self._play_alarm()
                self._speak(msg)


class Controller:
    def __init__(self, detector: Detector, announcer: Announcer):
        self.detector = detector
        self.announcer = announcer
        self.running = False
        self._cap = None
        self._lock = threading.Lock()

    def _init_camera(self):
        cap = cv2.VideoCapture(CONFIG["camera_id"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_height"])
        cap.set(cv2.CAP_PROP_FPS, CONFIG["camera_fps"])
        self._cap = cap

    def start(self):
        self.running = True
        self._init_camera()
        self.announcer.start()

    def stop(self):
        self.running = False
        try:
            self.announcer.stop()
        except Exception:
            pass
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass

    def loop(self):
        LOGGER.info("Controller loop started")
        last_blocked = False
        while self.running:
            ret, frame = (False, None)
            try:
                ret, frame = self._cap.read()
            except Exception:
                pass
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            try:
                dets = self.detector.model_infer(frame)
            except NotImplementedError:
                dets = []
            except Exception as e:
                LOGGER.warning(f"Detection error: {e}")
                dets = []

            h, w = frame.shape[:2]
            blocked_now = False
            for d in dets:
                label = d.get("label", "object")
                conf = float(d.get("conf", 0.0))
                bbox = d.get("bbox")
                if bbox is None or conf < CONFIG["min_conf"]:
                    continue
                direction = Detector.estimate_direction(bbox, w)
                key = f"{label}:{direction}"
                if Detector.is_blocking_path(bbox, frame.shape):
                    blocked_now = True
                    msg = f"URGENT! {label} blocking {direction}."
                    self.announcer.enqueue(msg, level="red", key=key)
                else:
                    msg = f"Warning! {label} {direction}."
                    self.announcer.enqueue(msg, level="warning", key=key)

            # Optional visualization for debugging
            try:
                if CONFIG.get("debug_visualize", False):
                    self.visualize(frame, dets)
            except Exception:
                pass

            # Manage red-alert state
            if blocked_now and not last_blocked:
                LOGGER.info("Red-alert activated")
                self.announcer.set_red_alert(True)
            elif (not blocked_now) and last_blocked:
                LOGGER.info("Red-alert cleared")
                self.announcer.set_red_alert(False)
                self.announcer.enqueue("Path clear.", level="info", key="clear")
            last_blocked = blocked_now

            # Small sleep to reduce CPU usage
            time.sleep(0.01)

    def visualize(self, frame: np.ndarray, detections: List[dict]):
        # Draw boxes and labels for debug; press 'q' to quit
        try:
            for d in detections:
                bbox = d.get("bbox")
                label = d.get("label", "obj")
                conf = float(d.get("conf", 0.0))
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Third Eye - debug', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                LOGGER.info("Quit pressed")
                self.running = False
        except Exception:
            pass


def _install_signals(ctrl: Controller):
    def handler(signum, frame):
        LOGGER.info(f"Signal {signum} received; shutting down...")
        ctrl.stop()
    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    except Exception:
        pass


def main():
    # CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=CONFIG['camera_id'])
    parser.add_argument('--width', type=int, default=CONFIG['camera_width'])
    parser.add_argument('--height', type=int, default=CONFIG['camera_height'])
    parser.add_argument('--fps', type=int, default=CONFIG['camera_fps'])
    parser.add_argument('--debug', action='store_true', help='Enable debug visualization window')
    args = parser.parse_args()

    CONFIG['camera_id'] = int(args.camera)
    CONFIG['camera_width'] = int(args.width)
    CONFIG['camera_height'] = int(args.height)
    CONFIG['camera_fps'] = int(args.fps)
    CONFIG['debug_visualize'] = bool(args.debug)

    det = Detector()
    ann = Announcer()
    ctrl = Controller(det, ann)
    ctrl.start()
    _install_signals(ctrl)
    try:
        ctrl.loop()
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()