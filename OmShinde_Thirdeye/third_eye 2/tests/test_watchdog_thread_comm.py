import time
import threading
import queue
import pytest

import main as rd


def _stub_yolo_thread(yolo, device, ensemble_models=None):
    # Simulate brief activity and exit
    rd.state.last_yolo_time = time.time()
    time.sleep(0.01)


def _stub_depth_thread(depth_pipe):
    rd.state._last_depth_time = time.time()
    time.sleep(0.01)


def _stub_shortcuts_server_thread():
    time.sleep(0.01)


def _stub_announcement_thread():
    rd.state._last_announce_tick = time.time()
    time.sleep(0.01)


def _stub_audio_worker():
    rd.state._last_audio_tick = time.time()
    time.sleep(0.01)


@pytest.fixture
def patched_threads(monkeypatch):
    # Patch thread targets to light stubs
    monkeypatch.setattr(rd, 'yolo_thread', _stub_yolo_thread)
    monkeypatch.setattr(rd, 'depth_thread', _stub_depth_thread)
    monkeypatch.setattr(rd, 'shortcuts_server_thread', _stub_shortcuts_server_thread)
    monkeypatch.setattr(rd, 'announcement_thread', _stub_announcement_thread)
    monkeypatch.setattr(rd, 'audio_worker', _stub_audio_worker)

    # Provide minimal tuple and pipe for restarts
    rd.state._yolo_tuple = (None, None, None)
    rd.state._depth_pipe = lambda img: {'depth': None}

    # Fresh audio queue
    rd.audio_queue = queue.PriorityQueue()

    # Configure aggressive stall thresholds
    rd.config.setdefault('watchdog', {})
    rd.config['watchdog']['yolo_stall_s'] = 0.2
    rd.config['watchdog']['depth_stall_s'] = 0.2
    rd.config['watchdog']['announce_stall_s'] = 0.2
    rd.config['watchdog']['audio_stall_s'] = 0.2
    rd.config['watchdog']['min_restart_interval_s'] = 0.1
    rd.config['watchdog']['camera_stall_s'] = 0.2

    # Ensure liveness timestamps are old to trigger restarts
    rd.state.last_yolo_time = 0.0
    rd.state._last_depth_time = 0.0
    rd.state._last_announce_tick = 0.0
    rd.state._last_audio_tick = 0.0
    yield


def test_watchdog_initializes_and_restarts_threads(patched_threads):
    # Run watchdog briefly
    rd.state.running = True
    t_wd = threading.Thread(target=rd.watchdog_thread, daemon=True)
    t_wd.start()
    # Allow a couple of cycles
    time.sleep(0.8)
    # Verify threads created/replaced by watchdog
    assert isinstance(rd.state.threads.get('yolo'), threading.Thread), "YOLO thread should be initialized by watchdog"
    assert isinstance(rd.state.threads.get('depth'), threading.Thread), "Depth thread should be initialized by watchdog"
    assert isinstance(rd.state.threads.get('announcement'), threading.Thread), "Announcement thread should be initialized by watchdog"
    assert isinstance(rd.state.threads.get('audio'), threading.Thread), "Audio worker thread should be initialized by watchdog"
    # Stop watchdog
    rd.state.running = False
    t_wd.join(timeout=1.0)