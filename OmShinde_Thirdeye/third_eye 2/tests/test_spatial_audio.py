import time
import queue
import pytest

import main as rd


@pytest.fixture(autouse=True)
def reset_audio_queue_and_config():
    # Fresh queue and timing
    rd.audio_queue = queue.PriorityQueue()
    rd._last_enqueued_time = 0.0
    # Configure audio knobs for predictable behavior
    rd.config.setdefault('audio', {})
    rd.config['audio']['enqueue_rate_limit_s'] = 0.2
    rd.config['audio']['max_queue_size'] = 4
    rd.config['audio']['speech_rate_default'] = 0.5
    # Disable earcons to avoid side effects
    rd.state._earcons_enabled = False
    rd.state._global_silence_until = 0.0
    # Clear dedup history
    rd.state._recent_messages = {}
    yield


def test_warning_message_segmented_with_prefix_and_tail():
    rd.speak_status('Warning! Low clearance ahead, duck.')
    # We expect at least the two segments; there may be additional messages from formatting
    assert rd.audio_queue.qsize() >= 2, "Warning should enqueue at least two segments"
    # Read underlying heap list to avoid comparison errors on equal priorities
    items = [tpl[2] for tpl in list(rd.audio_queue.queue)]
    prefix = next((m for m in items if m['msg'].lower() == 'warning!'), None)
    tail = next((m for m in items if m['msg'].lower() == 'low clearance ahead, duck.'), None)
    assert prefix is not None, "Prefix segment should be present"
    assert tail is not None, "Tail segment should be present"
    assert prefix['urgent'] is True, "Prefix segment should be urgent"
    assert abs(prefix['rate'] - 0.56) < 1e-3, "Warning prefix rate should be ~0.56"
    assert tail['urgent'] is False, "Tail segment should not be urgent"


def test_enqueue_rate_limit_blocks_non_urgent_repeats():
    rd.speak_status('Status update.')
    rd.speak_status('Status update.')
    # Non-urgent rate limit should drop immediate repeat
    assert rd.audio_queue.qsize() == 1


def test_queue_overflow_drops_excess_non_urgent():
    rd.config['audio']['max_queue_size'] = 1
    rd.speak_status('First message')
    rd.speak_status('Second message')
    assert rd.audio_queue.qsize() == 1, "Second enqueue should be dropped at capacity"