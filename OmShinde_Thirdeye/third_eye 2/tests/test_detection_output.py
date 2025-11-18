import time
import types
import pytest

import main as rd


class _MockBoxes:
    def __init__(self, confs):
        import torch
        self.conf = torch.tensor(confs, dtype=torch.float32)

    def __len__(self):
        return len(self.conf)


class _MockResult:
    def __init__(self, confs):
        self.boxes = _MockBoxes(confs)


def test_combine_detections_picks_highest_confidence():
    det_low = _MockResult([0.2, 0.3])
    det_mid = _MockResult([0.5])
    det_high = _MockResult([0.85, 0.7])

    best = rd.combine_detections([det_low, det_mid, det_high])
    assert best is det_high, "Expected detection with highest max(conf) selected"


def test_combine_detections_handles_empty_and_none():
    assert rd.combine_detections([]) is None
    # Detection with no boxes should be ignored in voting
    class _Empty:
        boxes = None
    assert rd.combine_detections([_Empty()]) is None