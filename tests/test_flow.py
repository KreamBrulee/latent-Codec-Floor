"""P-3 B5: the flow estimator. Skipped where torch is absent (core-only install)."""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from codecfloor import flow, synthetic  # noqa: E402


def _pan(speed: float, t: int = 5) -> tuple[np.ndarray, np.ndarray]:
    canvas = synthetic._texture_bg(512, 512, seed=4)
    vx, vy = speed * math.cos(math.radians(30)), speed * math.sin(math.radians(30))
    clip = np.stack([synthetic._window(canvas, 32 + i * vx, 32 + i * vy, 128, 128) for i in range(t)])
    return clip, np.array([-vx, -vy])


def test_accuracy_on_known_pan():
    # Regression guard on the estimator choice: the benchmark measured
    # 0.10 px EPE at 2.5 px/frame; a wrong model, weights or preprocessing
    # lands far above the bound.
    clip, gt = _pan(2.5)
    f = flow.video_flow(clip)[:, flow.BORDER:-flow.BORDER, flow.BORDER:-flow.BORDER]
    assert np.linalg.norm(f - gt, axis=-1).mean() < 0.25


def test_result_does_not_depend_on_chunking(monkeypatch):
    clip, _ = _pan(5.0, t=9)
    ref = flow.video_flow(clip)
    monkeypatch.setattr(flow, "CHUNK", 3)
    assert np.abs(flow.video_flow(clip) - ref).max() < 1e-3


def test_tf32_flags_are_restored():
    torch.backends.cudnn.allow_tf32 = True
    flow.video_flow(_pan(1.0, t=2)[0])
    assert torch.backends.cudnn.allow_tf32 is True


def test_rejects_sizes_raft_would_pad():
    with pytest.raises(ValueError, match="divisible by 8"):
        flow.video_flow(np.zeros((2, 100, 128, 3), np.uint8))
