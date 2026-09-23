"""Synthetic clips spanning the three motion strata.

Used to validate the measurement harness before real data is available, and to
provide reproducible test fixtures. Real evaluation clips replace these; these
exist so the harness can be unit-tested deterministically.
"""
from __future__ import annotations

import cv2
import numpy as np

rng = np.random.default_rng(0)


def _texture_bg(h: int, w: int, seed: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed)
    base = r.integers(0, 255, size=(h // 4, w // 4, 3), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def slow_pan(t: int = 32, h: int = 256, w: int = 256, speed: float = 0.6) -> np.ndarray:
    """Large smooth scene translating slowly. Low temporal frequency."""
    canvas = _texture_bg(h * 2, w * 2, seed=1)
    canvas = cv2.GaussianBlur(canvas, (21, 21), 0)
    frames = []
    for i in range(t):
        x = int(i * speed)
        frames.append(canvas[h // 2:h // 2 + h, x:x + w].copy())
    return np.stack(frames)


def fast_action(t: int = 32, h: int = 256, w: int = 256) -> np.ndarray:
    """Several objects moving quickly. High temporal frequency."""
    bg = cv2.GaussianBlur(_texture_bg(h, w, seed=2), (31, 31), 0)
    objs = [
        {"p": np.array([40.0, 60.0]), "v": np.array([9.0, 5.0]), "r": 22, "c": (240, 60, 60)},
        {"p": np.array([200.0, 90.0]), "v": np.array([-7.0, 8.0]), "r": 18, "c": (60, 240, 90)},
        {"p": np.array([120.0, 200.0]), "v": np.array([6.0, -9.0]), "r": 26, "c": (70, 90, 250)},
    ]
    frames = []
    for _ in range(t):
        f = bg.copy()
        for o in objs:
            o["p"] += o["v"]
            for k in (0, 1):
                lim = (w, h)[k] - o["r"]
                if o["p"][k] < o["r"] or o["p"][k] > lim:
                    o["v"][k] *= -1
                    o["p"][k] = np.clip(o["p"][k], o["r"], lim)
            cv2.circle(f, (int(o["p"][0]), int(o["p"][1])), o["r"], o["c"], -1)
        frames.append(f)
    return np.stack(frames)


def fine_texture(t: int = 32, h: int = 256, w: int = 256) -> np.ndarray:
    """Dense high-frequency detail in modest motion -- foliage / water analogue."""
    canvas = _texture_bg(h * 2, w * 2, seed=3)
    frames = []
    for i in range(t):
        dx, dy = int(2.2 * i), int(1.3 * i)
        crop = canvas[dy:dy + h, dx:dx + w].copy()
        # small per-frame jitter, as leaves/water surfaces have
        noise = (np.random.default_rng(100 + i).normal(0, 6, crop.shape)).astype(np.float32)
        frames.append(np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    return np.stack(frames)


CLIPS = {
    "slow_pan": slow_pan,
    "fast_action": fast_action,
    "fine_texture": fine_texture,
}
