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


def _window(canvas: np.ndarray, x: float, y: float, w: int, h: int) -> np.ndarray:
    """(h, w) window of `canvas` at sub-pixel offset (x, y).

    Integer-truncated offsets turn a uniform pan into a stutter (repeated
    frames, then 1px jumps), which trivialises the baseline warping error
    (DECISIONS.md, P-4 B5). Lanczos-4 is local, so no wrap-around ringing.
    """
    m = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(canvas, m, (w, h), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


def slow_pan(t: int = 33, h: int = 256, w: int = 256, speed: float = 0.6) -> np.ndarray:
    """Large smooth scene translating slowly. Low temporal frequency."""
    canvas = _texture_bg(h * 2, w * 2, seed=1)
    canvas = cv2.GaussianBlur(canvas, (21, 21), 0)
    frames = []
    for i in range(t):
        frames.append(_window(canvas, i * speed, h // 2, w, h))
    return np.stack(frames)


def ball_tracks(t: int = 33, h: int = 256, w: int = 256) -> list[list[tuple[int, int, int]]]:
    """Per-frame (cx, cy, r) of each fast_action ball, as rendered.

    The renderer draws from these, so they are exact ground truth for the
    flow benchmark: a ball pixel at frame i moves by the change in its
    ball's centre (texture is rigidly attached).
    """
    objs = [
        {"p": np.array([40.0, 60.0]), "v": np.array([9.0, 5.0]), "r": 22},
        {"p": np.array([200.0, 90.0]), "v": np.array([-7.0, 8.0]), "r": 18},
        {"p": np.array([120.0, 200.0]), "v": np.array([6.0, -9.0]), "r": 26},
    ]
    tracks = []
    for _ in range(t):
        frame = []
        for o in objs:
            o["p"] += o["v"]
            for k in (0, 1):
                lim = (w, h)[k] - o["r"]
                if o["p"][k] < o["r"] or o["p"][k] > lim:
                    o["v"][k] *= -1
                    o["p"][k] = np.clip(o["p"][k], o["r"], lim)
            frame.append((int(o["p"][0]), int(o["p"][1]), o["r"]))
        tracks.append(frame)
    return tracks


_BALL_COLOURS = [(240, 60, 60), (60, 240, 90), (70, 90, 250)]


def fast_action(t: int = 33, h: int = 256, w: int = 256,
                textured: bool = True) -> np.ndarray:
    """Several objects moving quickly. High temporal frequency.

    Each ball carries its own texture, moving with it. Flat-coloured discs
    have no interior structure to track: Farneback recovered 35% of their
    true displacement against 80% when textured (Q-11, P-3 B1), so a flat
    fixture measures the flow estimator, not the codec. `textured=False`
    reproduces the old appearance; trajectories are identical either way.
    """
    bg = cv2.GaussianBlur(_texture_bg(h, w, seed=2), (31, 31), 0)
    skins = [_texture_bg(64, 64, seed=10 + k) for k in range(3)]
    yy, xx = np.mgrid[:h, :w]
    frames = []
    for balls in ball_tracks(t, h, w):
        f = bg.copy()
        for k, (cx, cy, r) in enumerate(balls):
            if textured:
                m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
                # Texture is indexed relative to the ball centre, so it moves
                # rigidly with the ball.
                tex = skins[k][(yy[m] - cy) % 64, (xx[m] - cx) % 64]
                f[m] = (0.5 * tex + 0.5 * np.array(_BALL_COLOURS[k])).astype(np.uint8)
            else:
                cv2.circle(f, (cx, cy), r, _BALL_COLOURS[k], -1)
        frames.append(f)
    return np.stack(frames)


def fine_texture(t: int = 33, h: int = 256, w: int = 256) -> np.ndarray:
    """Dense high-frequency detail in modest motion -- foliage / water analogue."""
    canvas = _texture_bg(h * 2, w * 2, seed=3)
    frames = []
    for i in range(t):
        crop = _window(canvas, 2.2 * i, 1.3 * i, w, h)
        # small per-frame jitter, as leaves/water surfaces have
        noise = (np.random.default_rng(100 + i).normal(0, 6, crop.shape)).astype(np.float32)
        frames.append(np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    return np.stack(frames)


CLIPS = {
    "slow_pan": slow_pan,
    "fast_action": fast_action,
    "fine_texture": fine_texture,
}
