"""Proxy codec: isolates the temporal-decimation + interpolation mechanism.

HONESTY REQUIREMENT -- read before using any output of this module.

This is NOT the LTX-Video VAE. It is a deliberately transparent stand-in that
reproduces two of the nine mechanisms in the Objective-1 taxonomy, and only
those two:

    Mechanism 1 -- temporal decimation (stride along T in the encoder)
    Mechanism 4 -- decoder-side temporal upsampling by interpolation

It contains no learned component, so it cannot reproduce mechanisms 2, 3, 5-9.
Its purpose is to demonstrate that the *mechanism* produces measurable temporal
degradation, and to validate the measurement harness end-to-end before any GPU
time is spent. Every figure derived from it must be labelled

    "proxy codec (temporal decimation x8), NOT the LTX-Video VAE"

Presenting proxy output as a measured VAE floor would be fabrication.
"""
from __future__ import annotations

import cv2
import numpy as np


def temporal_decimate_restore(video: np.ndarray, factor: int = 8,
                              mode: str = "linear") -> np.ndarray:
    """Decimate along time by `factor`, then restore the original length.

    mode:
      "nearest"  -- frame replication (what naive upsampling does)
      "linear"   -- linear blend between retained keyframes (what interpolation
                    based decoder upsampling approximates)

    Note the decimation keeps every `factor`-th frame rather than averaging,
    matching a strided convolution rather than a pooling operation.
    """
    t = video.shape[0]
    keep_idx = np.arange(0, t, factor)
    kept = video[keep_idx].astype(np.float32)

    out = np.empty_like(video, dtype=np.float32)
    for i in range(t):
        pos = i / factor
        lo = int(np.floor(pos))
        hi = min(lo + 1, len(kept) - 1)
        if mode == "nearest":
            out[i] = kept[min(int(round(pos)), len(kept) - 1)]
        else:
            w = pos - lo
            out[i] = (1.0 - w) * kept[lo] + w * kept[hi]
    return np.clip(out, 0, 255).astype(np.uint8)


def spatial_blur_restore(video: np.ndarray, factor: int = 32) -> np.ndarray:
    """Spatial downsample/upsample -- the spatial analogue, for contrast.

    Included so the demo can show that spatial compression at a comparable
    ratio does NOT produce the same temporal signature, which is the point:
    temporal error is its own axis.
    """
    t, h, w, _ = video.shape
    sh, sw = max(1, h // factor), max(1, w // factor)
    out = np.empty_like(video)
    for i, f in enumerate(video):
        small = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)
        out[i] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out
