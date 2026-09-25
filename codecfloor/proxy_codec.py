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

from .codec import Codec


class ProxyCodec(Codec):
    """Temporal decimation + interpolation behind the Codec interface.

    mode:
      "nearest"  -- frame replication (what naive upsampling does)
      "linear"   -- linear blend between retained keyframes (what interpolation
                    based decoder upsampling approximates)

    The latent is the kept keyframes. Decimation keeps every `factor`-th frame
    rather than averaging, matching a strided convolution rather than a
    pooling operation.

    Requires T = factor*k + 1, the same contract as LTX-Video. Otherwise frames
    after the last keyframe have nothing to interpolate towards and would be
    held static: a boundary artefact that inflated the original Table III
    (DECISIONS.md, P-4 B1). The contract also lets decode recover T from the
    keyframe count alone.
    """

    name = "proxy"
    MODES = ("linear", "nearest")

    def __init__(self, factor: int = 8, mode: str = "linear"):
        # Any unrecognised mode used to fall through to linear silently.
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.factor, self.mode = factor, mode

    def config(self) -> dict:
        return {"codec": self.name, "factor": self.factor, "mode": self.mode,
                "label": f"proxy codec (temporal decimation x{self.factor}), NOT the LTX-Video VAE"}

    def encode(self, video: np.ndarray) -> np.ndarray:
        t = video.shape[0]
        if t < self.factor + 1 or (t - 1) % self.factor:
            raise ValueError(f"need T = {self.factor}k+1 frames (k >= 1), got {t}")
        return video[::self.factor].copy()

    def decode(self, latent: np.ndarray) -> np.ndarray:
        kept = latent.astype(np.float32)
        t = self.factor * (len(kept) - 1) + 1
        out = np.empty((t, *kept.shape[1:]), dtype=np.float32)
        for i in range(t):
            pos = i / self.factor
            lo = int(np.floor(pos))
            hi = min(lo + 1, len(kept) - 1)
            if self.mode == "nearest":
                # floor(pos + 1/2): Python's round() is half-to-even, which
                # sent frame 4 back to keyframe 0 but frame 12 forward to 16.
                out[i] = kept[min(int(np.floor(pos + 0.5)), len(kept) - 1)]
            else:
                w = pos - lo
                out[i] = (1.0 - w) * kept[lo] + w * kept[hi]
        # Round, as a real decoder's uint8 conversion does; truncation biased
        # blended frames by ~-0.3 levels and moved slow-pan warping error by
        # 4 points (DECISIONS.md, P-3 B2).
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def temporal_decimate_restore(video: np.ndarray, factor: int = 8,
                              mode: str = "linear") -> np.ndarray:
    """Round trip through ProxyCodec. Kept for existing callers."""
    return ProxyCodec(factor, mode).round_trip(video)


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
