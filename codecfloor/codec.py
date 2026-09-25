"""The Codec interface: the one seam between the measurement pipeline and a model.

ProxyCodec and LTXCodec are peers behind it (D-3). Everything downstream sees
only (T, H, W, 3) uint8 RGB in and out, which is what lets the full pipeline be
validated on CPU with the proxy and then run unchanged on the real codec.

This module must stay importable without torch (D-6); implementations import
their heavy dependencies inside methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Codec(ABC):
    name: str

    @abstractmethod
    def encode(self, video: np.ndarray) -> Any:
        """(T, H, W, 3) uint8 RGB -> implementation-defined latent."""

    @abstractmethod
    def decode(self, latent: Any) -> np.ndarray:
        """Latent -> (T, H, W, 3) uint8 RGB.

        Separate from encode because the generation arm decodes latents that
        were never encoded, and the paper's R1 requires the decode path to be
        isolable.
        """

    @abstractmethod
    def config(self) -> dict:
        """Every setting that affects output, for the run manifest."""

    def round_trip(self, video: np.ndarray) -> np.ndarray:
        # The pixel contract is enforced here rather than in each codec: a
        # reconstruction with a different frame count or size would still
        # produce metrics, just on misaligned frames.
        if video.dtype != np.uint8 or video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"expected (T,H,W,3) uint8, got {video.shape} {video.dtype}")
        out = self.decode(self.encode(video))
        if out.shape != video.shape or out.dtype != np.uint8:
            raise ValueError(
                f"{self.name}: round trip changed the video from {video.shape} uint8 "
                f"to {out.shape} {out.dtype}"
            )
        return out
