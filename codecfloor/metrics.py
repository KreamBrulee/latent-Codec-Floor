"""Reference-free temporal metrics.

Design note (important, and worth stating in the report):

The floor path has ground truth (original vs reconstruction) but the generation
path does not -- generated video has no corresponding real clip. Any metric used
for BOTH paths must therefore be *reference-free*: computed from a single video,
not from a pair. Otherwise the two measurements are not on a common scale and the
subtraction E_total - E_codec is meaningless.

`warping_error` is a *secondary* diagnostic (D-2); FVMD is primary. It
estimates optical flow within the video itself, uses it to predict each frame
from its predecessor, and reports the residual. A temporally smooth video is
well predicted by its own motion field; a flickering or drifting one is not.
It also improves when motion is removed (finding F-2), so it must always be
read alongside a measure of motion content, never on its own.

Paired metrics (PSNR / SSIM against ground truth) are still computed on the floor
path, but only as a *spatial counterweight* -- they never enter the decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .flow import BORDER, video_flow
from .video_io import to_gray


@dataclass
class TemporalMetrics:
    warp_error: float        # motion-compensated residual, [0,1] scale
    warp_error_p95: float    # 95th pct over frame pairs; descriptive, gates nothing
    raw_frame_diff: float    # mean |I_t - I_{t+1}| with no motion compensation
    flow_mag: float          # mean flow magnitude, for context

    def as_dict(self) -> dict:
        return asdict(self)


def _warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp `img` by `flow` using remap."""
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (gx + flow[..., 0]).astype(np.float32)
    map_y = (gy + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def warping_error_per_pair(video: np.ndarray, flows: np.ndarray | None = None,
                           border: int = BORDER) -> np.ndarray:
    """Per-frame-pair motion-compensated residual. Returns array of length T-1.

    For each pair (t, t+1): estimate flow t -> t+1, warp frame t+1 back onto the
    t grid, and take the mean absolute difference against frame t. A border is
    excluded because flow is unreliable at image edges.
    """
    if flows is None:
        flows = video_flow(video)
    gf = to_gray(video)
    errs = []
    for i, fl in enumerate(flows):
        warped = _warp(gf[i + 1], fl)
        d = np.abs(warped - gf[i])
        if border > 0:
            d = d[border:-border, border:-border]
        errs.append(float(d.mean()))
    return np.array(errs)


def raw_frame_diff(video: np.ndarray) -> float:
    g = to_gray(video)
    return float(np.mean(np.abs(np.diff(g, axis=0))))


def mean_flow_magnitude(video: np.ndarray, flows: np.ndarray | None = None) -> float:
    """Motion energy as the paper defines it (§V): mean flow magnitude."""
    if flows is None:
        flows = video_flow(video)
    return float(np.linalg.norm(flows, axis=-1).mean())


def temporal_metrics(video: np.ndarray) -> TemporalMetrics:
    flows = video_flow(video)  # once: RAFT is the expensive part
    errs = warping_error_per_pair(video, flows)
    return TemporalMetrics(
        warp_error=float(errs.mean()),
        warp_error_p95=float(np.percentile(errs, 95)),
        raw_frame_diff=raw_frame_diff(video),
        flow_mag=mean_flow_magnitude(video, flows),
    )


# ---------------------------------------------------------------- counterweight
def recon_psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Paired spatial fidelity. Floor path only -- never enters the decomposition."""
    x = a.astype(np.float32) / 255.0
    y = b.astype(np.float32) / 255.0
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))
