"""Motion stratification.

Implements the content-complexity proxy validated in DLFR-VAE (Yuan et al.,
ACM MM 2025): the mean structural dissimilarity between adjacent frames,

    C(S) = mean_j [ 1 - SSIM(x_j, x_{j+1}) ]

which they show correlates with both the effective temporal frequency of the
clip and with VAE reconstruction error. We use it to bin the evaluation corpus
into motion strata so the codec floor can be reported per motion domain rather
than as a single aggregate number.

We additionally report mean optical-flow magnitude, which is more directly
interpretable as "how much motion" and is used to disambiguate clips that have
high DSSIM for reasons other than motion (e.g. exposure flicker).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .video_io import to_gray


@dataclass
class MotionProfile:
    dssim: float           # mean(1 - SSIM) over adjacent frame pairs
    flow_mag: float        # mean optical-flow magnitude (px/frame)
    flow_p95: float        # 95th pct flow magnitude — catches localised fast motion
    hf_energy: float       # mean |Laplacian| — proxy for fine texture content
    stratum: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def dssim_complexity(video: np.ndarray) -> float:
    """Mean structural dissimilarity between adjacent frames."""
    g = to_gray(video)
    vals = [1.0 - ssim(g[i], g[i + 1], data_range=1.0) for i in range(len(g) - 1)]
    return float(np.mean(vals))


def flow_stats(video: np.ndarray, stride: int = 1) -> tuple[float, float]:
    """Mean and 95th-percentile optical-flow magnitude, in pixels per frame."""
    g = (to_gray(video) * 255).astype(np.uint8)
    mags = []
    for i in range(0, len(g) - 1, stride):
        flow = cv2.calcOpticalFlowFarneback(
            g[i], g[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mags.append(np.linalg.norm(flow, axis=-1))
    m = np.concatenate([x.ravel() for x in mags])
    return float(m.mean()), float(np.percentile(m, 95))


def hf_energy(video: np.ndarray) -> float:
    """Mean absolute Laplacian response — high for fine/detailed texture."""
    g = (to_gray(video) * 255).astype(np.uint8)
    return float(np.mean([np.abs(cv2.Laplacian(f, cv2.CV_32F)).mean() for f in g]))


def profile(video: np.ndarray) -> MotionProfile:
    mean_mag, p95 = flow_stats(video)
    return MotionProfile(
        dssim=dssim_complexity(video),
        flow_mag=mean_mag,
        flow_p95=p95,
        hf_energy=hf_energy(video),
    )


def assign_strata(profiles: list[MotionProfile],
                  texture_quantile: float = 0.70) -> list[MotionProfile]:
    """Assign each clip to slow_pan / fast_action / fine_texture.

    Rule, applied in order:
      * fast_action  — flow magnitude in the top tercile of the corpus
      * fine_texture — not fast, but high-frequency spatial energy above
                       `texture_quantile` (detailed content with modest global
                       motion: foliage, water, crowds, hair)
      * slow_pan     — everything else

    Terciles are computed over the corpus rather than fixed thresholds, because
    absolute flow magnitude is resolution-dependent. Record the cut points in
    the run manifest so the stratification is reproducible.
    """
    if not profiles:
        return profiles
    flow = np.array([p.flow_mag for p in profiles])
    hf = np.array([p.hf_energy for p in profiles])
    fast_cut = np.quantile(flow, 2 / 3)
    tex_cut = np.quantile(hf, texture_quantile)
    for p in profiles:
        if p.flow_mag >= fast_cut:
            p.stratum = "fast_action"
        elif p.hf_energy >= tex_cut:
            p.stratum = "fine_texture"
        else:
            p.stratum = "slow_pan"
    return profiles


def strata_cutpoints(profiles: list[MotionProfile],
                     texture_quantile: float = 0.70) -> dict:
    flow = np.array([p.flow_mag for p in profiles])
    hf = np.array([p.hf_energy for p in profiles])
    return {
        "fast_action_flow_cut": float(np.quantile(flow, 2 / 3)),
        "fine_texture_hf_cut": float(np.quantile(hf, texture_quantile)),
        "texture_quantile": texture_quantile,
    }
