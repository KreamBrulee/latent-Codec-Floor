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

from .flow import video_flow
from .video_io import to_gray


@dataclass
class MotionProfile:
    dssim: float           # mean(1 - SSIM) over adjacent frame pairs
    flow_mag: float        # mean optical-flow magnitude (px/frame)
    flow_p95: float        # 95th pct flow magnitude; descriptive
    flow_p99: float        # 99th pct; the stratification statistic (P-3 B4)
    hf_energy: float       # mean |Laplacian| — proxy for fine texture content
    stratum: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def dssim_complexity(video: np.ndarray) -> float:
    """Mean structural dissimilarity between adjacent frames."""
    g = to_gray(video)
    vals = [1.0 - ssim(g[i], g[i + 1], data_range=1.0) for i in range(len(g) - 1)]
    return float(np.mean(vals))


def flow_stats(video: np.ndarray) -> tuple[float, float, float]:
    """Mean, 95th and 99th percentile optical-flow magnitude, px per frame."""
    m = np.linalg.norm(video_flow(video), axis=-1)
    return float(m.mean()), float(np.percentile(m, 95)), float(np.percentile(m, 99))


def hf_energy(video: np.ndarray) -> float:
    """Mean absolute Laplacian response — high for fine/detailed texture."""
    g = (to_gray(video) * 255).astype(np.uint8)
    return float(np.mean([np.abs(cv2.Laplacian(f, cv2.CV_32F)).mean() for f in g]))


def profile(video: np.ndarray) -> MotionProfile:
    mean_mag, p95, p99 = flow_stats(video)
    return MotionProfile(
        dssim=dssim_complexity(video),
        flow_mag=mean_mag,
        flow_p95=p95,
        flow_p99=p99,
        hf_energy=hf_energy(video),
    )


def strata_cutpoints(profiles: list[MotionProfile]) -> dict:
    """Cut-points for equal-thirds stratification (DECISIONS.md, P-3 B4).

    * fast_action  -- top third of the corpus by p99 flow magnitude. A tail
                      statistic because fast motion is often local: mean flow
                      is diluted by static background (Q-6), and p_q only sees
                      motion covering more than (100-q)% of the frame, so p99
                      detects fast motion over >= 1% of it.
    * fine_texture -- of the rest, the half with higher Laplacian energy.
    * slow_pan     -- the remainder. (Name kept for continuity; the stratum
                      is "low motion, low texture", not only pans.)

    Equal thirds give each stratum the same N, which maximises the weakest
    stratum's sample for FVMD (Q-4); a fixed quantile such as the former 0.70
    left stratum sizes to chance. Cut-points are corpus-relative because
    flow magnitude is resolution-dependent; record them in the run manifest.
    """
    if len(profiles) < 3:
        raise ValueError(f"need at least 3 clips to form thirds, got {len(profiles)}")
    p99 = np.array([p.flow_p99 for p in profiles])
    fast_cut = float(np.quantile(p99, 2 / 3))
    rest = np.array([p.hf_energy for p in profiles if p.flow_p99 < fast_cut])
    return {
        "statistic": "flow_p99",
        "fast_action_flow_p99_cut": fast_cut,
        "fine_texture_hf_cut": float(np.median(rest)),
    }


def assign_strata(profiles: list[MotionProfile],
                  cutpoints: dict | None = None) -> dict:
    """Label each profile in place; return the cut-points used.

    Pass frozen `cutpoints` (C-3) to stratify new clips against a recorded
    corpus rather than recomputing, which would not be reproducible.
    """
    cut = cutpoints or strata_cutpoints(profiles)
    for p in profiles:
        if p.flow_p99 >= cut["fast_action_flow_p99_cut"]:
            p.stratum = "fast_action"
        elif p.hf_energy >= cut["fine_texture_hf_cut"]:
            p.stratum = "fine_texture"
        else:
            p.stratum = "slow_pan"
    return cut
