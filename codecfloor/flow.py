"""Dense optical flow: the single source for every flow-derived number.

Warping error, motion energy (paper §V: mean flow magnitude) and motion
stratification all read flow from here, so they cannot silently disagree on
settings (DECISIONS.md, P-3 B6).

The estimator is RAFT-large, chosen by endpoint error against exact ground
truth under a rule fixed before the comparison ran (P-3 B5; record
runs/flow_benchmark_20260925T091042Z.json). torch is imported inside the
functions, so the package still imports without it (D-6); computing flow
does need it.
"""
from __future__ import annotations

import cv2
import numpy as np

ESTIMATOR = "raft_large"

# RAFT computes flow on a 1/8-resolution feature grid and upsamples; the
# outermost 8px cell has no outward neighbours, so its flow is extrapolated.
BORDER = 8

# Pairs per forward pass. Any value gives the same flow to <2e-4 px with TF32
# off (measured 2026-09-25); fixed anyway so results never depend on VRAM.
CHUNK = 8

# The OpenCV dense-flow tutorial values. Kept as the benchmark baseline, not
# used by the metrics (mean EPE 0.123 px vs RAFT-large 0.073).
FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                 poly_n=5, poly_sigma=1.2, flags=0)

_model = None


def farneback(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(H, W, 2) Farneback flow from a to b. Inputs uint8 grayscale."""
    return cv2.calcOpticalFlowFarneback(a, b, None, **FARNEBACK)


def _raft():
    global _model
    if _model is None:
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        weights = Raft_Large_Weights.DEFAULT
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = (raft_large(weights=weights).eval().to(device), weights, device)
    return _model


def video_flow(video: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) uint8 RGB -> (T-1, H, W, 2) float32 flow, frame t -> t+1."""
    import torch

    t, h, w, _ = video.shape
    if h % 8 or w % 8:
        # Padding would change the flow near the edges; the corpus is
        # normalised to multiples of 32 (LTX) anyway, so refuse instead.
        raise ValueError(f"RAFT needs H and W divisible by 8, got {h}x{w}")
    model, weights, device = _raft()
    transform = weights.transforms()
    x = torch.from_numpy(video).permute(0, 3, 1, 2)

    # TF32 convolutions made the flow depend on batch size (up to 0.06 px);
    # full fp32 removes that. Restored afterwards: other torch code is not ours.
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = []
        with torch.no_grad():
            for s in range(0, t - 1, CHUNK):
                e = min(s + CHUNK, t - 1)
                a, b = transform(x[s:e], x[s + 1:e + 1])
                out.append(model(a.to(device), b.to(device))[-1].float().cpu())
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved
    return torch.cat(out).permute(0, 2, 3, 1).numpy()


def config() -> dict:
    """Flow settings for the run manifest."""
    import torch
    import torchvision

    _, weights, device = _raft()
    return {"estimator": ESTIMATOR, "weights": str(weights), "device": device,
            "tf32": False, "chunk": CHUNK, "border": BORDER,
            "torch": torch.__version__, "torchvision": torchvision.__version__}
